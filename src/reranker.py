"""
reranker.py — sentence/paragraph-level reranker для точной подсветки источников.

Использует cross-encoder модель которая принимает пару (вопрос, фрагмент)
и возвращает score релевантности. Это принципиально точнее любых эвристик:
модель понимает семантику, падежи, синонимы, аббревиатуры.

Модель: cross-encoder/ms-marco-MiniLM-L-6-v2
  — обучена на MS MARCO (passage retrieval)
  — понимает русский через multilingual токенизатор
  — размер ~80MB, CPU-inference ~5ms на фрагмент
  — альтернатива для лучшего русского: amberoad/bert-multilingual-passage-reranking-msmarco

Загрузка происходит один раз при первом вызове (ленивая инициализация).
"""

from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Глобальный синглтон — загружается один раз
_reranker = None
_reranker_model_name: Optional[str] = None

# Модели в порядке предпочтения (первая доступная будет использована)
CANDIDATE_MODELS = [
    "amberoad/bert-multilingual-passage-reranking-msmarco",  # лучший для RU
    "cross-encoder/ms-marco-MiniLM-L-6-v2",                 # быстрый, EN-ориентированный
    "cross-encoder/ms-marco-TinyBERT-L-2-v2",               # самый лёгкий fallback
]


def _get_reranker():
    """Ленивая загрузка reranker-модели."""
    global _reranker, _reranker_model_name
    if _reranker is not None:
        return _reranker

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning("sentence-transformers не установлен. pip install sentence-transformers")
        return None

    import os as _os
    for model_name in CANDIDATE_MODELS:
        try:
            logger.info(f"[RERANKER] Загрузка модели: {model_name}")
            # TRANSFORMERS_OFFLINE=1 — не проверять обновления на HuggingFace
            # если модель уже есть в кеше. Убирает таймауты при офлайн-работе.
            _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            _os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            _reranker = CrossEncoder(model_name, max_length=512)
            _reranker_model_name = model_name
            logger.info(f"[RERANKER] Модель загружена: {model_name}")
            return _reranker
        except Exception as e:
            logger.warning(f"[RERANKER] Не удалось загрузить {model_name}: {e}")
            # Если не нашли в кеше — пробуем онлайн
            try:
                _os.environ.pop("TRANSFORMERS_OFFLINE", None)
                _reranker = CrossEncoder(model_name, max_length=512)
                _reranker_model_name = model_name
                logger.info(f"[RERANKER] Модель загружена онлайн: {model_name}")
                return _reranker
            except Exception as e2:
                logger.warning(f"[RERANKER] Онлайн загрузка тоже не удалась {model_name}: {e2}")
                continue

    logger.error("[RERANKER] Ни одна модель не доступна")
    return None


def _split_into_spans(text: str, min_len: int = 30) -> list[tuple[int, int, str]]:
    """
    Разбивает текст на небольшие spans для reranking.

    Стратегия: сначала разбиваем по абзацам (\\n\\n или \\n),
    затем слишком длинные абзацы режем по ~200 символов с перекрытием.
    Возвращает список (start, end, text).
    """
    spans = []

    # Разбиваем по абзацам
    paragraphs = re.split(r'\n{2,}', text)
    pos = 0
    for para in paragraphs:
        para_start = text.find(para, pos)
        if para_start == -1:
            para_start = pos
        para_end = para_start + len(para)

        stripped = para.strip()
        if len(stripped) < min_len:
            pos = para_end
            continue

        # Длинный абзац — режем на подспаны по ~200 символов с перекрытием 40
        if len(stripped) > 250:
            sub_pos = 0
            while sub_pos < len(para):
                sub = para[sub_pos:sub_pos + 220]
                sub_stripped = sub.strip()
                if len(sub_stripped) >= min_len:
                    abs_start = para_start + sub_pos
                    abs_end = abs_start + len(sub)
                    spans.append((abs_start, min(abs_end, len(text)), sub_stripped))
                sub_pos += 180  # шаг с перекрытием 40
        else:
            spans.append((para_start, para_end, stripped))

        pos = para_end

    # Если абзацев нет — разбиваем по строкам
    if not spans:
        lines = text.split('\n')
        pos = 0
        for line in lines:
            line_start = text.find(line, pos)
            if line_start == -1:
                line_start = pos
            stripped = line.strip()
            if len(stripped) >= min_len:
                spans.append((line_start, line_start + len(line), stripped))
            pos = line_start + len(line) + 1

    return spans


def rerank_chunks(
    query: str,
    docs: list,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> list[dict]:
    """
    Основная функция: принимает вопрос и список LangChain-документов,
    возвращает список dict с точными диапазонами для подсветки.

    Каждый документ разбивается на spans, каждый span скорируется
    cross-encoder'ом. Возвращаются top_k лучших spans.

    Args:
        query: исходный вопрос пользователя
        docs: список LangChain Document (с metadata.source, metadata.start_char)
        top_k: сколько лучших фрагментов вернуть
        score_threshold: минимальный score (логит) для включения в результат

    Returns:
        Список dict: {source, start_char, end_char, text, relevance_score}
    """
    reranker = _get_reranker()
    if reranker is None:
        return []

    # Собираем все spans со всех документов
    all_spans = []  # (source, chunk_start, span_rel_start, span_rel_end, span_text)
    for doc in docs:
        src = doc.metadata.get("source", "")
        if not src:
            continue
        # FAISS хранит числовые метаданные как numpy-скаляры — принудительно int
        raw_start = doc.metadata.get("start_char", 0)
        try:
            chunk_start = int(raw_start) if raw_start is not None else 0
        except (TypeError, ValueError):
            chunk_start = 0
        chunk_text = doc.page_content

        spans = _split_into_spans(chunk_text)
        for rel_start, rel_end, span_text in spans:
            all_spans.append((src, chunk_start, int(rel_start), int(rel_end), span_text))

    if not all_spans:
        return []

    # Батч-скоринг через cross-encoder
    pairs = [(query, span[4]) for span in all_spans]
    try:
        import numpy as _np
        raw_scores = reranker.predict(pairs)
        # predict() возвращает скаляр при одном элементе — нормализуем в 1D массив
        scores_arr = _np.atleast_1d(raw_scores)
        # Конвертируем в обычные Python float — никаких numpy-скаляров дальше
        scores_list = [float(s) for s in scores_arr.flatten()]
    except Exception as e:
        logger.error(f"[RERANKER] Ошибка predict: {e}")
        return []

    # Сортируем по score, берём top_k
    scored = sorted(
        zip(scores_list, all_spans),
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    seen_ranges: dict[str, list[tuple[int, int]]] = {}  # дедупликация перекрытий

    for score, (src, chunk_start, rel_start, rel_end, span_text) in scored:
        if float(score) < score_threshold:
            break
        if len(results) >= top_k:
            break

        abs_start = int(chunk_start) + int(rel_start)
        abs_end = int(chunk_start) + int(rel_end)

        # Пропускаем если сильно перекрывается с уже добавленным
        overlaps = seen_ranges.get(src, [])
        overlap = False
        for existing_s, existing_e in overlaps:
            intersection = min(abs_end, existing_e) - max(abs_start, existing_s)
            union = max(abs_end, existing_e) - min(abs_start, existing_s)
            if union > 0 and intersection / union > 0.5:
                overlap = True
                break
        if overlap:
            continue

        seen_ranges.setdefault(src, []).append((abs_start, abs_end))
        results.append({
            "source": src,
            "start_char": abs_start,
            "end_char": abs_end,
            "text": span_text,
            "relevance_score": score,  # already Python float
        })

    return results


def is_available() -> bool:
    """Проверяет доступность reranker без загрузки модели."""
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        return True
    except ImportError:
        return False
