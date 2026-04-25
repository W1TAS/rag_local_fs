import os
import re
import json
import pickle
import hashlib
from collections import Counter, defaultdict
import numpy as np
import requests
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import SUPPORTED_FORMATS, get_llm_settings
from typing import Optional


def _extract_file_from_query(query: str, folder_path: str) -> Optional[str]:
    """
    Ищет в запросе имя файла и возвращает полный путь.
    Стратегии (в порядке приоритета):
      1. Точное имя в кавычках (с расширением или без)
      2. После служебных слов "файл/документ" — с расширением
      3. Fuzzy: слово из запроса совпадает с именем файла без расширения
    """
    if not folder_path:
        return None

    query_low = query.lower()

    # Собираем индекс файлов один раз
    file_index = []  # (lower_name_no_ext, lower_name_full, full_path)
    for root, _, files in os.walk(folder_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower().lstrip(".")
            if ext in SUPPORTED_FORMATS:
                full = os.path.join(root, f)
                no_ext = os.path.splitext(f.lower())[0]
                file_index.append((no_ext, f.lower(), full))

    if not file_index:
        return None

    # 1. Точное имя в кавычках (с расширением или без)
    m = re.search(r'[«"\'"]([^«"\'"\ ][^«"\'"]*)[\«"\'"]]', query_low)
    if m:
        candidate = m.group(1).strip()
        for no_ext, full_name, path in file_index:
            if full_name == candidate or no_ext == candidate:
                return path

    # 2. После служебных слов "файл/документ" — с расширением
    ext_pattern = '|'.join(SUPPORTED_FORMATS)
    m = re.search(
        r'(?:в|по|из|про|о|об|читай|смотри|анализируй)\s+'
        r'(?:файл[еаи]?|документ[еаи]?)?\s*'
        r'["\']?([^\s"\',;]+\.(?:' + ext_pattern + r'))',
        query_low
    )
    if m:
        candidate = m.group(1).strip()
        for _, full_name, path in file_index:
            if full_name == candidate:
                return path

    # 3. Fuzzy: слово из запроса совпадает с именем файла без расширения
    words = re.findall(r'\w{4,}', query_low)
    for word in words:
        for no_ext, full_name, path in file_index:
            if word == no_ext or (len(word) >= 5 and (word in no_ext or no_ext in word)):
                return path

    return None


# === СТОП-СЛОВА ===
RUSSIAN_STOP_WORDS = {
    'и', 'в', 'не', 'на', 'я', 'с', 'что', 'он', 'по', 'это', 'как', 'а', 'но', 'к', 'у',
    'да', 'ты', 'до', 'из', 'мы', 'за', 'бы', 'о', 'со', 'для', 'от', 'то', 'же', 'вы',
    'же', 'ли', 'ни', 'был', 'была', 'было', 'были', 'есть', 'быть', 'будет', 'все',
    'ещё', 'уже', 'только', 'даже', 'вот', 'там', 'тут', 'куда', 'откуда', 'когда',
    'если', 'то', 'или', 'ни', 'нибудь', 'какой', 'какая', 'какое', 'какие', 'такой',
    'такое', 'такой', 'такое', 'такои', 'такои', 'этот', 'эта', 'это', 'эти', 'тот',
    'та', 'то', 'те', 'очень', 'можно', 'нужно', 'надо', 'хочу', 'может', 'должен',
    'сказать', 'ответь', 'напиши', 'объясни', 'расскажи', 'что', 'какой', 'кто',
    'где', 'когда', 'почему', 'как', 'отвечай', 'русском'
}


# === КЭШИРОВАНИЕ СУММ ===
def get_folder_hash(folder_path):
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            path = os.path.join(root, file)
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            if ext in SUPPORTED_FORMATS:
                try:
                    stat = os.stat(path)
                    hash_md5.update(f"{file}:{stat.st_mtime}".encode())
                except:
                    pass
    return hash_md5.hexdigest()


def _ollama_available():
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def clear_summary_cache(folder_path: str, file_filter: str = None) -> bool:
    """Удалить кэш суммаризации. Если file_filter — только кэш конкретного файла."""
    try:
        from cache import get_folder_cache_dir
        folder_cache = get_folder_cache_dir(folder_path)
        if file_filter:
            slug = hashlib.md5(file_filter.encode()).hexdigest()[:10]
            targets = [
                os.path.join(folder_cache, f"summary_cache_{slug}.pkl"),
                os.path.join(folder_cache, f"summary_hash_{slug}.txt"),
            ]
        else:
            targets = [
                os.path.join(folder_cache, "summary_cache.pkl"),
                os.path.join(folder_cache, "summary_hash.txt"),
            ]
        removed = False
        for p in targets:
            if os.path.exists(p):
                os.remove(p)
                removed = True
        return removed
    except Exception:
        return False


def summarize_all_in_one(vectorstore, model_name, use_gpu=True, folder_path=None, file_filter=None):
    # Отдельный кэш для каждого file_filter (и для общего вызова)
    if file_filter:
        slug = hashlib.md5(file_filter.encode()).hexdigest()[:10]
        cache_basename = f"summary_cache_{slug}.pkl"
        hash_basename = f"summary_hash_{slug}.txt"
    else:
        cache_basename = "summary_cache.pkl"
        hash_basename = "summary_hash.txt"

    cache_file = cache_basename
    hash_file = hash_basename
    current_hash = get_folder_hash(folder_path) if folder_path else ""
    if folder_path:
        try:
            from cache import get_folder_cache_dir
            folder_cache = get_folder_cache_dir(folder_path)
            cache_file = os.path.join(folder_cache, cache_basename)
            hash_file = os.path.join(folder_cache, hash_basename)
        except Exception:
            pass

    if folder_path and os.path.exists(cache_file) and os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            cached_hash = f.read().strip()
        if cached_hash == current_hash:
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    if not _ollama_available():
        return "Локальный сервер моделей (Ollama) недоступен. Запустите 'ollama serve' или переключитесь на облачную модель в настройках."
    try:
        try:
            llm = ChatOllama(model=model_name,
                             num_gpu=-1 if use_gpu else 0,
                             temperature=0.1,
                             base_url="http://127.0.0.1:11434",
                             keep_alive="5m",
                             timeout=60)
        except TypeError:
            llm = ChatOllama(model=model_name,
                             num_gpu=-1 if use_gpu else 0,
                             temperature=0.1,
                             base_url="http://127.0.0.1:11434")
    except Exception as e:
        return f"Ошибка инициализации модели: {e}"
    seen = set()
    previews = []
    for doc in vectorstore.docstore._dict.values():
        source = doc.metadata.get("source")
        # If file_filter provided, only include that file's docs
        if file_filter:
            if source != file_filter:
                continue
        if source and source not in seen:
            seen.add(source)
            text = doc.page_content[:600]
            name = os.path.basename(source)
            previews.append(f"[Файл: {name}]\n{text}\n")
    context = "\n".join(previews)[:6000]

    prompt = ChatPromptTemplate.from_template(
        """Ты — эксперт по кратким описаниям. 
        Прочитай фрагменты из файлов ниже. Для КАЖДОГО файла дай описание в не менее чем 2 предложения.

    ПРИМЕР ТВОЕГО ОТВЕТА (В ТВОЕМ ОТВЕТЕ НЕ ОБЯЗАТЕЛЬНО БУДЕТ ДВА ФАЙЛА, МОЖЕТ БЫТЬ 1 ИЛИ БОЛЬШЕ. ОТВЕЧАЙ ЧЕТКО ПО ФАЙЛАМ В БАЗЕ ДАННЫХ)
        Формат ответа :
        - [имя_файла_1]:[краткое описание].
        - [имя_файла_2]:[краткое описание].
        и так далее...

        Файлы:
        {context}

        Ответ:"""
    )

    try:
        response = llm.invoke(prompt.format(context=context))
        result = response.content.strip()
        if folder_path:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                with open(hash_file, "w", encoding="utf-8") as f:
                    f.write(current_hash)
            except Exception:
                pass
        return result
    except Exception as e:
        return f"Ошибка суммирования: {e}"


def generate_suggested_questions(vectorstore, model_name, use_gpu=True, folder_path=None, max_q=5):
    """Генерирует 3-6 рекомендуемых вопросов по набору файлов в папке.
    Возвращает список строк (вопросов) на русском.
    """
    if not _ollama_available():
        return []

    # Собираем превью каждого файла (первые 300 символов)
    seen = set()
    previews = []
    for doc in vectorstore.docstore._dict.values():
        source = doc.metadata.get("source")
        if not source or source in seen:
            continue
        seen.add(source)
        name = os.path.basename(source)
        text = doc.page_content[:300].replace('\n', ' ')
        previews.append(f"[{name}]: {text}")

    context = "\n".join(previews)[:4000]

    try:
        try:
            llm = ChatOllama(model=model_name,
                             num_gpu=-1 if use_gpu else 0,
                             temperature=0.2,
                             base_url="http://127.0.0.1:11434",
                             keep_alive="2m",
                             timeout=30)
        except TypeError:
            llm = ChatOllama(model=model_name,
                             num_gpu=-1 if use_gpu else 0,
                             temperature=0.2,
                             base_url="http://127.0.0.1:11434")
    except Exception:
        return []

    prompt = ChatPromptTemplate.from_template(
        """Ты — помощник, который генерирует короткие вопросы, которые пользователь мог бы задать по приведённым файлам.

Файлы и их фрагменты:
{context}

Сгенерируй от 3 до {max_q} кратких пользовательских вопросов (на русском), каждый на новой строке. Вопросы должны быть разные и покрывать основные темы файлов.

Ответ:"""
    )

    try:
        response = llm.invoke(prompt.format(context=context, max_q=max_q))
        text = response.content.strip()
        # Разбиваем по строкам и фильтруем
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # Обрезаем до max_q
        return lines[:max_q]
    except Exception:
        return []


# === КЛЮЧЕВЫЕ СЛОВА ===
def extract_keywords_from_query(query, top_n=10, min_length=4):
    query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
    words = query_clean.split()
    filtered = [w for w in words if len(w) >= min_length and w not in RUSSIAN_STOP_WORDS]
    freq = Counter(filtered)
    total = len(filtered)
    weights = {}
    for word, f in freq.items():
        tf = f / total if total > 0 else 0
        idf = np.log(len(word) + 1) * np.log(f + 1)
        weights[word] = tf * idf
    return [k for k, _ in sorted(weights.items(), key=lambda x: -x[1])[:top_n]]


# === ОЦЕНКА РЕЛЕВАНТНОСТИ ФАЙЛА ===
def score_file_relevance(docs, query):
    keywords = extract_keywords_from_query(query, top_n=10)
    if not keywords:
        return {}
    file_scores = defaultdict(list)
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        matches = sum(1 for kw in keywords if kw in doc.page_content.lower())
        file_scores[source].append(matches)
    avg_scores = {
        source: sum(matches) / len(matches) if matches else 0
        for source, matches in file_scores.items()
    }
    return avg_scores


def select_best_file(docs, query):
    avg_scores = score_file_relevance(docs, query)
    return max(avg_scores, key=avg_scores.get) if avg_scores else None


# === ФОРМАТИРОВАНИЕ КОНТЕКСТА ===
def format_context_with_sources(docs, query):
    chunks = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "Неизвестный"))
        chunks.append(f"[Источник: {source}]\n{doc.page_content}\n")
    return "\n\n".join(chunks) if chunks else "Нет релевантного контекста"


# === НОВАЯ ФУНКЦИЯ: ВЫВОД ВСЕГО ТЕКСТА ИЗ ФАЙЛА ===
def print_full_file_text(vectorstore, file_path):
    """
    Выводит в консоль ВЕСЬ текст из указанного файла (все чанки).
    """
    print("\n" + "=" * 80)
    print(f"ПОЛНЫЙ ТЕКСТ ФАЙЛА: {os.path.basename(file_path)}")
    print("=" * 80)
    full_text = []
    for doc in vectorstore.docstore._dict.values():
        if doc.metadata.get("source") == file_path:
            full_text.append(doc.page_content)
    print("\n".join(full_text))
    print("\n" + "=" * 80)
    print(f"Всего чанков: {len(full_text)}")
    print("=" * 80 + "\n")


def _write_debug_chunks(query: str, docs: list) -> None:
    """Записывает последние обработанные чанки в ПОСЛЕДНИЕ_ЧАНКИ.txt рядом с rag.py."""
    from datetime import datetime
    save_dir = os.path.dirname(os.path.abspath(__file__))
    debug_file = os.path.join(save_dir, "ПОСЛЕДНИЕ_ЧАНКИ.txt")
    try:
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(f"Время: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Вопрос: {query}\n")
            f.write("=" * 100 + "\n\n")
            for i, doc in enumerate(docs, 1):
                filename = os.path.basename(doc.metadata.get("source", "неизвестно"))
                f.write(f"ЧАНК {i}  ←  {filename}\n")
                f.write("\u2014" * 80 + "\n")
                f.write(doc.page_content.strip())
                f.write("\n\n" + "\u2550" * 80 + "\n\n")
            f.write(f"Всего чанков: {len(docs)}\n")
        print(f"Чанки сохранены: {debug_file}")
    except Exception as e:
        print(f"Не удалось записать чанки: {e}")


def _docs_to_highlights(docs: list, top_k: int = 3, base_score: float = 0.5) -> list[dict]:
    """
    Конвертирует список документов в highlight-диапазоны.
    Используется как fallback когда reranker недоступен или не смог проскорить.
    Если start_char/end_char отсутствуют — ищет текст чанка в файле напрямую.
    """
    results = []
    for doc in docs[:top_k]:
        src = doc.metadata.get("source", "")
        if not src:
            continue
        try:
            start = int(doc.metadata.get("start_char") or 0)
            end = int(doc.metadata.get("end_char") or 0)
        except (TypeError, ValueError):
            start, end = 0, 0

        # Если позиции нулевые или некорректные — ищем текст в файле
        if end <= start:
            try:
                ext = os.path.splitext(src)[1].lower().lstrip(".")
                if ext in ("txt", "md"):
                    with open(src, "r", encoding="utf-8", errors="replace") as f:
                        file_text = f.read()
                elif ext == "pdf":
                    import PyPDF2
                    with open(src, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        file_text = "".join(p.extract_text() or "" for p in reader.pages)
                elif ext == "docx":
                    from docx import Document as _D
                    file_text = "\n".join(p.text for p in _D(src).paragraphs)
                else:
                    file_text = doc.page_content
                anchor = doc.page_content.strip()[:120]
                pos = file_text.find(anchor)
                if pos != -1:
                    start = pos
                    end = pos + len(doc.page_content)
                else:
                    # Последний resort: позиция 0, длина текста чанка
                    start = 0
                    end = len(doc.page_content)
                    # Но тогда src должен быть заменён на in-memory текст
            except Exception:
                start, end = 0, len(doc.page_content)

        results.append({
            "source": src,
            "start_char": start,
            "end_char": end,
            "text": doc.page_content,
            "relevance_score": base_score,
        })
    return results


def _rerank_highlight(query: str, docs: list, top_k: int = 3) -> list[dict]:
    """
    Скорирует каждый чанк reranker-моделью по паре (вопрос, чанк).
    Возвращает top_k чанков с наибольшим score в виде highlight-диапазонов.
    При любой ошибке reranker — возвращает все входные чанки как есть (не пустой список).
    """
    if not docs:
        return []

    try:
        from reranker import _get_reranker
        reranker = _get_reranker()
    except Exception:
        reranker = None

    if reranker is None:
        return _docs_to_highlights(docs, top_k)

    # Батч-скоринг
    pairs = [(query, doc.page_content) for doc in docs]
    try:
        import numpy as _np
        raw_scores = reranker.predict(pairs)
        scores = [float(s) for s in _np.atleast_1d(raw_scores).flatten()]
    except Exception:
        # Reranker упал — возвращаем все чанки без ранжирования
        return _docs_to_highlights(docs, top_k)

    # Берём top_k по score
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[:top_k]]
    return _docs_to_highlights(top_docs, top_k, base_score=0.0)



# Максимальная длина промпта в символах — обрезаем чтобы не получить 400
_OPENROUTER_MAX_PROMPT_CHARS = 12_000


class OpenRouterLLM:
    """
    Враппер для OpenRouter API с автоматическим retry и fallback на другие модели.

    При 429 (rate limit) или 400 (контекст слишком длинный) автоматически
    пробует следующую модель из OPENROUTER_FREE_MODELS.
    Использует requests напрямую — без langchain-openai.
    """

    def __init__(self, api_key: str, model: str, temperature: float = 0.1):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    class _Response:
        def __init__(self, text: str):
            self.content = text

    def _prompt_to_text(self, prompt) -> str:
        if hasattr(prompt, "to_string"):
            text = prompt.to_string()
        elif hasattr(prompt, "text"):
            text = prompt.text
        else:
            text = str(prompt)
        # Обрезаем слишком длинный промпт
        if len(text) > _OPENROUTER_MAX_PROMPT_CHARS:
            text = text[:_OPENROUTER_MAX_PROMPT_CHARS] + "\n...[контекст обрезан]"
        return text

    def _call_model(self, model: str, text: str):
        import requests as _req
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/W1TAS/rag_local_fs",
            "X-Title": "RAG Local FS",
        }
        resp = _req.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        return resp

    def invoke(self, prompt) -> "_Response":
        import logging as _log

        text = self._prompt_to_text(prompt)
        _log.info(f"[OPENROUTER] Запрос к модели: {self.model}")

        resp = self._call_model(self.model, text)

        if resp.status_code == 200:
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            _log.info(f"[OPENROUTER] Ответ получен от: {self.model}")
            result = self._Response(answer)
            result.model_used = self.model
            return result

        # Человекочитаемые сообщения об ошибках
        code = resp.status_code
        try:
            err = resp.json().get("error", {})
            msg = err.get("message", resp.text[:200])
        except Exception:
            msg = resp.text[:200]

        if code == 429:
            raise RuntimeError(f"Модель {self.model} перегружена (rate limit). Попробуйте позже или выберите другую модель в настройках.")
        elif code == 400:
            raise RuntimeError(f"Модель {self.model} не приняла запрос (возможно слишком длинный контекст). Выберите другую модель.")
        elif code == 401:
            raise RuntimeError("Неверный API ключ OpenRouter. Проверьте ключ в настройках.")
        elif code == 404:
            raise RuntimeError(f"Модель {self.model} не найдена на OpenRouter. Выберите другую модель в настройках.")
        else:
            raise RuntimeError(f"Ошибка OpenRouter {code}: {msg}")



def get_rag_chain(vectorstore, model_name, use_gpu=True, embedding_model="embeddinggemma:latest", folder_path=None,
                  llm_provider="ollama", openrouter_api_key="", openrouter_model=""):
    global llm
    llm = None

    if llm_provider == "openrouter" and openrouter_api_key:
        import logging as _log
        _log.info(f"[RAG] Инициализация OpenRouter, модель: {openrouter_model}")
        llm = OpenRouterLLM(
            api_key=openrouter_api_key,
            model=openrouter_model or "openrouter/elephant-alpha",
        )
    else:
        import logging as _log
        _log.info(f"[RAG] Инициализация Ollama, модель: {model_name}")
        ollama_ok = _ollama_available()
        if ollama_ok:
            try:
                try:
                    llm = ChatOllama(model=model_name,
                                     num_gpu=-1 if use_gpu else 0,
                                     temperature=0.1,
                                     base_url="http://127.0.0.1:11434",
                                     keep_alive="5m",
                                     timeout=60)
                except TypeError:
                    llm = ChatOllama(model=model_name,
                                     num_gpu=-1 if use_gpu else 0,
                                     temperature=0.1,
                                     base_url="http://127.0.0.1:11434")
            except Exception as e:
                llm = None

    prompt = ChatPromptTemplate.from_template(
        """Отвечай на русском языке строго на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {input}

Ответ:"""
    )


        # Паттерны запросов о содержании конкретного файла
    _FILE_SUMMARY_PATTERNS = [
        "о чём", "о чем", "что в", "содержание", "содержит",
        "про что", "расскажи про", "опиши", "что такое", "что это"
    ]

    def wrapped_qa_chain(query, file_filter=None):
        query_lower = query.lower().strip()

        # --- Определяем конкретный файл из запроса ---
        inferred_file = None
        if not file_filter and folder_path:
            inferred_file = _extract_file_from_query(query, folder_path)

        effective_file = file_filter or inferred_file

        # === ВЕТКА 1: общий вопрос "о чём все файлы" ===
        general_patterns = [
            "о чём файлы", "что в файлах", "опиши файлы",
            "о чем файлы", "о чём эти файлы", "о чём все файлы",
            "что в этих файлах", "опиши содержимое", "о чём документы"
        ]
        if any(p in query_lower for p in general_patterns) and not effective_file:
            summary = summarize_all_in_one(vectorstore, model_name, use_gpu, folder_path=folder_path)
            return {
                "result": summary,
                "source_documents": [],
                "sources": "все файлы",
                "keywords": [],
                "formatted_context": ""
            }

        # === ВЕТКА 2: "о чём файл X" — суммаризация конкретного файла ===
        if effective_file and any(p in query_lower for p in _FILE_SUMMARY_PATTERNS):
            # Собираем ВСЕ чанки конкретного файла из vectorstore для дебаг-файла
            eff_base = os.path.basename(effective_file).lower()
            file_chunks = [
                doc for doc in vectorstore.docstore._dict.values()
                if os.path.basename(doc.metadata.get("source", "")).lower() == eff_base
            ]
            _write_debug_chunks(query, file_chunks)

            summary = summarize_all_in_one(
                vectorstore, model_name, use_gpu,
                folder_path=folder_path, file_filter=effective_file
            )
            fname = os.path.basename(effective_file)
            return {
                "result": summary,
                "source_documents": file_chunks[:5],
                "sources": fname,
                "keywords": [],
                "formatted_context": summary[:500]
            }

        # === ВЕТКА 3: обычный вопрос — MMR retriever ===
        # MMR (Max Marginal Relevance) выбирает релевантные И разнообразные чанки,
        # не 5 похожих кусков из одного места, а из разных частей документов.
        k = 8  # берём больше чанков
        fetch_k = min(30, k * 4)  # кандидатов для MMR-фильтрации

        try:
            if effective_file:
                # Фильтр по полному пути (как хранится в метаданных FAISS)
                raw_docs = vectorstore.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k,
                    filter={"source": effective_file}
                )
                # Если по полному пути ничего нет — попробуем без фильтра и отфильтруем вручную
                if not raw_docs:
                    all_docs = vectorstore.max_marginal_relevance_search(query, k=20, fetch_k=60)
                    eff_base = os.path.basename(effective_file).lower()
                    raw_docs = [
                                   d for d in all_docs
                                   if os.path.basename(d.metadata.get("source", "")).lower() == eff_base
                               ][:k]
            else:
                raw_docs = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        except Exception:
            # Fallback: обычный similarity search
            search_kwargs = {"k": k}
            if effective_file:
                search_kwargs["filter"] = {"source": effective_file}
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
            raw_docs = retriever.invoke(query)

        if not raw_docs:
            return {
                "result": "Информация отсутствует в доступных документах",
                "source_documents": [],
                "sources": "Нет источников",
                "keywords": extract_keywords_from_query(query, top_n=7),
                "formatted_context": ""
            }

        # === Выбор файла-источника ===
        if effective_file:
            # Уже знаем файл — используем все найденные чанки
            final_docs = raw_docs
            best_file = os.path.basename(effective_file)
        else:
            # Keyword reranking для выбора наиболее релевантного файла
            best_file = select_best_file(raw_docs, query)
            final_docs = [
                doc for doc in raw_docs
                if os.path.basename(doc.metadata.get("source", "")) == best_file
            ]
            # Если после фильтрации ничего нет — берём всё
            if not final_docs:
                final_docs = raw_docs

        context = format_context_with_sources(final_docs, query)
        if llm is None:
            return {
                "result": (
                    "Локальный сервер моделей (Ollama) недоступен или модель не инициализирована. "
                    "Запустите Ollama (ollama serve) и выполните 'ollama pull' для модели: "
                    f"{model_name}."
                ),
                "source_documents": [],
                "sources": "Нет источников",
                "keywords": extract_keywords_from_query(query, top_n=7),
                "formatted_context": ""
            }
        try:
            prompt_text = prompt.format(context=context, input=query)
            # detect streaming capability on the LLM object
            stream_fn = next(
                (n for n in ("stream", "stream_invoke", "invoke_stream", "stream_chat", "stream_invoke_chat") if hasattr(llm, n)),
                None,
            )

            if stream_fn:
                gen = getattr(llm, stream_fn)(prompt_text)

                def _stream_generator():
                    """Streaming generator that yields only the delta (new text) on each chunk.
                    This reduces duplication in the UI and allows append-only updates.
                    Yields dicts like {"delta": "..." , "final": False} and a final payload with "final": True.
                    """
                    cum = ""
                    last_text_chunk = ""
                    try:
                        for chunk in gen:
                            # chunk may be object with textual attributes or plain string
                            text_chunk = None
                            # 1) plain string
                            if isinstance(chunk, str):
                                text_chunk = chunk
                            # 2) objects with attributes
                            elif hasattr(chunk, "content") and isinstance(getattr(chunk, "content"), str):
                                text_chunk = getattr(chunk, "content")
                            elif hasattr(chunk, "text") and isinstance(getattr(chunk, "text"), str):
                                text_chunk = getattr(chunk, "text")
                            elif isinstance(chunk, dict):
                                # avoid dumping large metadata dicts (they have many keys but no textual keys)
                                text_keys = [k for k in ("delta", "content", "text", "partial", "result") if k in chunk]
                                if text_keys:
                                    for k in text_keys:
                                        v = chunk.get(k)
                                        if isinstance(v, str) and v.strip():
                                            text_chunk = v
                                            break
                                else:
                                    # try nested choices
                                    choices = chunk.get("choices")
                                    if isinstance(choices, list) and choices:
                                        first = choices[0]
                                        if isinstance(first, dict):
                                            for k in ("text", "message", "content"):
                                                if k in first and isinstance(first[k], str) and first[k].strip():
                                                    text_chunk = first[k]
                                                    break
                            else:
                                # fallback: try to convert to str but only if short
                                try:
                                    s = str(chunk)
                                    if 0 < len(s) < 200 and "response_metadata" not in s and "additional_kwargs" not in s:
                                        text_chunk = s
                                except Exception:
                                    text_chunk = None

                            if not text_chunk:
                                # skip non-textual metadata
                                continue

                            # sanitize
                            text_chunk = text_chunk.replace('\r', '')
                            # compute delta — ideally text_chunk itself represents the new portion
                            delta = text_chunk
                            if not delta.strip():
                                continue

                            # avoid re-emitting identical consecutive chunks
                            if delta == last_text_chunk:
                                continue
                            last_text_chunk = delta

                            cum += delta
                            yield {"delta": delta, "final": False}

                    except Exception:
                        # streaming failed — fall back to non-stream below
                        pass

                    answer = cum.strip()

                    # Post-process highlight_chunks same as non-streaming branch
                    sources = {best_file} if best_file else set()
                    _write_debug_chunks(query, raw_docs)

                    highlight_chunks = []
                    _no_answer_markers = [
                        "информация отсутствует", "нет информации", "нет данных",
                        "не содержит", "не найдено", "не упоминается", "отсутствует в",
                        "не могу ответить", "не нашёл", "не нашел", "нет ответа",
                        "не указано", "контекст не содержит",
                    ]
                    answer_lower = answer.lower()
                    answer_has_info = not any(m in answer_lower for m in _no_answer_markers)

                    if answer_has_info and final_docs:
                        answer_words = set(re.findall(r'\w{4,}', answer_lower))
                        answer_words -= RUSSIAN_STOP_WORDS

                        for doc in final_docs:
                            src = doc.metadata.get("source", "")
                            if not src:
                                continue
                            chunk_words = set(re.findall(r'\w{4,}', doc.page_content.lower()))
                            chunk_words -= RUSSIAN_STOP_WORDS
                            if not chunk_words:
                                continue
                            overlap = chunk_words & answer_words
                            score = len(overlap) / len(chunk_words)
                            if score >= 0.06:
                                highlight_chunks.append({
                                    "source": src,
                                    "start_char": doc.metadata.get("start_char"),
                                    "end_char": doc.metadata.get("end_char"),
                                    "start_line": doc.metadata.get("start_line"),
                                    "chunk_index": doc.metadata.get("chunk_index"),
                                    "text": doc.page_content,
                                    "relevance_score": score,
                                })

                        highlight_chunks.sort(key=lambda x: -x["relevance_score"])

                    yield {
                        "result": answer,
                        "source_documents": final_docs,
                        "sources": ", ".join(sources),
                        "keywords": extract_keywords_from_query(query, top_n=7),
                        "formatted_context": context[:500] + "..." if len(context) > 500 else context,
                        "highlight_chunks": highlight_chunks,
                        "final": True,
                    }

                return _stream_generator()

            # Fallback: synchronous invoke
            answer_obj = llm.invoke(prompt.format(context=context, input=query))
            answer = answer_obj.content.strip()
        except Exception as e:
            return {
                "result": f"Ошибка при запросе к модели: {e}",
                "source_documents": [],
                "sources": "Нет источников",
                "keywords": extract_keywords_from_query(query, top_n=7),
                "formatted_context": context[:500] + "..." if len(context) > 500 else context,
            }

        sources = {best_file} if best_file else set()

        _write_debug_chunks(query, raw_docs)

        # Подсветка: reranker скорирует чанки по вопросу, возвращаем топ релевантных.
        # Подсвечиваем чанки целиком — честно и предсказуемо, без попыток угадать фразу.
        highlight_chunks = _rerank_highlight(query, final_docs)

        # Сортируем по позиции в файле для последовательной подсветки
        highlight_chunks.sort(key=lambda x: (x.get("source", ""), x.get("start_char", 0)))

        # Определяем какая модель реально ответила
        model_used = openrouter_model if llm_provider == "openrouter" else model_name
        if llm_provider == "openrouter" and hasattr(answer_obj, "model_used"):
            model_used = answer_obj.model_used

        return {
            "result": answer,
            "source_documents": final_docs,
            "sources": ", ".join(sources),
            "keywords": extract_keywords_from_query(query, top_n=7),
            "formatted_context": context[:500] + "..." if len(context) > 500 else context,
            "highlight_chunks": highlight_chunks,
            "model_used": model_used,
            "llm_provider": llm_provider,
        }

    return wrapped_qa_chain