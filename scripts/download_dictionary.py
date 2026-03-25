"""
Скачивает частотный словарь русского языка.

Источники:
  1. Wikipedia frequency list (wikitionary/Wikipedia dumps)
  2. Fallback: генерирует из nltk corpus

Результат: data/ru_freq_dict.txt
Формат: слово<TAB>частота (одно слово на строку)

Использование:
    python scripts/download_dictionary.py
"""

import os
import re
import sys
import gzip
import io
from urllib import request, error
from typing import List, Tuple


# ── Источники ───────────────────────────────────────────────────────────

# Частотный словарь из проекта Hermit Dave
# (на основе OpenSubtitles, ~50k слов, очень качественный)
# https://github.com/hermitdave/FrequencyWords
FREQ_WORDS_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
    "master/content/2018/ru/ru_50k.txt"
)

# Запасной: полный словарь (~400k)
FREQ_WORDS_FULL_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
    "master/content/2018/ru/ru_full.txt"
)


def _get_output_path() -> str:
    """Путь к выходному файлу."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return os.path.join(repo_root, "data", "ru_freq_dict.txt")


def _download(url: str, timeout: int = 30) -> str:
    """Скачать текст по URL."""
    print(f"  Downloading: {url}")
    req = request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (RAG-Autocomplete)")

    with request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        # Пробуем декодировать
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")


def _parse_freq_words(text: str) -> List[Tuple[str, int]]:
    """
    Парсит формат 'слово частота' или 'слово\tчастота'.
    Фильтрует мусор, оставляет только нормальные слова.
    """
    result = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Формат: "слово частота" или "слово\tчастота"
        parts = re.split(r"[\t ]+", line, maxsplit=1)
        if len(parts) != 2:
            continue

        word, freq_str = parts[0].strip(), parts[1].strip()

        # Проверяем частоту
        try:
            freq = int(freq_str)
        except ValueError:
            continue

        # Фильтруем
        if len(word) < 2:
            continue
        if not re.match(r"^[А-Яа-яЁё]+$", word):
            # Пропускаем слова с цифрами, латиницей, спецсимволами
            continue
        if freq < 2:
            continue

        result.append((word.lower(), freq))

    return result


def _add_rag_words(words: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """
    Добавить слова, специфичные для RAG-запросов.
    Эти слова могут отсутствовать в частотном словаре,
    но часто используются в поисковых запросах.
    """
    rag_specific = {
        # Глаголы-команды
        "найди": 5000, "покажи": 5000, "суммируй": 4000,
        "объясни": 4000, "перечисли": 3500, "сравни": 3500,
        "выдели": 3000, "проанализируй": 3000, "извлеки": 2500,
        "опиши": 2500, "расскажи": 2500, "подскажи": 2000,
        "определи": 2000, "укажи": 2000, "выведи": 1500,
        "составь": 1500, "подготовь": 1500, "переведи": 1000,
        "подсчитай": 1000, "вычисли": 1000, "отсортируй": 800,
        "сгруппируй": 800, "отфильтруй": 800,

        # Существительные для RAG
        "документ": 8000, "файл": 7000, "таблица": 6000,
        "диаграмма": 3000, "график": 4000, "отчёт": 5000,
        "презентация": 3000, "статистика": 3000,
    }

    # Множество уже имеющихся слов
    existing = {w for w, _ in words}

    for word, freq in rag_specific.items():
        if word not in existing:
            words.append((word, freq))

    return words


def download_dictionary() -> None:
    """Основная функция: скачать и сохранить словарь."""
    output_path = _get_output_path()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=== Downloading Russian frequency dictionary ===")
    print()

    words = []

    # Пробуем основной источник (50k слов)
    try:
        raw = _download(FREQ_WORDS_URL)
        words = _parse_freq_words(raw)
        print(f"  ✓ Parsed: {len(words)} words from 50k list")
    except Exception as e:
        print(f"  ✗ Failed (50k): {e}")

        # Пробуем полный словарь
        try:
            raw = _download(FREQ_WORDS_FULL_URL)
            words = _parse_freq_words(raw)
            # Берём только топ-50k
            words.sort(key=lambda x: -x[1])
            words = words[:50000]
            print(f"  ✓ Parsed: {len(words)} words from full list")
        except Exception as e2:
            print(f"  ✗ Failed (full): {e2}")
            print()
            print("ERROR: Could not download dictionary.")
            print("Check your internet connection and try again.")
            print()
            print("Alternative: manually download from")
            print(f"  {FREQ_WORDS_URL}")
            print(f"  and save as {output_path}")
            sys.exit(1)

    # Добавляем RAG-специфичные слова
    words = _add_rag_words(words)

    # Сортируем по частоте (от частых к редким)
    words.sort(key=lambda x: -x[1])

    # Убираем дубликаты
    seen = set()
    unique_words = []
    for word, freq in words:
        if word not in seen:
            seen.add(word)
            unique_words.append((word, freq))

    # Сохраняем
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Russian frequency dictionary\n")
        f.write(f"# Source: {FREQ_WORDS_URL}\n")
        f.write(f"# Words: {len(unique_words)}\n")
        f.write("# Format: word<TAB>frequency\n")
        f.write("#\n")
        for word, freq in unique_words:
            f.write(f"{word}\t{freq}\n")

    size_kb = os.path.getsize(output_path) / 1024
    print()
    print(f"✓ Saved: {output_path}")
    print(f"  Words: {len(unique_words)}")
    print(f"  Size:  {size_kb:.1f} KB")


if __name__ == "__main__":
    download_dictionary()