import os
import re
import pickle
import hashlib
from collections import Counter, defaultdict
import numpy as np
import requests
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import SUPPORTED_FORMATS
from typing import Optional

def _extract_file_from_query(query: str, folder_path: str) -> Optional[str]:
    """Ищет в запросе имя файла и возвращает полный путь, если файл существует."""
    if not folder_path:
        return None

    query_low = query.lower()

    # 1. Файл в кавычках
    import re
    m = re.search(r'[«"\'“”]([^«"\'“”]+\.(?:' + '|'.join(SUPPORTED_FORMATS) + r'))[«"\'“”]', query_low)
    if m:
        name = m.group(1)
    else:
        # 2. Паттерн "в файле имя.pdf", "про файл имя.docx" и т.д.
        m = re.search(r'(?:в|по|из|про)\s+(?:файл[ае]?|документ[ае]?)\s+["\']?([^\s"\'.,;]+(?:\.(?:' + '|'.join(SUPPORTED_FORMATS) + r')))', query_low)
        if m:
            name = m.group(1)
        else:
            return None

    # Ищем точное совпадение среди реальных файлов
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() == name.lower():
                return os.path.join(root, f)
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

def summarize_all_in_one(vectorstore, model_name, use_gpu=True, folder_path=None, file_filter=None):
    cache_file = "summary_cache.pkl"
    hash_file = "summary_hash.txt"
    current_hash = get_folder_hash(folder_path) if folder_path else ""
    if folder_path:
        try:
            from cache import get_folder_cache_dir
            folder_cache = get_folder_cache_dir(folder_path)
            cache_file = os.path.join(folder_cache, cache_file)
            hash_file = os.path.join(folder_cache, hash_file)
        except Exception:
            pass

    if folder_path and os.path.exists(cache_file) and os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            cached_hash = f.read().strip()
        if cached_hash == current_hash:
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    if not _ollama_available():
        return "Локальный сервер моделей (Ollama) недоступен. Запустите 'ollama serve' и выполните 'ollama pull' для нужных моделей."
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
    print("\n" + "="*80)
    print(f"ПОЛНЫЙ ТЕКСТ ФАЙЛА: {os.path.basename(file_path)}")
    print("="*80)
    full_text = []
    for doc in vectorstore.docstore._dict.values():
        if doc.metadata.get("source") == file_path:
            full_text.append(doc.page_content)
    print("\n".join(full_text))
    print("\n" + "="*80)
    print(f"Всего чанков: {len(full_text)}")
    print("="*80 + "\n")

# === ОСНОВНАЯ ЦЕПОЧКА ===
def get_rag_chain(vectorstore, model_name, use_gpu=True, embedding_model="embeddinggemma:latest", folder_path=None):
    global llm
    llm = None
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
        """Отвечай на русском языке и строго на предоставленный вопрос. Используй только предоставленный тебе контекст.    

Контекст:
{context}

Вопрос: {input}

Ответ:"""
    )

    def wrapped_qa_chain(query, file_filter=None):
        query_lower = query.lower().strip()

        inferred_file = None
        if not file_filter and folder_path:
            inferred_file = _extract_file_from_query(query, folder_path)

        effective_file = file_filter or inferred_file

        # === ОБЩИЕ ЗАПРОСЫ (О ЧЕМ ФАЙЛЫ) ===
        general_patterns = [
            "о чём файлы", "что в файлах", "опиши файлы",
            "о чем файлы", "о чём эти файлы", "о чём все файлы",
            "что в этих файлах", "опиши содержимое", "о чём документы"
        ]
        if any(p in query_lower for p in general_patterns):
            # Generate summary for all files (works for real folders and virtual single-file folders)
            summary = summarize_all_in_one(vectorstore, model_name, use_gpu, folder_path=folder_path)
            return {
                "result": summary,
                "source_documents": [],
                "sources": "все файлы",
                "keywords": [],
                "formatted_context": ""
            }

        # === РЕТРИВЕР ===
        search_kwargs = {"k": 5}
        if file_filter:
            search_kwargs["filter"] = {"source": file_filter}
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

        # === ВЫБОР ЛУЧШЕГО ФАЙЛА ===
        best_file = select_best_file(raw_docs, query)
        file_path = next((d.metadata["source"] for d in raw_docs if os.path.basename(d.metadata["source"]) == best_file), None)

        # === ВЫВОДИМ ВЕСЬ ТЕКСТ ФАЙЛА В КОНСОЛЬ ===
        # if file_path:
        #     print_full_file_text(vectorstore, file_path)

        # === ФОРМИРУЕМ КОНТЕКСТ ИЗ ЛУЧШЕГО ФАЙЛА ===
        final_docs = [
            doc for doc in raw_docs
            if os.path.basename(doc.metadata.get("source", "")) == best_file
        ]

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
            answer = llm.invoke(prompt.format(context=context, input=query)).content.strip()
        except Exception as e:
            return {
                "result": f"Ошибка при запросе к модели: {e}",
                "source_documents": [],
                "sources": "Нет источников",
                "keywords": extract_keywords_from_query(query, top_n=7),
                "formatted_context": context[:500] + "..." if len(context) > 500 else context
            }

        sources = {best_file} if best_file else set()

        from datetime import datetime

        # Надёжный путь — в папка с приложением или временная папка
        save_dir = os.path.dirname(os.path.abspath(__file__))  # папка, где лежит rag.py
        debug_file = os.path.join(save_dir, "ПОСЛЕДНИЕ_ЧАНКИ.txt")

        try:
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"Время: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write(f"Вопрос: {query}\n")
                f.write("=" * 100 + "\n\n")

                for i, doc in enumerate(raw_docs, 1):
                    filename = os.path.basename(doc.metadata.get("source", "неизвестно"))
                    f.write(f"ЧАНК {i}  ←  {filename}\n")
                    f.write("—" * 80 + "\n")
                    f.write(doc.page_content.strip())
                    f.write("\n\n" + "═" * 80 + "\n\n")

                f.write(f"Всего чанков передано модели: {len(raw_docs)}\n")

            # Дополнительно покажем в приложении, куда сохранили
            print(f"Чанки сохранены: {debug_file}")
        except Exception as e:
            print(f"Не удалось записать чанки: {e}")

        return {
            "result": answer,
            "source_documents": final_docs,
            "sources": ", ".join(sources),
            "keywords": extract_keywords_from_query(query, top_n=7),
            "formatted_context": context[:500] + "..." if len(context) > 500 else context
        }

    return wrapped_qa_chain