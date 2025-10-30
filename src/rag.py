# rag.py — ПОЛНЫЙ, С ВЫВОДОМ ВСЕГО ТЕКСТА ИЗ ФАЙЛА
import os
import re
import pickle
import hashlib
from collections import Counter, defaultdict
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import SUPPORTED_FORMATS

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

def summarize_all_in_one(vectorstore, model_name, use_gpu=True, folder_path=None):
    cache_file = "summary_cache.pkl"
    hash_file = "summary_hash.txt"
    current_hash = get_folder_hash(folder_path) if folder_path else ""

    if folder_path and os.path.exists(cache_file) and os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            cached_hash = f.read().strip()
        if cached_hash == current_hash:
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    llm = ChatOllama(model=model_name, num_gpu=1 if use_gpu else 0, temperature=0.1)
    seen = set()
    previews = []
    for doc in vectorstore.docstore._dict.values():
        source = doc.metadata.get("source")
        if source and source not in seen:
            seen.add(source)
            text = doc.page_content[:600]
            name = os.path.basename(source)
            previews.append(f"[Файл: {name}]\n{text}\n")
    context = "\n".join(previews)[:6000]

    prompt = ChatPromptTemplate.from_template(
        """Ты — эксперт по кратким описаниям. 
        Прочитай фрагменты из файлов ниже. Для КАЖДОГО файла дай описание в 1 предложение.

        Формат ответа:
        - имя_файла.pdf: Краткое описание.
        - имя_файла.docx: Ещё одно.

        Файлы:
        {context}

        Ответ:"""
    )

    try:
        response = llm.invoke(prompt.format(context=context))
        result = response.content.strip()
        if folder_path:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            with open(hash_file, "w", encoding="utf-8") as f:
                f.write(current_hash)
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
    llm = ChatOllama(model=model_name, num_gpu=1 if use_gpu else 0, temperature=0.1)

    prompt = ChatPromptTemplate.from_template(
        """Ты — точный ассистент. Отвечай ТОЛЬКО на основе контекста.
Если информации нет — скажи "Информация отсутствует в доступных документах".

Контекст:
{context}

Вопрос: {input}

Ответ:"""
    )

    def wrapped_qa_chain(query, file_filter=None):
        query_lower = query.lower().strip()

        # === ОБЩИЕ ЗАПРОСЫ ===
        general_patterns = [
            "о чём файлы", "что в файлах", "опиши файлы",
            "о чем файлы", "о чём эти файлы", "о чём все файлы",
            "что в этих файлах", "опиши содержимое", "о чём документы"
        ]
        if any(p in query_lower for p in general_patterns):
            summary = summarize_all_in_one(vectorstore, model_name, use_gpu, folder_path=folder_path)
            return {
                "result": f"Краткое описание файлов:\n{summary}",
                "source_documents": [],
                "sources": "все файлы",
                "keywords": [],
                "formatted_context": ""
            }

        # === РЕТРИВЕР ===
        search_kwargs = {"k": 10}
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
        answer = llm.invoke(prompt.format(context=context, input=query)).content.strip()

        sources = {best_file} if best_file else set()

        return {
            "result": answer,
            "source_documents": final_docs,
            "sources": ", ".join(sources),
            "keywords": extract_keywords_from_query(query, top_n=7),
            "formatted_context": context[:500] + "..." if len(context) > 500 else context
        }

    return wrapped_qa_chain