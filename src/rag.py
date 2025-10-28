import os
import re
import pickle
from collections import Counter, defaultdict
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Простой список стоп-слов на русском языке
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


def summarize_all_in_one(vectorstore, model_name, use_gpu=True):
    cache_file = "summary_cache.pkl"

    # Проверяем кэш
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except:
            pass  # если кэш битый — пересчитаем

    llm = ChatOllama(model=model_name, num_gpu=1 if use_gpu else 0, temperature=0.1)

    # Собираем уникальные файлы и короткие превью
    seen = set()
    previews = []
    for doc in vectorstore.docstore._dict.values():
        source = doc.metadata.get("source")
        if source not in seen:
            seen.add(source)
            text = doc.page_content[:600]
            name = os.path.basename(source)
            previews.append(f"[Файл: {name}]\n{text}\n")

    context = "\n".join(previews)[:6000]  # Ограничиваем

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

        # Сохраняем в кэш
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result
    except Exception as e:
        return f"Ошибка суммирования: {e}"

def extract_keywords_from_query(query, top_n=5, min_length=4):
    """
    Извлекает ключевые слова из запроса пользователя
    """
    query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
    words = query_clean.split()

    filtered_words = [
        word for word in words
        if len(word) >= min_length and word not in RUSSIAN_STOP_WORDS
    ]

    word_freq = Counter(filtered_words)
    total_words = len(filtered_words)

    word_weights = {}
    for word, freq in word_freq.items():
        tf = freq / total_words if total_words > 0 else 0
        idf_like = np.log(len(word) + 1) * np.log(freq + 1)
        word_weights[word] = tf * idf_like

    sorted_keywords = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)
    keywords = [keyword for keyword, _ in sorted_keywords[:top_n]]

    return keywords


def format_context_with_sources(docs, query):
    """
    Форматирует контекст, включая только релевантные чанки на основе извлечённых ключевых слов
    """
    keywords = extract_keywords_from_query(query, top_n=5)

    formatted_chunks = []
    for doc in docs:
        doc_content = doc.page_content.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in doc_content)

        if keyword_matches > 0:
            source_path = doc.metadata.get("source", "Неизвестный источник")
            source_name = os.path.basename(source_path)
            chunk = f"[Источник: {source_name}]\n{doc.page_content}\n"
            formatted_chunks.append(chunk)

    return "\n\n".join(formatted_chunks) if formatted_chunks else "Нет релевантного контекста"


def summarize_documents(vectorstore, model_name, use_gpu=True):
    """
    Создает краткое описание содержимого всех документов в индексе.
    Возвращает список вида: [{"file": имя_файла, "summary": краткое_описание}, ...]
    """
    llm = ChatOllama(
        model=model_name,
        num_gpu=1 if use_gpu else 0,
        temperature=0.1,
    )

    # Шаблон для суммирования
    summary_prompt = ChatPromptTemplate.from_template(
        """Суммаризируй следующий текст в 1-2 предложениях, выделяя основную тему или содержание. 
        Игнорируй метаданные, такие как [Источник: имя_файла]. 
        Текст: {text}

        Краткое описание:"""
    )

    # Группируем чанки по файлам
    documents = vectorstore.docstore._dict.values()  # Получаем все документы из FAISS
    file_chunks = defaultdict(list)
    for doc in documents:
        source = doc.metadata.get("source", "Неизвестный источник")
        file_chunks[source].append(doc.page_content)

    summaries = []
    for source, chunks in file_chunks.items():
        # Объединяем чанки одного файла, но ограничиваем объем (например, 2000 символов)
        combined_text = "\n".join(chunks)[:2000]
        source_name = os.path.basename(source)

        # Суммируем содержимое
        try:
            response = llm.invoke(summary_prompt.format(text=combined_text))
            summary = response.content.strip()
        except Exception as e:
            summary = f"Ошибка при суммировании: {e}"

        summaries.append({"file": source_name, "summary": summary})

    return summaries


def get_rag_chain(vectorstore, model_name, use_gpu=True, embedding_model="embeddinggemma:latest"):
    llm = ChatOllama(model=model_name, num_gpu=1 if use_gpu else 0, temperature=0.1)

    prompt = ChatPromptTemplate.from_template(
        """Ты интеллектуальный ассистент. Используй только предоставленный контекст для ответа на вопрос.

Инструкции:
- Отвечай только на основе содержимого документов, игнорируя строки с метаданными в формате [Источник: имя_файла]
- Если информации нет в содержимом контекста, скажи "Информация отсутствует в доступных документах"
- Не включай в ответ метаданные, такие как [Источник: имя_файла] или номера документов, если об этом не просит пользователь
- Отвечай кратко и по существу, используя только релевантную информацию из текста документа

Контекст:
{context}

Вопрос: {input}

Ответ:"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = create_retrieval_chain(retriever, document_chain)

    def wrapped_qa_chain(query, file_filter=None):
        query_lower = query.lower().strip()

        # === РАСШИРЕННЫЕ ПАТТЕРНЫ ===
        general_patterns = [
            "о чём файлы", "что в файлах", "опиши файлы",
            "о чем файлы", "о чём эти файлы", "о чём все файлы",
            "что в этих файлах", "опиши содержимое", "о чём документы"
        ]
        is_general = any(p in query_lower for p in general_patterns)

        if is_general:
            summary = summarize_all_in_one(vectorstore, model_name, use_gpu)
            return {
                "result": f"Краткое описание файлов:\n{summary}",
                "source_documents": [],
                "sources": "все файлы",
                "keywords": [],
                "formatted_context": ""
            }

        # Обработка специфичных запросов
        if file_filter:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3, "filter": {"source": file_filter}}
            )
            qa_chain.retriever = retriever

        # Выполняем запрос
        result = qa_chain.invoke({"input": query})
        source_documents = result["context"]
        keywords = extract_keywords_from_query(query, top_n=5)
        formatted_context = format_context_with_sources(source_documents, query)

        # Повторно выполняем запрос с отформатированным контекстом
        final_result = qa_chain.invoke({
            "input": query,
            "context": formatted_context
        })

        answer = final_result["answer"]

        relevant_sources = set()
        for doc in source_documents:
            doc_content = doc.page_content.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in doc_content)
            if keyword_matches > 0:
                source_path = doc.metadata.get("source", "Неизвестный источник")
                source_name = os.path.basename(source_path)
                relevant_sources.add(source_name)

        print(f"Query: {query}")
        print(f"Extracted keywords: {keywords}")
        for doc in source_documents:
            source_name = os.path.basename(doc.metadata.get("source", "Неизвестный источник"))
            doc_content = doc.page_content.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in doc_content)
            print(f"Chunk from {source_name} (matches: {keyword_matches}): {doc.page_content[:100]}...")

        return {
            "result": answer,
            "source_documents": source_documents,
            "sources": ", ".join(relevant_sources) if relevant_sources else "Нет релевантных источников",
            "keywords": keywords,
            "formatted_context": formatted_context[:500] + "..." if len(formatted_context) > 500 else formatted_context
        }

    return wrapped_qa_chain