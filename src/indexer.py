# indexer.py
import os
import time
import pickle
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import SUPPORTED_FORMATS, EMBEDDING_MODEL
from cache import get_folder_cache_dir
import PyPDF2
from docx import Document as DocxDocument
import bs4
import easyocr

# Глобальный OCR reader
ocr_reader = None

def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        print("Загрузка EasyOCR (один раз, может занять 10-20 сек)...")
        ocr_reader = easyocr.Reader(['ru', 'en'], gpu=True)
    return ocr_reader

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    print(f"  → extract_text: {os.path.basename(file_path)} [{ext}]")
    if ext not in SUPPORTED_FORMATS:
        return ""

    try:
        # === OCR для изображений ===
        if ext in ["png", "jpg", "jpeg"]:
            reader = get_ocr_reader()
            result = reader.readtext(file_path, detail=0, paragraph=True)
            text = "\n".join(result) if result else "Текст не распознан"
            return text

        # === PDF ===
        elif ext == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "".join(page.extract_text() or "" for page in reader.pages)

        # === TXT / MD ===
        elif ext in ["txt", "md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        # === DOCX ===
        elif ext == "docx":
            doc = DocxDocument(file_path)
            return "\n".join(para.text for para in doc.paragraphs)

        # === HTML ===
        elif ext == "html":
            with open(file_path, "r", encoding="utf-8") as f:
                soup = bs4.BeautifulSoup(f.read(), "html.parser")
                return soup.get_text()

    except Exception as e:
        logging.error(f"Ошибка извлечения текста из {file_path}: {e}")
        return ""

    return ""

# src/indexer.py
def build_index(folder_path, embedding_model):
    # Store index and metadata in centralized per-folder cache directory
    cache_dir = get_folder_cache_dir(folder_path)
    index_path = os.path.join(cache_dir, "faiss_index")
    timestamp_path = os.path.join(cache_dir, "file_timestamps.pkl")

    # === 1. Сбор файлов ===
    print(f"[INDEXER] Сканирование папки: {folder_path}")
    current_timestamps = {}
    supported_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            if ext in SUPPORTED_FORMATS:
                current_timestamps[path] = os.path.getmtime(path)
                supported_files.append(path)

    print(f"[INDEXER] Найдено файлов: {len(supported_files)}")

    # === 2. Проверка кэша ===
    try:
        with open(timestamp_path, "rb") as f:
            previous_timestamps = pickle.load(f)
    except FileNotFoundError:
        previous_timestamps = {}

    needs_reindex = len(current_timestamps) != len(previous_timestamps)
    if not needs_reindex:
        for path, mtime in current_timestamps.items():
            if path not in previous_timestamps or previous_timestamps[path] != mtime:
                needs_reindex = True
                break

    if os.path.exists(index_path) and not needs_reindex:
        print("[INDEXER] Загрузка кэша...")
        embeddings = OllamaEmbeddings(model=embedding_model)
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("[INDEXER] Кэш загружен!")
        return vectorstore

    print("[INDEXER] Построение нового индекса...")

    # === 3. Извлечение текста ===
    texts = []
    metadatas = []
    for i, path in enumerate(supported_files):
        print(f"[INDEXER] [{i+1}/{len(supported_files)}] Обработка: {os.path.basename(path)}")
        text = extract_text(path)
        text.strip()
        texts.append(text)
        metadatas.append({"source": path, "type": "image" if os.path.splitext(path)[1].lower().lstrip(".") in ["png", "jpg", "jpeg"] else "document"})
    else:
        print(f"  → Текст пустой")

    if not texts:
        print("[INDEXER] Нет текста для индексации")
        return None

    # === 4. Разбиение ===
    print(f"[INDEXER] Разбиение на чанки: {len(texts)} документов")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    split_texts = []
    split_metadatas = []
    total_chunks = 0
    for i, (text, meta) in enumerate(zip(texts, metadatas)):
        chunks = text_splitter.split_text(text)
        total_chunks += len(chunks)
        split_texts.extend(chunks)
        split_metadatas.extend([meta] * len(chunks))
        print(f"  → Док {i+1}: {len(chunks)} чанков")

    print(f"[INDEXER] Всего чанков: {total_chunks}")

    # === 5. Эмбеддинги ===
    print("[INDEXER] Загрузка модели эмбеддингов...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    print("[INDEXER] Создание FAISS...")
    vectorstore = FAISS.from_texts(
        texts=split_texts,
        embedding=embeddings,
        metadatas=split_metadatas
    )

    # === 6. Сохранение ===
    print("[INDEXER] Сохранение индекса в:", index_path)
    vectorstore.save_local(index_path)
    with open(timestamp_path, "wb") as f:
        pickle.dump(current_timestamps, f)

    print("[INDEXER] Индексация завершена!")
    return vectorstore