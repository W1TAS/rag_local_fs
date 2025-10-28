# indexer.py
import os
import time
import pickle
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import SUPPORTED_FORMATS, EMBEDDING_MODEL
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

def build_index(folder_path, embedding_model):
    index_path = "faiss_index"
    timestamp_path = "file_timestamps.pkl"

    # Считываем текущие файлы
    current_timestamps = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            if ext in SUPPORTED_FORMATS:
                current_timestamps[path] = os.path.getmtime(path)

    # Загружаем старые метки
    try:
        with open(timestamp_path, "rb") as f:
            previous_timestamps = pickle.load(f)
    except FileNotFoundError:
        previous_timestamps = {}

    # Нужно ли переиндексировать?
    needs_reindex = False
    for path, mtime in current_timestamps.items():
        if path not in previous_timestamps or previous_timestamps[path] != mtime:
            needs_reindex = True
            break
    if len(current_timestamps) != len(previous_timestamps):
        needs_reindex = True

    if os.path.exists(index_path) and not needs_reindex:
        print("Loading existing index...")
        embeddings = OllamaEmbeddings(model=embedding_model)
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    print("Building new index...")
    texts = []
    metadatas = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            text = extract_text(path)
            if text.strip():
                texts.append(text)
                metadatas.append({"source": path, "type": "image" if os.path.splitext(path)[1].lower().lstrip(".") in ["png", "jpg", "jpeg"] else "document"})

    if not texts:
        return None

    # Разбиваем текст
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    split_texts = []
    split_metadatas = []
    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        split_texts.extend(chunks)
        split_metadatas.extend([metadatas[i]] * len(chunks))

    # Создаём индекс
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_texts(
        texts=split_texts,
        embedding=embeddings,
        metadatas=split_metadatas
    )
    vectorstore.save_local(index_path)

    # Сохраняем метки
    with open(timestamp_path, "wb") as f:
        pickle.dump(current_timestamps, f)

    return vectorstore