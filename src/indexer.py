import os
import time
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from config import SUPPORTED_FORMATS, EMBEDDING_MODEL
import PyPDF2
from docx import Document as DocxDocument
import bs4

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        return ""

    try:
        if ext == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() or "" for page in reader.pages)
        elif ext == "txt" or ext == "md":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext == "docx":
            doc = DocxDocument(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        elif ext == "html":
            with open(file_path, "r", encoding="utf-8") as f:
                soup = bs4.BeautifulSoup(f.read(), "html.parser")
                text = soup.get_text()
        return text
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return ""

def build_index(folder_path, embedding_model):
    index_path = "faiss_index"
    timestamp_path = "file_timestamps.pkl"

    # Проверяем временные метки файлов
    current_timestamps = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            if os.path.splitext(path)[1].lower().lstrip(".") in SUPPORTED_FORMATS:
                current_timestamps[path] = os.path.getmtime(path)

    # Загружаем предыдущие временные метки
    try:
        with open(timestamp_path, "rb") as f:
            previous_timestamps = pickle.load(f)
    except FileNotFoundError:
        previous_timestamps = {}

    # Сравниваем метки времени
    needs_reindex = False
    for path, mtime in current_timestamps.items():
        if path not in previous_timestamps or previous_timestamps[path] != mtime:
            needs_reindex = True
            break

    # Если индекс существует и файлы не изменились
    if os.path.exists(index_path) and not needs_reindex:
        print("Loading existing index...")
        embeddings = OllamaEmbeddings(model=embedding_model)
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    # Если нужно перестроить индекс
    print("Building new index...")
    texts = []
    metadatas = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            text = extract_text(path)
            if text:
                texts.append(text)
                metadatas.append({"source": path})

    if not texts:
        return None

    # Разбиваем текст каждого файла отдельно
    chunk_size = 1500
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=150)
    split_texts = []
    split_metadatas = []
    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        split_texts.extend(chunks)
        split_metadatas.extend([metadatas[i]] * len(chunks))

    # Проверяем метаданные
    for meta in split_metadatas:
        if "source" not in meta:
            print(f"Warning: Missing source metadata for chunk")

    # Загружаем эмбеддинги через Ollama
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_texts(
        texts=split_texts,
        embedding=embeddings,
        metadatas=split_metadatas
    )
    vectorstore.save_local(index_path)

    # Сохраняем новые временные метки
    with open(timestamp_path, "wb") as f:
        pickle.dump(current_timestamps, f)

    return vectorstore