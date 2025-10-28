# main.py
import sys
import time
import psutil
import os
from config import MODEL_NAME, EMBEDDING_MODEL
from indexer import build_index
from rag import get_rag_chain

def detect_power_source():
    try:
        battery = psutil.sensors_battery()
        return battery is None or battery.power_plugged
    except:
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    use_gpu = detect_power_source()
    power_mode = "GPU mode (AC power)" if use_gpu else "CPU mode (Battery)"

    print(f"Building or loading index... ({power_mode})")
    vectorstore = build_index(folder_path, EMBEDDING_MODEL)

    if vectorstore is None:
        print("No supported documents found.")
        sys.exit(1)

    qa_chain = get_rag_chain(vectorstore, MODEL_NAME, use_gpu=use_gpu, folder_path=folder_path)

    print(f"Index built/loaded. Ask questions: {power_mode}")
    print("-" * 50)

    while True:
        query = input("Question (or 'exit'): ")
        if query.lower() == "exit":
            break

        start_time = time.time()
        file_filter = None
        if " в " in query.lower():
            parts = query.lower().split(" в ")
            file_name = parts[1].strip().strip('"\'')
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                file_filter = file_path
                print(f"Фильтр: {os.path.basename(file_path)}")
            else:
                print(f"Файл не найден: {file_name}")

        response = qa_chain(query, file_filter=file_filter)
        answer = response["result"]
        sources = response["sources"]

        print("Retrieved documents:")
        for doc in response["source_documents"]:
            source_name = os.path.basename(doc.metadata.get('source', 'Неизвестный источник'))
            doc_type = doc.metadata.get('type', 'document')
            print(f" - {source_name} ({doc_type}): {doc.page_content[:100]}...")

        print(f"Context preview: {response['formatted_context']}")
        print(f"Sources: {sources}")
        print(f"Answer: {answer}")
        print(f"Response time: {time.time() - start_time:.2f} seconds ({power_mode})")
        print("-" * 50)