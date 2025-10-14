import sys
import time
import psutil
from config import MODEL_NAME, EMBEDDING_MODEL
from indexer import build_index
from rag import get_rag_chain


def detect_power_source():
    """
    Определяет тип электропитания: True = от сети (GPU), False = от аккумулятора (CPU)
    """
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            # Нет батареи (десктоп) = всегда GPU
            return True

        if battery.power_plugged:
            return True  # От сети = GPU
        else:
            return False  # От аккумулятора = CPU
    except Exception:
        # Ошибка = по умолчанию GPU
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    # Определяем тип питания и режим
    use_gpu = detect_power_source()
    power_mode = "GPU mode (AC power)" if use_gpu else "CPU mode (Battery)"

    print(f"Building or loading index... ({power_mode})")
    vectorstore = build_index(folder_path, EMBEDDING_MODEL)

    if vectorstore is None:
        print("No supported documents found.")
        sys.exit(1)

    qa_chain = get_rag_chain(vectorstore, MODEL_NAME, use_gpu=use_gpu)

    print(f"Index built/loaded. Ask questions: {power_mode}")
    print("-" * 50)

    while True:
        query = input("Question (or 'exit'): ")
        if query.lower() == "exit":
            break

        start_time = time.time()
        if " в " in query:
            file_name = query.split(" в ")[1].strip()
            filtered_retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3, "filter": {"source": {"$contains": file_name}}}
            )
            qa_chain.retriever = filtered_retriever

        # Используем invoke вместо run (без предупреждений)
        response = qa_chain.invoke({"query": query})["result"]

        elapsed = time.time() - start_time
        print(f"Answer: {response}")
        print(f"Response time: {elapsed:.2f} seconds ({power_mode})")
        print("-" * 50)