import sys
from config import MODEL_NAME, EMBEDDING_MODEL
from indexer import build_index
from rag import get_rag_chain

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    print("Building or loading index...")
    vectorstore = build_index(folder_path, EMBEDDING_MODEL)

    if vectorstore is None:
        print("No supported documents found.")
        sys.exit(1)

    # Создаем qa_chain один раз
    qa_chain = get_rag_chain(vectorstore, MODEL_NAME)

    print("Index built/loaded. Ask questions:")
    while True:
        query = input("Question (or 'exit'): ")
        if query.lower() == "exit":
            break
        # Проверяем, есть ли уточнение файла (например, "Что в sample.pdf?")
        if " в " in query:
            file_name = query.split(" в ")[1].strip()
            # Создаем новый retriever с фильтром
            filtered_retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3, "filter": {"source": {"$contains": file_name}}})
            qa_chain.retriever = filtered_retriever  # Обновляем retriever
        response = qa_chain.run(query)
        print("Answer:", response)