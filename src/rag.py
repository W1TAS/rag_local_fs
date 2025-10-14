from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

def get_rag_chain(vectorstore, model_name, use_gpu=True):
    """
    Создает RAG-цепочку с автоматическим выбором GPU/CPU
    use_gpu: True = GPU (сеть), False = CPU (аккумулятор)
    """
    llm = OllamaLLM(
        model=model_name,
        num_gpu=1 if use_gpu else 0  # 1 = GPU, 0 = CPU
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    )
    return qa_chain