from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

def get_rag_chain(vectorstore, model_name):
    llm = Ollama(model=model_name)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    )
    return qa_chain