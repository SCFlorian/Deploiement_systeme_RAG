from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings


def build_vectorstore(documents, embeddings: Embeddings, index_name: str):
    """
    Construit et sauvegarde un index FAISS à partir de documents LangChain
    """
    vectorstore = FAISS.from_documents(
        documents,
        embedding=embeddings
    )

    vectorstore.save_local(index_name)
    return vectorstore
