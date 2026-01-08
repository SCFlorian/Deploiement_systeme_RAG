from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import numpy as np
import faiss

# =================================================
# CRÉEZ ET INTERROGEZ L'INDEX FAISS AVEC LANGCHAIN
# =================================================

# ===============================================
# Initialisation et sauvegarde de l'index Faiss
# ===============================================

def build_vectorstore(documents, client):
    embeddings = MistralEmbeddings(client)

    vectorstore = FAISS.from_documents(
        documents,
        embedding=embeddings
    )

    vectorstore.save_local("faiss_index")
    return vectorstore

# ==================
# Tests d'une query
# ==================

def generate_query(vectorstore):
    query = "évènement dans le 15ème arrondissement"

    results = vectorstore.similarity_search(query, k=5)

    for doc in results:
        print("----")
        print(doc.page_content[:200])
        print(doc.metadata)

# ========================================================================
# Création de la classe MistralEmbeddings pour l'intégration de Langchain
# ========================================================================

class MistralEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=[text]
        )
        return response.data[0].embedding
