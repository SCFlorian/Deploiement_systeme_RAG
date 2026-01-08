# Librairies nécessaires
from mistralai.client import MistralClient
import os
from dotenv import load_dotenv
from mistralai import Mistral
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # ou SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================================================================================
# Génération des embeddings avec le modèle d'embeddings de Mistral, mistral-embed
# Recommandé par la mission
# ================================================================================

def generation_embeddings(df_long):
    # On active notre client mistral
    client = Mistral(
        api_key=os.getenv("MISTRAL_KEY")
    )

    # On mets nos textes dans une liste avec une limitation de token par Mistral
    texts = df_long["text_for_embedding"].astype(str).tolist()[:50]

    # Application d'une stratégie de découpage avec RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Taille de chaque segment
        chunk_overlap=50 # Chevauchement entre les segments
    )
    documents = text_splitter.create_documents(
        texts=texts,
        metadatas=[
            {"Date de fin": df_long.iloc[i]["lastdate_end"],
             "Lieu": df_long.iloc[i]["location_name"],
             "Adresse postale": df_long.iloc[i]["location_address"],
             }
            for i in range(len(texts))
        ]
    )
    texts_for_embedding = [doc.page_content for doc in documents]
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts_for_embedding
    )
    # TOUS les embeddings
    vectors_long = np.array([item.embedding for item in response.data], dtype="float32")
    
    return vectors_long, documents, client

# ===================================
# Test de similarité avec le cosinus
# ===================================
def cos_test(vectors_long):
    # Embeddings de deux événements différents
    vec1 = np.array(vectors_long[0])
    vec2 = np.array(vectors_long[1])

    similarite = np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )
    return similarite