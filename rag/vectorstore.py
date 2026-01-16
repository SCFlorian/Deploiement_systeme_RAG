# =======================
# Librairies nécessaires
# =======================
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# Fonction afin de générer les vecteurs et de les sauvegarder
def build_vectorstore(documents, embeddings: Embeddings, index_name: str):
    """
    Construit et sauvegarde un index FAISS à partir de documents LangChain
    Utilisation de l’index par défaut qui est le IndexFlatL2. Il mesure la distance L2 (ou euclidienne)
    entre tous les points donnés de notre vecteur de requête et les vecteurs chargés dans l'index.
    Souvent utilisé pour des recherches similaires à la nôtre.
    """
    vectorstore = FAISS.from_documents(
        documents,
        embedding=embeddings
    )

    vectorstore.save_local(index_name)
    return vectorstore
