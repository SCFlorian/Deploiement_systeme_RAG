# GÉNÈRATION DU LLM
# =======================
# Librairies nécessaires
# =======================
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
# ==================================
# On charge notre environnement env
# ==================================
load_dotenv()
# On recharge les éléments stockés dans env
SEUIL_RAG = float(os.getenv("SEUIL_RAG"))
k = int(os.getenv("k"))
model = os.getenv("model")
# ===============================================================
# Fonction de génération de la fonction pour se connecter au LLM
# ===============================================================
def generate_LLM(index_name: str):
    # Embeddings
    embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_KEY")
    )
    # Charger FAISS
    vectorstore = FAISS.load_local(
        index_name,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    # Récupération des informations dans notre base FAISS
    retriever = vectorstore.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs={"k": k,
                       "score_threshold": SEUIL_RAG}
    )
    
    # ============================
    # LLM Mistral
    # ============================
    llm = ChatMistralAI(
        model=model,
        temperature=0.2
    )
    # Afin d'avoir les métadonnées enregistrées dans nos documents
    document_prompt = PromptTemplate.from_template(
        """
        [Événement trouvé]
        Date: Du {Date de début} au {Date de fin}
        Lieu: {Lieu} - {Adresse postale}
        Détails:
        {page_content}
        ---------------------------------------------------
        """
    )

    # ============================
    # Prompt RAG
    # ============================
    prompt = ChatPromptTemplate.from_template("""
Tu es un assistant culturel.
Tu réponds uniquement à partir des événements fournis ci-dessous.
Mais répond de manière naturelle et courte. Ne pas d'éléments en gras.

{context}

Question :
{input}

Si l'information n'est pas présente dans les événements, dis-le clairement.
Réponds en français, de manière concise.
""")

    # ============================
    # Chaînes LangChain
    # ============================
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=document_prompt,
        document_variable_name="context"
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    return rag_chain
