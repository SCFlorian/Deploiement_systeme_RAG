# Librairies nécessaires
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()


def generate_LLM(index_name: str):
    # ============================
    # Embeddings (IDENTIQUES)
    # ============================
    embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_KEY")
    )

    # ============================
    # Charger FAISS
    # ============================
    vectorstore = FAISS.load_local(
        index_name,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    # ============================
    # LLM Mistral
    # ============================
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2
    )

    # ============================
    # Prompt RAG
    # ============================
    prompt = ChatPromptTemplate.from_template("""
Tu es un assistant culturel.
Tu réponds uniquement à partir des événements fournis ci-dessous.
Mais répond de manière naturelle. Ne pas d'éléments en gras.

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
        prompt=prompt
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    return rag_chain
