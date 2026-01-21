# =======================
# Librairies nécessaires
# =======================
from rag.connexion_llm import generate_LLM

# Transformation de la question en vecteurs
# On recharge notre base d'index FAISS
# Mise en place du retrieval
# Connexion au LLM mistral-small-latest
# Définition du prompt
# Mise en place de chaîne langchain avec LCEL(create_stuff_documents_chain) avec comme helper create_rerieval_chain

def answer_chat(question):
    rag_chain = generate_LLM("faiss_index_short")

    response = rag_chain.invoke({
        "input": question
    })

    return response["answer"]

question = "Quels sont les concerts en 2026 ?"
answer = answer_chat(question)
print(answer)