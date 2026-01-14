# Librairies nécessaires
from rag.connexion_llm import generate_LLM

# Transformation de la question en vecteurs
# On recharge notre base d'index FAISS
# Mise en place du retrieval
# Connexion au LLM mistral-small-latest
# Définition du prompt
# Mise en place de chaîne langchain avec LCEL(create_stuff_documents_chain) avec comme helper create_rerieval_chain

rag_chain = generate_LLM("faiss_index_short")

response = rag_chain.invoke({
    "input": "J'aimrais aller voir une pièce de théâtre, que peux-tu me conseiller ?"
})

print(response["answer"])
