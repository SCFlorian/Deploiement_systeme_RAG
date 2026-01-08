
# =======================
# Librairies nécessaires
# =======================
import os
from dotenv import load_dotenv
import faiss
from langchain_community.vectorstores import FAISS

# ==========================
# Importation des fonctions
# ==========================
from rag.data_loader import prepare_dataframe
from rag.data_checks import check_city_filter
from rag.preprocessing import del_col_business_side
from rag.preprocessing import del_col_missed_val
from rag.preprocessing import col_rename
from rag.preprocessing import short_version
from rag.preprocessing import long_version
from rag.embeddings import generation_embeddings
from rag.embeddings import cos_test
from rag.vectorstore import build_vectorstore
from rag.vectorstore import generate_query

# ====================================================================
# Script global pour vérifier que toutes les fonctions sont correctes
# ====================================================================

# On charge notre clé API Mistral depuis env
load_dotenv()

# En attendant la création d'une API, on peut créer une fonction main
def main():
    # On récupère les informations de l'api open agenda
    print("Extraction OpenAgenda")
    df = prepare_dataframe()

    print("Aperçu des données")
    print(df[["location_city", "lastdate_end"]].head())

    # On fait un petit check pour s'assurer que nous avons que Paris dans le df
    check_city_filter(df, "Paris")
    print(df.shape)

    # On supprime les colonnes non utiles d'un point de vu métier
    df_test_clean = del_col_business_side(df)

    print(df_test_clean.shape)

    # On supprime les colonnes avec plus de 70% de données manquantes (hors âge min car potentiellement utile)
    df_firstclean = del_col_missed_val(df_test_clean)
    print(df_firstclean.shape)

    # Création d'une seule colonne avec les informations utiles
    # Deux versions dont une pour le côté pédagogique
    df_firstclean_new, df_short, df_long = col_rename(df_firstclean)

    df_short = short_version(df_short)
    print(df_short["text_for_embedding"][0])

    df_long = long_version(df_long)
    print(df_long["text_for_embedding"][0])

    # Génération des embeddings avec mistral-embed (pour la version longue)
    vectors_long, documents, client = generation_embeddings(df_long)
    print(documents[0])
    print(type(vectors_long))
    print(len(vectors_long))
    print(len(documents))

    # Test de similarité avec cosinus
    similarite = cos_test(vectors_long)
    print(similarite)

    # Création de notre Index Fais et de la sauvegarde de ce dernier avec l'intégration via Langchain
    vectorstore, embeddings = build_vectorstore(documents, client)
    # Génération d'une question via l'index pour voit les recherches sont pertinentes
    # generate_query(vectorstore)

    # étape 4 - intégrez langchain pour le système rag

    # On récupère nos index FAISS
    vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
    )
    # On définit notre récupération
    retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
    )
    # Définition de notre question
    question = "Est-ce qu'il y a un opéra de prévu cette année?"

    docs = retriever.invoke(question)

    # Préparation du contexte
    context = ""
    for doc in docs:
        context += f"""
    - {doc.page_content}
    Lieu : {doc.metadata.get("Lieu")}
    Date de fin : {doc.metadata.get("Date de fin")}
    """
    # prompt afin de guider le LLM
    prompt = f"""
    Tu es un assistant culturel.
    Tu réponds uniquement à partir des événements fournis ci-dessous.
    Événements :
    {context}
    Question :
    {question}
    Si l'information n'est pas présente dans les événements, dis-le clairement.
    Réponds en français, de manière concise.
    """

    # Génération de la réponse du LLM via notre RAG
    response = client.chat.complete(
    model="mistral-small",
    messages=[
        {"role": "user", "content": prompt}])

    print("Réponse du chatbot :\n")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()