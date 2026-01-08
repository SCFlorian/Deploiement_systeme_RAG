
# =======================
# Librairies nécessaires
# =======================
import os
from dotenv import load_dotenv
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
    print(f"La liste à regarder\n{df_firstclean.columns}")

    # Création d'une seule colonne avec les informations utiles
    # Deux versions dont une pour le côté pédagogique
    df_firstclean_new, df_short, df_long = col_rename(df_firstclean)
    print(df_firstclean_new.shape)
    print(df_short.columns)
    print(df_long.columns)

    df_short = short_version(df_short)
    print(df_short["text_for_embedding"][0])

    df_long = long_version(df_long)
    print(df_long["text_for_embedding"][0])
    print(df_long["text_for_embedding"][1])

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
    vectorstore = build_vectorstore(documents, client)
    # Génération d'une question via l'index pour voit les recherches sont pertinentes
    generate_query(vectorstore)


if __name__ == "__main__":
    main()


