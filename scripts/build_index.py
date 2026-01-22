# =======================
# Librairies nécessaires
# =======================
import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from langchain_mistralai.embeddings import MistralAIEmbeddings

from rag.data_loader import prepare_dataframe
from rag.data_checks import check_city_filter
from rag.preprocessing import (
    del_col_business_side,
    del_col_missed_val,
    col_rename,
    short_version,
    long_version
)
from rag.embeddings import generation_embeddings, generation_embeddings_short, create_embeddings
from rag.vectorstore import build_vectorstore
from rag.similarity_test import cos_test

# =====================================
# Définition minimale de notre logging
# =====================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()


def data_update():
    # ===========================
    # Extraction & preprocessing
    # ===========================
    logging.info("Extraction OpenAgenda")
    df = prepare_dataframe()

    logging.info("Vérification du filtre Paris")
    check_city_filter(df, "Paris")

    logging.info("Nettoyage des variables métier")
    df_clean = del_col_business_side(df)

    logging.info("Nettoyage des variables data")
    df_clean = del_col_missed_val(df_clean)

    logging.info("Renommage des colonnes et création des versions")
    _, df_short, df_long = col_rename(df_clean)

    df_short = short_version(df_short)
    df_long = long_version(df_long)

    # ===================
    # Création documents
    # ===================
    logging.info("Création des documents (version longue)")
    documents_long = generation_embeddings(df_long)

    logging.info("Création des documents (version courte)")
    documents_short = generation_embeddings_short(df_short)

    # ===================
    # Embeddings
    # ===================
    embeddings = create_embeddings()

    # ==================
    # Vectorstores FAISS
    # ==================
    logging.info("Construction du vectorstore FAISS (long)")
    vectorstore_long = build_vectorstore(
        documents_long,
        embeddings,
        "faiss_index_long"
    )

    logging.info("Construction du vectorstore FAISS (short)")
    vectorstore_short = build_vectorstore(
        documents_short,
        embeddings,
        "faiss_index_short"
    )

    # ==================
    # Tests de similarité
    # ==================
    #logging.info("Test de similarité (short)")
    #sim_short = cos_test(vectorstore_short)
    #logging.info(f"Similarité short : {sim_short}")

    #logging.info("Test de similarité (long)")
    #sim_long = cos_test(vectorstore_long)
    #logging.info(f"Similarité long : {sim_long}")


if __name__ == "__main__":
    data_update()
