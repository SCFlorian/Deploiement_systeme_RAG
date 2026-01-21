# ======================
# Libraires nécessaires
# ======================
import os
import pytest
# Imports nécessaires
from rag.data_loader import prepare_dataframe
from rag.data_checks import check_city_filter
from rag.preprocessing import (
    del_col_business_side,
    del_col_missed_val,
    col_rename,
    short_version,
    long_version
)
from rag.embeddings import generation_embeddings, create_embeddings
from rag.vectorstore import build_vectorstore

# ==========================
# Définition de la fonction
# ==========================
def test_real_pipeline_indexation_and_retrieval(tmp_path):
    
    # ===========================
    # Pipeline réel
    # ===========================
    df = prepare_dataframe()
    check_city_filter(df, "Paris")

    df_clean = del_col_business_side(df)
    df_clean = del_col_missed_val(df_clean)

    _, _, df_long = col_rename(df_clean)
    df_long = long_version(df_long)

    documents = generation_embeddings(df_long)

    # Sécurité
    assert len(documents) > 50, "Pas assez de documents générés"

    # ===========================
    # Indexation FAISS réelle
    # ===========================
    embeddings = create_embeddings()
    # Base temporaire, permet de valider le processus
    index_path = tmp_path / "faiss_real_test"

    vectorstore = build_vectorstore(
        documents,
        embeddings,
        str(index_path)
    )

    # ===========================
    # 3. Requête métier réelle
    # ===========================
    query = "concert à Paris"
    results = vectorstore.similarity_search(query, k=5)

    # ===========================
    # 4. Validation indexation
    # ===========================
    assert len(results) > 0, "Aucun document retourné"

    assert any(
        "paris" in doc.page_content.lower()
        for doc in results
    ), "Les documents retournés ne semblent pas pertinents"
