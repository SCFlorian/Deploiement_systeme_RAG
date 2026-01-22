# ======================
# Libraires nécessaires
# ======================
import pandas as pd
import pytest
import sys, os

# Ajout du chemin pour trouver build_index.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Imports nécessaires
from scripts.build_index import (
    prepare_dataframe,
    del_col_business_side,
    del_col_missed_val,
    col_rename,
    short_version,
    long_version)

# Test de la préparation des documents
def test_preprocessing_pipeline_output_consistency():
    df = prepare_dataframe()
    df_clean = del_col_business_side(df)
    df_clean = del_col_missed_val(df_clean)

    df_base, df_short, df_long = col_rename(df_clean)

    df_short = short_version(df_short)
    df_long = long_version(df_long)

    assert not df_short.empty
    assert not df_long.empty
    assert "text_for_embedding" in df_short.columns
    assert "text_for_embedding" in df_long.columns

# Test du bon nettoyage des balise HTML sur la description longue
def test_long_version_html_is_cleaned():
    df = pd.DataFrame({
        "title": ["Expo"],
        "longdescription": ["<p>Texte <b>important</b></p>"],
        "conditions": ["Libre"],
        "location_name": ["Paris"],
        "daterange": ["2025"]
    })

    df_long = long_version(df)

    text = df_long.loc[0, "text_for_embedding"]

    assert "<" not in text and ">" not in text, "Le HTML n'a pas été nettoyé"
    assert "Texte" in text

# Test de bon renommage de certaines colonnes
def test_col_rename_columns_and_outputs():
    df = pd.DataFrame({
        "title_fr": ["event"],
        "description_fr": ["desc"],
        "conditions_fr": ["free"],
        "keywords_fr": ["music"],
        "daterange_fr": ["2025"]
    })

    df_new, df_short, df_long = col_rename(df)

    expected_cols = {"title", "description", "conditions", "keywords", "daterange"}

    assert expected_cols.issubset(df_new.columns)
    assert df_new.equals(df_short)
    assert df_new.equals(df_long)
