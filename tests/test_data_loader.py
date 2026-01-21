# ===================================
# TESTS SUR LA RÉCUPÉRATION DES DONNÉES
# ===================================

# ======================
# Libraires nécessaires
# ======================
import pandas as pd
import pytest
import sys, os

# Ajout du chemin pour trouver build_index.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Imports nécessaires
from build_index import prepare_dataframe

def test_prepare_dataframe_structure():
    """
    On vérifie que la fonction renvoie bien un DataFrame
    avec les bonnes colonnes et les bonnes données.
    """
    # Appel de la fonction
    df = prepare_dataframe()
    # Est-ce un DataFrame ?
    assert isinstance(df, pd.DataFrame), "Ce n'est pas un DataFrame !"
    # Est-ce qu'il y a des données ?
    assert not df.empty, "Le DataFrame est vide (Aucun événement trouvé ?)"
    # On vérifie que TOUT le monde est bien à Paris
    assert (df['location_city'] == "Paris").all(), "Alerte : On a récupéré des événements hors de Paris !"
    # Vérification des colonnes vitales
    colonnes_attendues = ["title_fr", "description_fr", "lastdate_end"]
    for col in colonnes_attendues:
        assert col in df.columns, f"Il manque la colonne : {col}"