# =======================
# Librairies nécessaires
# =======================
import requests
import pandas as pd
from datetime import datetime, timedelta 

# ==============================================================================================
# Fonction pour récupérer les données (50 passés / 50 futurs) avec dates automatiques
# ==============================================================================================

def prepare_dataframe():

    # Calcul des dates (Aujourd'hui, -6 mois, +6 mois)
    today = datetime.now()
    date_jour = today.strftime("%Y-%m-%d")
    date_debut = (today - timedelta(days=180)).strftime("%Y-%m-%d")
    date_fin = (today + timedelta(days=180)).strftime("%Y-%m-%d")

    # Chemin pour l'API
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

    # =======================================
    # PARTIE 1 : 50 événements du PASSÉ
    # =======================================
    params_past = {
        "limit": 50,
        "lang": "fr",
        "refine": ["location_city:Paris"],
        # On cherche entre il y a 6 mois et aujourd'hui
        "where": f"lastdate_end >= '{date_debut}' AND lastdate_end < '{date_jour}'",
        "order_by": "lastdate_end desc"
    }
    
    response_past = requests.get(url, params=params_past)
    response_past.raise_for_status()
    df_past = pd.DataFrame(response_past.json().get("results"))

    # =======================================
    # PARTIE 2 : 50 événements du FUTUR
    # =======================================
    params_future = {
        "limit": 50,
        "lang": "fr",
        "refine": ["location_city:Paris"],
        # On cherche entre aujourd'hui et dans 6 mois
        "where": f"lastdate_end >= '{date_jour}' AND lastdate_end <= '{date_fin}'",
        "order_by": "lastdate_end asc"
    }

    response_future = requests.get(url, params=params_future)
    response_future.raise_for_status()
    df_future = pd.DataFrame(response_future.json().get("results"))

    # =======================================
    # FUSION
    # =======================================
    # On colle les deux morceaux ensemble
    df = pd.concat([df_past, df_future], ignore_index=True)

    return df