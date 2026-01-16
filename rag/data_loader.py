# =======================
# Librairies nécessaires
# =======================
import requests
import pandas as pd

# ==============================================================================================
# Fonction pour récupérer les données de l'API d'Open Agenda en prenant les filtres nécessaires
# ==============================================================================================

def prepare_dataframe():

    # Chemin pour l'API d'Open Agenda sur le dataset des évènements publics
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

    params = {
        "limit": 100,
        "lang": "fr",
        # Choix de la localisation à Paris
        "refine": ["location_city:Paris"],
        "where": (
            # lastdate_end = dernière date à laquelle se termine l’évènement
            # Choix de prendre des évènements qui finissent après le 01/01/2025
            "lastdate_end >= '2025-01-01' "
            # Prendre les évènements en  compte jusqu'avril 2026
            "AND lastdate_end <= '2026-04-01'"
        ),
        # Classé par ordre décroissant afin d'avoir des données en 2025 et 2026
        "order_by": "lastdate_end desc"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data.get("results"))

    return df