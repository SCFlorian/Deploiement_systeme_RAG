# FONCTIONS AFIN DE GÉNÉRER LE PREPROCESSING

# ========================================================================
# Fonction pour supprimer les colonnes identifiées d'un point de vu métier
# ========================================================================
def del_col_business_side(df):
    """Colonnes identifiées à la main afin d'enlever les variables non pertinentes"""

    col_business_side = [
        "image", "imagecredits","thumbnail","originalimage","accessibility", "location_uid",
        "location_coordinates", "location_city","location_insee","location_department",
        "location_region","location_countrycode","location_image","location_imagecredits","location_tags",
        "attendancemode", "onlineaccesslink","country_fr","registration","links","originagenda_uid"
        ]
    # Supression des colonnes
    df_test_clean = df.drop(columns=col_business_side)
    print("Colonnes bien supprimer")

    return df_test_clean

# ============================================================================
# Fonction pour supprimer les colonnes avec plus de 70% de données manquantes
# ============================================================================

def del_col_missed_val(df_test_clean):
    """Supprimer les colonnes avec plus de 70% de données manquantes mais en gardant
    la variable age min qui peut avoir une importance par la suite"""
    df_missed_values = df_test_clean.loc[
        :,
        (df_test_clean.isnull().mean() >= 0.70)
        & (df_test_clean.columns != "age_min")]
    df_col_missed = df_missed_values.columns
    # Supression des colonnes
    df_firstclean = df_test_clean.drop(columns=df_col_missed)
    
    return(df_firstclean)
    
# ===================================================================
# Préparation du document afin de passer à la vectorisation
# Choix de faire deux versions afin de procéder de manière itérative
# ===================================================================

def col_rename(df_firstclean):
    """Simplification des noms de plusieurs variables qui semblent importantes.
    Et création de deux versions pour appliquer un modèle d'embedding"""
    df_firstclean_new = df_firstclean.rename(columns={
    "title_fr":"title",
    "description_fr":"description",
    "conditions_fr":"conditions",
    "longdescription_fr":"longdescription",
    "conditions_fr":"conditions",
    "keywords_fr":"keywords",
    "daterange_fr":"daterange_fr"
    })
    # Création de nos deux dataframe
    df_short = df_firstclean_new.copy()
    df_long = df_firstclean_new.copy()

    return df_firstclean_new, df_short, df_long

# =========================================================
# Nettoyage des colonnes sélectionnées pour les embeddings
# =========================================================
# Version courte
def short_version(df_short):
    """Préparation d'une colonne pour la version courte"""
    df_short["text_for_embedding"] = (
        df_short["title"] + "\n" +
        df_short["description"].fillna("") + "\n" +
        df_short["conditions"].fillna("") + "\n")

    return df_short

# Version longue
def long_version(df_long):
    """Préparation d'une colonne pour la version longue"""
    # Ajout de cette ligne afin de gérer les données html de la longue description
    df_long["longdescription"] = df_long["longdescription"].str.replace("<[^>]+>", " ", regex=True)
    df_long["text_for_embedding"] = (
        "TITRE: " + df_long["title"] + "\n" +
        "DESCRIPTION: " + df_long["longdescription"].fillna("") + "\n" +
        "CONDITIONS: " + df_long["conditions"].fillna("") + "\n")

    return df_long
