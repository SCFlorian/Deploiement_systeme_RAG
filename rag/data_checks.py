# ==============================================
# FONCTION POUR CHECKER LA BONNE IMPLEMENTATION
# ==============================================

def check_city_filter(df, city):
    """
    Vérifie que tous les événements sont bien dans la ville attendue.
    """
    assert (df["location_city"] == city).all()
    
    print(f"Tous les événements sont bien à {city}")