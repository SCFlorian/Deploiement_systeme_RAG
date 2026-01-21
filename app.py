# =============================
# LANCEMENT DE NOTRE API LOCAL
# =============================
# Librairies nécessaires
# =======================
from fastapi import FastAPI
import os
from pydantic import BaseModel
import uvicorn
# Imports nécessaires
# ====================
from chat import answer_chat
from build_index import data_update

# ========================
# Activation de notre API
# ========================
app = FastAPI(title="Puls-Events - POC Chatbot intelligent")

# Modèle d'entrée API
# =========================
class Question(BaseModel):
    question: str

# Routes simples
# =========================
@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API opérationnelle"}

# ================================================
# Fonction de génération de la réponse du chatbot
# ================================================
@app.post("/ask")
def speak_to_chatbot(data: Question):
    try:
        question = "Y a-t-il un événement de recrutement pour la SNCF ? N'hésite pas à développer"
        answer = answer_chat(data.question)
        return {
            "answer": answer
        }
    except Exception as e:
        return {
            "status": "Erreur",
            "message": str(e)
        }
# ==========================================================================
# Fonction afin de prendre des nouvelles données depuis l'API d'Open Agenda
# ==========================================================================
@app.post("/rebuild")
def rebuild_data():
    """
    Déclenche la mise à jour de la base vectorielle.
    
    Le script sous-jacent (build_index.py) va automatiquement :
    1. Calculer la date du jour.
    2. Récupérer 50% d'événements passés (-6 mois) et 50% futurs (+6 mois).
    3. Reconstruire l'index FAISS.
    """
    try:
        # On lance la mise à jour sans arguments.
        # C'est build_index qui gère l'intelligence des dates.
        data_update()
        
        return {
            "status": "Succès",
            "message": "Base de données mise à jour (Période : J-6 mois à J+6 mois)"
        }
    except Exception as e:
        return {
            "status": "Erreur",
            "message": str(e)
        }
# =========================
# Lancement local
# =========================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)