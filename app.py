# =============================
# LANCEMENT DE NOTRE API LOCAL
# =============================
# Librairies nécessaires
# =======================
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import os
from pydantic import BaseModel
import uvicorn
# Imports nécessaires
# ====================
from scripts.chat import answer_chat
from scripts.build_index import data_update

# ========================
# Activation de notre API
# ========================
app = FastAPI(title="Puls-Events - POC Chatbot intelligent")

# Modèle d'entrée API
# =========================
class Question(BaseModel):
    question: str

# Gestion de l'erreur 404 (Route inexistante)
# ============================================
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error_code": 404,
            "message": "Cette route n'existe pas.",
            "detail": f"L'URL '{request.url.path}' est inconnue.",
            "suggestion": "Essayez plutôt /ask ou /rebuild ou encore /health"
        }
    )
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
        question = "Je cherche une pièce de théâtre en 2026"
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