# ===================================================
# EVALUATION RAGAS
# ===================================================
# ======================
# Librairies nécessaires
# ======================
import csv
import os
import pandas as pd
import logging
import json
from dotenv import load_dotenv
# Pour intégrer langchain/Faiss & ragas
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall)
# Pour avoir accès à Hugging Face Dataset
from datasets import Dataset

# On recharge les éléments stockés dans env
load_dotenv()
model = os.getenv("model")

# Initialisation du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

# Pour que les lignes soient visibles
pd.set_option('display.max_colwidth', None)

# ========================================
# Récupération du ficher CSV d'évaluation
# ======================================== 
df_ragas = pd.read_csv("data/resultat_evaluation.csv")
logging.info(df_ragas.head())

# Modification de la colonne 'contexts' pour qu'elle soit compréhensible pour le format dataset
df_ragas['contexts'] = df_ragas['Context tiré des métadonnées'].fillna("[]").apply(lambda x: [json.dumps(d, ensure_ascii=False) for d in json.loads(x)])
logging.info(df_ragas['Context tiré des métadonnées'][10])

# Renommage des colonnes
evaluation_rename = df_ragas.rename(columns={
    "Question": "questions_test",
    "Réponse LLM": "answers",
    "contexts": "placeholder_contexts",
    "Réponse Attendue": "ground_truths"
})
# Choix des colonnes à sélectionner dans notre df
logging.info("Préparation de notre nouveau df")
evaluation_data = {
    "question": evaluation_rename["questions_test"],
    "answer": evaluation_rename["answers"],
    "contexts": evaluation_rename["placeholder_contexts"],
    "ground_truth": evaluation_rename["ground_truths"]}

# Préparation du dataset afin de le rendre compatible avec le système d'évaluation
evaluation_dataset = Dataset.from_dict(evaluation_data)
logging.info("Dataset d'évaluation prêt.")

# Initialisation du même modèle de language utilisé pour le chatbot
llm = ChatMistralAI(
        model=model,
        temperature=0.2
    )
# Initialisation du même modèle d'embeddings que pour la préparation de nos vecteurs
embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_KEY"))

# Définition des métriques à calculer
metrics_to_evaluate = [
        faithfulness,       # Génération: fidèle au contexte ?
        answer_relevancy,   # Génération: réponse pertinente à la question ?
        context_precision,  # Récupération: contexte précis (peu de bruit) ?
        context_recall,     # Récupération: infos clés récupérées ?
    ]
logging.info(f"Métriques sélectionnées: {[m.name for m in metrics_to_evaluate]}")

# Lancement de l'évaluation Ragas
logging.info("\nLancement de l'évaluation Ragas")
results = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics_to_evaluate,
    llm=llm,                # LLM pour juger certaines métriques
    embeddings=embeddings)   # Embeddings pour juger d'autres métriques
logging.info("\n--- Évaluation Ragas terminée ---")

# Affichage des résultats sous forme de DataFrame
logging.info("\n--- Résultats de l'évaluation (DataFrame) ---")
results_df = results.to_pandas()
df_final = df_ragas.copy()
# On ajoute les colonnes de scores (si elles existent dans results_df)
for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
    if metric in results_df.columns:
        df_final[metric] = results_df[metric]

# Affichage pour vérifier
pd.set_option('display.max_rows', None)
logging.info("\n--- Aperçu des résultats finaux ---")
logging.info(df_final[['Question', 'faithfulness', 'answer_relevancy']].head())

# SAUVEGARDE DANS UN NOUVEAU FICHIER
output_filename = "data/new_valuation_ragas.csv" 
df_final.to_csv(output_filename, index=False)

logging.info(f"\nSuccès ! Les résultats enrichis sont sauvegardés dans : {output_filename}")

# Calcul des moyennes (juste pour info dans la console)
logging.info("\n--- Scores Moyens ---")
logging.info(df_final[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean())