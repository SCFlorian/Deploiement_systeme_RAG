# ===================================================
# EVALUATION DE NOS QUESTIONS POSÉES À NOTRE SYSTÈME
# ===================================================
# ======================
# Librairies nécessaires
# ======================
import csv
import logging
import sys, os
import time
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
# Identification du chemin pour les informations de generate_LLM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.connexion_llm import generate_LLM
from dotenv import load_dotenv

# ==================================
# On charge notre environnement env
# ==================================
load_dotenv()
# On recharge les éléments stockés dans env
SEUIL_RAG = float(os.getenv("SEUIL_RAG"))
k = int(os.getenv("k"))
INDEX_NAME = os.getenv("INDEX_NAME")
CSV_FILE = os.getenv("CSV_FILE")
# ==========================
# Initialisation du logging
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
# ===========================
# Les 20 questions à évaluer
# ===========================
questions_reponses = [
    # --- Groupe A : Questions factuelles (Réponse précise dans 1 document) ---
    {"b":"Question 1 :","q": "Quel est le prix des places pour le spectacle 'Le Balcon : Les Lundis Musicaux' ?", "a": "Le prix est compris entre 12 et 30€."},
    {"b":"Question 2 :","q": "Qui assure la direction artistique de la pièce 'La Mouche' ?","a": "C'est la Compagnie Point Fixe, sous la direction de Valérie Lesort et Christian Hecq."},
    {"b":"Question 3 :","q": "Quelles sont les dates de représentation de 'Kohlhaas' ?","a": "Du 24 mars 2026 au 1er avril 2026."},
    {"b":"Question 4 :","q": "De quoi parle l'exposition 'Soulages, une autre lumière' ?","a": "Elle présente l'œuvre sur papier de Pierre Soulages."},
    {"b":"Question 5 :","q": "Y a-t-il un événement de recrutement pour la SNCF ?","a": "Oui, il y a une présentation du métier de Conducteur de Trains le 9 février 2026."},
    {"b":"Question 6 :","q": "Quel compositeur est joué par Les Talens Lyriques dans 'Ascanio in Alba' ?","a": "C'est une œuvre de Mozart."},
    {"b":"Question 7 :","q": "Est-ce que Naïssam Jalal va lancer quelque chose prochainement ?", "a": "Naïssam Jalal va créer un laboratoire d’expérimentation autour de son répertoire 'Landscapes of Eternity', dont l’album paraîtra en avril 2026.."},
    # --- Groupe B : Recommendation et Synthèse (Réponse basée sur plusieurs docs) ---
    {"b":"Question 8 :","q": "Quels sont les spectacles du cycle Court-Circuit ?", "a": "Le cycle propose l'opéra 'L'homme qui aimait les chiens' et le spectacle 'Yokai Matsuri'."},
    {"b":"Question 9 :","q": "Quelles pièces de théâtre de Molière sont disponibles ?", "a": "Il y a 3 pièces de Molière : 'Les Femmes Savantes', 'Les Fourberies de Scapin' et 'Le Misanthrope'."},
    {"b":"Question 10 :","q": "Quels sont les trois spectacles dont le titre contient le mot 'Danse' ?", "a": "Les spectacles sont : Ensemble 2E2M : Danses en-jeu(x), Danser (l'exposition à la Cité des sciences) et L'École de Danse (de Carlo Goldoni)."},
    {"b":"Question 11 :","q": "Quels sont les spectacles programmés au Théâtre des Bouffes du Nord ?","a": "Il y en a trois : Lettres non-écrites, La Mouche et Karaoké."},
    {"b":"Question 12 :","q": "Quelles pièces sont jouées à la Comédie-Française ?","a": "Les pièces sont : Les Fourberies de Scapin (Molière), Une Mouette (Tchekhov), L'École de Danse (Goldoni) et Le Misanthrope (Molière)."},
    # --- Groupe C : Hors corpus (Doit répondre qu'il ne sait pas / Hallucination Check) ---
    {"b":"Question 13 :","q": "Y a-t-il un concert de Beyoncé prévu ?","a": "Je ne trouve aucune information sur un concert de Beyoncé dans les événements."},
    {"b":"Question 14 :","q": "Quel est le programme du Festival d'Avignon 2026 ?","a": "Je ne dispose que des événements culturels à Paris (ou dans la base fournie), pas ceux d'Avignon."},
    {"b":"Question 15 :","q": "Quels sont les horaires d'ouverture de la Tour Eiffel ?","a": "Je n'ai pas d'information sur les horaires touristiques de la Tour Eiffel, seulement sur des événements culturels ponctuels."},
    {"b":"Question 16 :","q": "Y a-t-il un événement le 14 juillet 2030 ?","a": "Je n'ai pas d'événements enregistrés pour cette date (ma base s'arrête avant)."},
    {"b":"Question 17 :","q": "Qui a gagné le match PSG-OM hier ?","a": "Je suis un assistant culturel, je n'ai pas d'informations sur les résultats sportifs récents."},
    # --- Groupe D : Ambiguës (Doit demander précision ou proposer large) ---
    {"b":"Question 18 :","q": "Je veux sortir.","a": "Quel type de sortie cherchez-vous ? (Théâtre, Concert, Exposition...)"},
    {"b":"Question 19 :","q": "C'est combien ?","a": "Le prix dépend de l'événement. De quel spectacle parlez-vous ?"},
    {"b":"Question 20 :","q": "Est-ce que c'est bien ?","a": "Je peux vous donner la description des événements, mais je n'ai pas d'avis subjectif."}
    ]
# Définition d'une fonction main pour générer l'évaluation
def main():
    logging.info("--- Chargement du RAG ---")
    
    # On a besoin des embeddings pour calculer les scores
    embeddings_model = MistralAIEmbeddings(api_key=os.getenv("MISTRAL_KEY"))
    
    # On charge l'index FAISS "à la main" juste pour récupérer les scores de distance (pour le seuil)
    vectorstore_raw = FAISS.load_local(
        INDEX_NAME, 
        embeddings_model, 
        allow_dangerous_deserialization=True
    )

    # On charge le RAG complet pour avoir la réponse du LLM
    # Récupération de notre fonction LLM
    rag_chain = generate_LLM(INDEX_NAME)
    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Voici les colonnes à apparaître dans le CSV
        if write_header:
            writer.writerow([
                "index",
                "Numéro question",
                "Question", 
                "Réponse Attendue", 
                "Réponse LLM", 
                "Score Qualité (LLM)", 
                "Documents & Scores Distance (Pour le Seuil)",
                "Context tiré des métadonnées",
                "Seuil utilisé min",
                "Nombre de documents utilisés max",
                "Eval Humaine"
            ])

        logging.info(f"--- Démarrage de l'évaluation ---")
        # On fait une boucle pour récupérer chaque question
        for item in questions_reponses:
            index = INDEX_NAME
            b = item["b"]
            q = item["q"]
            attendu = item["a"]
            print(f"Traitement : {q}")

            # Utilisation de simility_search pour récupérer les documents de FAISS ainsi que le score de pertinence et nos métadonnées
            # Utilisation de relevance_scores afin d'avoir un score normalisée pour être compatible 
            # avec le score de similarity_score_threshold dans la connexion LLM
            raw_results = vectorstore_raw.similarity_search_with_relevance_scores(q, k=k)

            # On garde seulement si le score est plus grand que le seuil
            docs_et_scores = [(d, s) for d, s in raw_results if s >= SEUIL_RAG]
            import json
            # Formatage pour le CSV
            if not docs_et_scores:
                sources_str = "Aucun document (Filtré par le seuil)"
                place_str = ""
                context_str = ""
            else:
                # 's' est maintenant un score de pertinence (ex: 0.74), pas une distance
                sources_str = " | ".join([f"Doc {d.metadata.get('doc_id')} (Score: {round(s, 3)})" for d, s in docs_et_scores])
                context_str = json.dumps([{
                    "doc_id": d.metadata.get("doc_id"),
                    "contenu": d.page_content,
                    "Date de début": d.metadata.get("Date de début"),
                    "Date de fin": d.metadata.get("Date de fin"),
                    "lieu": d.metadata.get("Lieu"),
                    "Adresse postale": d.metadata.get("Adresse postale")} for d, s in docs_et_scores],
                    ensure_ascii=False)
                
            # Interroger le LLM
            response = rag_chain.invoke({"input": q})
            reponse_llm = response["answer"]
            
            # Calculer la qualité pour l'évaluation avec cosine_similarity
            vec_attendu = embeddings_model.embed_query(attendu)
            vec_llm = embeddings_model.embed_query(reponse_llm)
            score_qualite = cosine_similarity([vec_attendu], [vec_llm])[0][0]

            # Sauvegarder tout dans le même fichier
            writer.writerow([
                index,
                b,
                q,
                attendu, 
                reponse_llm, 
                round(score_qualite, 4), 
                sources_str,
                context_str,
                SEUIL_RAG,
                k,
                ""
            ])
            print(f"   -> Pause de 3 secondes pour l'API Mistral...")
            time.sleep(3)  
    logging.info(f"\nTerminé ! Ouvre '{CSV_FILE}'.")
    logging.info("Colonne 'Documents & Scores' pour choisir le seuil.")
    logging.info("Colonne 'Eval Humaine' pour mettre les notes.")
 

if __name__ == "__main__":
    main()