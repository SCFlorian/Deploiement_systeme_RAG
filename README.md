# Assistant IA Culturel - Puls-Events (POC RAG)

## Description du projet

Ce projet est une **Preuve de Concept (POC)** développée pour Puls-Events. Il s'agit d'un chatbot intelligent capable de recommander des événements culturels à Paris (concerts, théâtres, expositions) en se basant sur des données fraîches via l'API OpenAgenda.

Le système utilise une architecture RAG (Retrieval-Augmented Generation) pour garantir que les réponses sont factuelles et basées sur les documents fournis, évitant ainsi les hallucinations des modèles de langage classiques.

## Stack technique

- Langage : Python 3.13
- API : FastAPI
- LLM & Embeddings : Mistral AI (mistral-small-latest / mistral-embed)
- Base Vectorielle : FAISS (Facebook AI Similarity Search)
- Orchestration : LangChain
- Conteneurisation : Docker
- Gestion de dépendances : Poetry

## Données utilisées

Nous allons nous connecter à l'API d'Open Agenda afin de sélectionner une localisation, les données d'événements et une période récente.
L'actualisation des données pourra se faire en local ou via un bouton "/rebuild" sur l'API. Lors de la reconstruction les données seront actualisées de cette manière :
```
 “date du jour - 6 mois” à “date du jour + 6 mois”
 ```


## Architecture du projet
```
├── data/                                   # Données des évaluations
│    ├── evaluation_ragas.csv               # Résultats de l'évaluation des questions avec ragas
│    ├── resultat_evaluation.csv            # Résultats de l'évaluation des questions "manuelles"
├── evaluation/                             # Scripts pour les évaluations
│    ├── ragas_evaluation.py                # Script réutilisable de l'évaluation ragas
│    ├── run_evaluation.py                  # Script réutilisable de l'évaluation "manuelle"
├── faiss_index_long/                       # Base de données vectorielles version longue - pas versionné sur Git
├── faiss_index_short/                      # Base de données vectorielles version courte - pas versionné sur Git
├── notebook/                               # Notebooks d'analyse
│    ├── notebook_evaluation_ragas.ipynb    # Notebook afin de faciliter les premières étapes d'observation et de nettoyage
│    ├── notebook_preprocessing.ipynb       # Notebook afin d'interpréter plus facilement les résultats des évaluations
├── rag/                                    # Scripts logique RAG
│   ├── connexion_llm.py                    # Prompt template + chaîne Langchain LLM
│   ├── data_checks.py                      # Check rapide de la localisation
│   ├── data_loader.py                      # Chargement des données OpenAgenda
│   ├── embeddings.py                       # Génération des documents et des embeddings
│   ├── preprocessing.py                    # Sélection des bonnes colonnes avant génération des embeddings
│   ├── similarity_test.py                  # Test de similarité
│   ├── vectorstore.py                      # Gestion et sauvegarde de l'index FAISS 
├── scripts/                                # Outils
│   ├── build_index.py                      # Script de nettoyage et de construction de l'index
│   └── chat.py                             # Pipeline de réponse (Prompt template + LLM)
├── tests/                                  # Tests unitaires et focntionnels
│   └── test_data_loader.py                 # Tests validant le chargement des données (validation de ce test dans le chargement de l'image Docker)
│   └── test_indexation_retrieval.py        # Tests validant l'indexation et le retrieval
│   └── test_preprocessing.py               # Tests validant le preprocessing (validation de ce test dans le chargement de l'image Docker)
├── .dockerignore                           # Permet de ne pas afficher notre env lors de la construction de l'image
├── .env                                    # Permet de gérer les informations sensibles - pas versionné sur Git
├── .env.sample                             # Exemple de variables d'environnement (Non versionné)
├── .gitignore                              # Permet de ne pas afficher les éléments sélectionnés sur GitHub
├── app.py                                  # Point d'entrée de l'API (FastAPI)
├── Dockerfile                              # Configuration de l'image Docker (avec tests intégrés à la création de l'image)
├── poetry.lock                             # Pas versionné sur Git
├── pyproject.toml                          # Gestion des dépendances Poetry
├── README.md                               # Documentation du projet
├── rapport_technique.pdf                   # Rapport technique du projet
```

## Installation et utilisation
### Avant un lancement
1. Créez un fichier .env à la racine du projet et ajoutez votre clé API (sans guillemets) :
	1. Vous pouvez remettre les paramètres de l'exemple mais surtout il faut ajouter **votre clé API Mistral** :
```
MISTRAL_KEY=***
SEUIL_RAG=0.65
k=3
```
2. Exécutez le script **scripts/build_index.py** afin de construire vos bases FAISS


### Installation sans Docker
1. Clonez le projet :
```
git clone git@github.com:SCFlorian/Deploiement_systeme_RAG.git
cd Deploiement_systeme_RAG
```
2. Installez les dépendances : Le projet utilise pyproject.toml pour la gestion des dépendances :
```
poetry install --no-root
```
3. Ouvrir le projet dans VS Code :
```
code .
```
4. Configurez l’environnement Python dans VS Code
	1.	Installez l’extension Python (si ce n’est pas déjà fait).
	2.	Appuyez sur Ctrl+Shift+P (Windows/Linux) ou Cmd+Shift+P (Mac).
	4.	Recherchez “Python: Select Interpreter”.
	5.	Sélectionnez l’environnement créé par Poetry ou celui dans lequel tu as installé le projet.

5. Exécutez le script **app.py**

### Installation avec Docker
La méthode privilégiée pour exécuter ce projet est **Docker**, garantissant un environnement isolé et stable.
1. Prérequis
- Docker installé sur votre machine.
- Une clé API Mistral AI active.
2. Construction et Test (Build)
- Lors de la construction de l'image, les tests unitaires sont exécutés automatiquement. Si les tests critiques échouent, la construction s'arrête. Cela garantit la qualité du livrable.
```
docker build --progress=plain -t chatbot-rag .
```
- L'option --progress=plain permet de voir les résultats des tests et le rapport de couverture (Coverage) dans le terminal.
3. Lancement (Run)
- Démarrez le conteneur en lui passant les variables d'environnement :
```
docker run -d --name mon-chatbot-rag -p 7860:7860 --env-file .env chatbot-rag
```
### Utilisation de l'API
- Dans les 2 cas, l'API est accessible localement sur le port 7860.
- **Documentation Interactive (Swagger UI)**, rendez-vous sur :
```
http://localhost:7860/docs
```
- Endpoints Principaux :
	- **/health**	Vérifie que l'API tourne correctement.
	- **/rebuild**	Télécharge les données et construit l'index vectoriel. À lancer au premier démarrage.
	- **/ask**	Pose une question au chatbot.

## Qualité et Tests
La stratégie de test adoptée pour ce POC repose sur deux piliers :
- Tests unitaires au build :
	- Les modules critiques (preprocessing, data_loader) sont testés pendant la création de l'image Docker.
	- Si le code de nettoyage des données est défectueux, l'image ne se construit pas.

- Couverture actuelle sur les modules critiques : 100%.

- Tests d'Intégration (Local) :
	- Les tests nécessitant une connexion à l'API Mistral (test_indexation_retrieval.py) sont exécutés localement avant le déploiement pour valider la chaîne complète.

## Métriques et Évaluation
Réalisation de deux types d'évaluation :
- La première a été de générer un score de pertinence afin de déterminer le bon seuil entre nos données et nous permettre d'évaluer de manière "manuelle" si les réponses sont pertinentes ou non.
- La deuxième, ragas, est un framework qui permet d'évaluer les performances d'un pipeline RAG à travers plusieurs métriques:
	- **faithfulness**, est-ce que la génération est fidèle au contexte ?
    - **answer_relevancy**, est-ce que la génération de la réponse est pertinente à la question ?
    - **context_precision**, est-ce que la récupération du contexte est précise (peu de bruit) ?
    - **context_recall**, est-ce que la récupération des infos clés sont correctements récupérées ?

- Résultats : En combinant les 2 évaluations (voir notebook_evaluation_ragas.ipynb), on voit que les meilleurs paramètres sont :
	- k=3 et seuil à 0.65
	- k=3 et seuil à 0.50
- On retient pour le moment la deuxième proposition car les résultats sont meilleurs lorsque l'on prend uniquement des questions factuelles.
- Et nous prenons la base de FAISS version courte (version sans chunk) car cela va nous permettre de générer moins de token dans un premier temps et les réponses sont pour le moment en adéquation avec ce que l'on recherche pour ce POC.

