# Mise en place d'un système RAG

## Description

La mission est la mise en place d'un système RAG afin de créer un chatbot intelligent capable de répondre à des questions utilisateurs sur les événements culturels à venir.

## Objectifs

Les objectifs à remplir sont les suivants :
- un système RAG fonctionnel
- une API REST
- un rapport technique
- un jeu de test annoté
- des tests unitaires

## Données utilisées

Nous allons nous connecter à l'API d'Open Agenda afin de sélectionner une localisation, les données d'événements et une période récente (moins d'un an).

## Organisation du projet (en cours)
```
├── .env        # pas versionné
├── .gitignore
├── poetry.lock # pas versionné
├── pyproject.toml
├── README.md
├── test.ipynb  # pas versionné
```

## Installation et utilisation
1. Cloner le projet :
```
git clone git@github.com:SCFlorian/Deploiement_systeme_RAG.git
cd Deploiement_systeme_RAG
```
2. Installer les dépendances : Le projet utilise pyproject.toml pour la gestion des dépendances :
```
poetry install --no-root
```
3. Ouvrir le projet dans VS Code :
```
code .
```
4. Configurer l’environnement Python dans VS Code
	1.	Installez l’extension Python (si ce n’est pas déjà fait).
	2.	Appuyez sur Ctrl+Shift+P (Windows/Linux) ou Cmd+Shift+P (Mac).
	4.	Recherchez “Python: Select Interpreter”.
	5.	Sélectionnez l’environnement créé par Poetry ou celui dans lequel tu as installé le projet.

5. Vérifier les imports principaux dans un notebook ou un script :
```
# Imports standards
import faiss

# Imports LangChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # ou SentenceTransformerEmbeddings

# Import Mistral
from mistralai.client import MistralClient
```
