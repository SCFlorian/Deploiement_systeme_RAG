# Image de base
FROM python:3.13-slim

# Installation des outils système nécessaires
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installation de Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Ajout de Poetry au PATH
ENV PATH="/root/.local/bin:$PATH"

# Configuration : pas de virtualenv dans le Docker (inutile car le conteneur est déjà isolé)
RUN poetry config virtualenvs.create false

# Dossier de travail
WORKDIR /app

# On copie uniquemet les fichiers de définition de dépendances
COPY pyproject.toml poetry.lock ./

# Installation des dépendances
RUN poetry install --no-interaction --no-ansi --no-root

# Copie du reste du code source
COPY . .

# Port exposé
EXPOSE 7860

# Lancement de l'API
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]