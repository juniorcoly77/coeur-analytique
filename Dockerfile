# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

# Empêche Python d'écrire des .pyc et force les logs non bufferisés (utile avec `docker logs`)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dépendances installées avant de copier le code source : le cache Docker n'est invalidé
# que si requirements.txt change, pas à chaque modification du code applicatif.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source, module de prétraitement/modèle, et frontend statique
COPY app/ ./app/
COPY heart_model/ ./heart_model/
COPY public/ ./public/

# Utilisateur non-root (bonne pratique de sécurité pour un conteneur exposé)
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
