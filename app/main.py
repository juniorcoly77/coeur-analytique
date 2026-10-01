"""
API FastAPI — sert à la fois le frontend statique (public/) et l'endpoint de
prédiction /api/predict. Conçue pour être conteneurisée avec Docker.
"""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Permet d'importer le package heart_model/ situé à la racine du projet
RACINE_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE_PROJET))

from heart_model.pipeline import COLONNES_BRUTES, flags_lisibles  # noqa: E402

CHEMIN_MODELE = RACINE_PROJET / "heart_model" / "model_pipeline.joblib"

app = FastAPI(title="Coeur Analytique — API de classification")

# CORS ouvert : l'API peut être appelée depuis un frontend hébergé ailleurs
# (utile si le frontend Vercel et l'API Docker sont déployés séparément)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Le modèle est chargé UNE SEULE FOIS au démarrage du conteneur, pas à chaque requête
modele = joblib.load(CHEMIN_MODELE)


class Patient(BaseModel):
    """Schéma de validation des 13 variables cliniques brutes attendues par le modèle."""

    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=60, le=250)
    chol: float = Field(..., ge=80, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=60, le=230)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=3)
    thal: float = Field(..., ge=3, le=7)


class Prediction(BaseModel):
    prediction: int
    probabilite: float
    indicateurs: list[str]


@app.post("/api/predict", response_model=Prediction)
def predire(patient: Patient):
    """Reçoit les données brutes d'un patient et renvoie la prédiction du modèle."""
    try:
        donnees = pd.DataFrame([patient.model_dump()])[COLONNES_BRUTES]
        proba = float(modele.predict_proba(donnees)[0, 1])
        prediction = int(modele.predict(donnees)[0])
    except Exception as exc:  # garde-fou : ne jamais laisser fuiter une trace serveur brute
        raise HTTPException(status_code=400, detail="Données invalides pour la prédiction.") from exc

    return Prediction(
        prediction=prediction,
        probabilite=proba,
        indicateurs=flags_lisibles(patient.model_dump()),
    )


@app.get("/api/health")
def sante():
    """Vérification de bon fonctionnement du service (utile pour Docker/monitoring)."""
    return {"status": "ok"}


# Sert le frontend statique (index.html, style.css, script.js) sur "/"
app.mount("/", StaticFiles(directory=str(RACINE_PROJET / "public"), html=True), name="static")
