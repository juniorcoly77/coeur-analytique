"""
Fonction serverless Vercel — endpoint POST /api/predict.

Réutilise EXACTEMENT le même artefact (`heart_model/model_pipeline.joblib`) et la
même logique de prétraitement que l'API Docker (app/main.py), via le package
partagé `heart_model/`. Écrite en Flask (WSGI) car c'est le format que le
runtime Python de Vercel prend nativement en charge.

Vercel route automatiquement /api/predict vers ce fichier d'après son emplacement
(routage par système de fichiers) : aucune configuration de route supplémentaire
n'est nécessaire au-delà de vercel.json (voir includeFiles pour embarquer heart_model/).
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify

RACINE_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE_PROJET))

from heart_model.pipeline import COLONNES_BRUTES, flags_lisibles  # noqa: E402

CHEMIN_MODELE = RACINE_PROJET / "heart_model" / "model_pipeline.joblib"

app = Flask(__name__)

# Chargé une seule fois par instance de fonction serverless (réutilisé entre invocations
# tant que l'instance reste "chaude" — comportement standard des fonctions Vercel).
modele = joblib.load(CHEMIN_MODELE)

CHAMPS_ATTENDUS = {
    "age": (1, 120), "sex": (0, 1), "cp": (1, 4), "trestbps": (60, 250),
    "chol": (80, 700), "fbs": (0, 1), "restecg": (0, 2), "thalach": (60, 230),
    "exang": (0, 1), "oldpeak": (0, 10), "slope": (1, 3), "ca": (0, 3), "thal": (3, 7),
}


def valider(payload: dict):
    """Validation manuelle simple (équivalent du schéma Pydantic côté FastAPI)."""
    patient = {}
    for champ, (mini, maxi) in CHAMPS_ATTENDUS.items():
        if champ not in payload:
            raise ValueError(f"Champ manquant : {champ}")
        try:
            valeur = float(payload[champ])
        except (TypeError, ValueError):
            raise ValueError(f"Champ invalide : {champ}")
        if not (mini <= valeur <= maxi):
            raise ValueError(f"Champ hors limites : {champ}")
        patient[champ] = valeur
    return patient


@app.route("/api/predict", methods=["POST"])
def predire():
    try:
        patient = valider(request.get_json(force=True, silent=True) or {})
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400

    donnees = pd.DataFrame([patient])[COLONNES_BRUTES]
    proba = float(modele.predict_proba(donnees)[0, 1])
    prediction = int(modele.predict(donnees)[0])

    return jsonify({
        "prediction": prediction,
        "probabilite": proba,
        "indicateurs": flags_lisibles(patient),
    })


@app.route("/api/predict", methods=["GET"])
def methode_non_autorisee():
    return jsonify({"detail": "Utilisez POST avec un JSON patient."}), 405
