"""
Fonction serverless Vercel — POST /api/predict

Reçoit les 16 variables saisies dans le formulaire, applique le pipeline entraîné
(prétraitement + régression logistique) et renvoie la probabilité de risque de crise
cardiaque, les facteurs qui pèsent le plus dans le calcul et les facteurs de risque détectés.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from heart_model.pipeline import (  # noqa: E402
    COLONNES_BRUTES, TYPES_DOULEUR, TYPES_ECG, facteurs_contributifs, flags_lisibles,
)

modele = joblib.load(RACINE / "heart_model" / "model_pipeline.joblib")
with open(RACINE / "heart_model" / "explication.json", encoding="utf-8") as f:
    EXPLICATION = json.load(f)
with open(RACINE / "heart_model" / "metrics.json", encoding="utf-8") as f:
    METRIQUES = json.load(f)

app = Flask(__name__)

# Bornes de validation (plausibilité physiologique) pour les champs numériques
BORNES = {
    "age": (18, 110), "ta_systolique": (70, 260), "ta_diastolique": (40, 160),
    "ldl": (20, 400), "glycemie": (30, 600), "poids": (25, 300), "taille": (100, 230),
    "acide_urique": (1, 15), "freq_cardiaque": (30, 220),
}
CHOIX = {
    "sexe": ["Homme", "Femme"],
    "type_douleur": TYPES_DOULEUR,
    "ecg_repos": TYPES_ECG,
    "tabagisme": ["Oui", "Non"],
    "sedentarite": ["Active", "Sédentaire"],
    "angine_effort": ["Oui", "Non"],
    "sous_decalage_st": ["Oui", "Non"],
}
LIBELLES = {
    "age": "Âge", "ta_systolique": "Pression systolique", "ta_diastolique": "Pression diastolique",
    "ldl": "LDL", "glycemie": "Glycémie", "poids": "Poids", "taille": "Taille",
    "acide_urique": "Acide urique", "freq_cardiaque": "Fréquence cardiaque",
}


def valider(payload):
    patient = {}
    for champ, (mini, maxi) in BORNES.items():
        if champ not in payload:
            raise ValueError(f"Champ manquant : {champ}")
        try:
            valeur = float(payload[champ])
        except (TypeError, ValueError):
            raise ValueError(f"Valeur invalide pour « {LIBELLES[champ]} ».")
        if not (mini <= valeur <= maxi):
            raise ValueError(f"« {LIBELLES[champ]} » doit être compris entre {mini} et {maxi}.")
        patient[champ] = valeur
    for champ, options in CHOIX.items():
        valeur = str(payload.get(champ, "")).strip()
        if valeur not in options:
            raise ValueError(f"Choix invalide pour « {champ} ».")
        patient[champ] = valeur
    if patient["ta_diastolique"] >= patient["ta_systolique"]:
        raise ValueError("La pression diastolique doit être inférieure à la systolique.")
    return patient


def niveau(p):
    if p < 0.35:
        return "faible"
    if p < 0.65:
        return "intermediaire"
    return "eleve"


@app.route("/api/predict", methods=["POST"])
def predire():
    try:
        patient = valider(request.get_json(force=True, silent=True) or {})
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400

    X = pd.DataFrame([patient])[COLONNES_BRUTES]
    proba = float(modele.predict_proba(X)[0, 1])

    hors_plage = [
        f"{LIBELLES[c]} ({patient[c]:g}) est hors de la plage vue à l'entraînement "
        f"({b[0]:g}–{b[1]:g}) : la prédiction est moins fiable."
        for c, b in EXPLICATION["plages"].items()
        if not (b[0] <= patient[c] <= b[1])
    ]

    return jsonify({
        "prediction": int(proba >= 0.5),
        "probabilite": proba,
        "niveau": niveau(proba),
        "facteurs": facteurs_contributifs(modele, EXPLICATION["moyennes"], X),
        "indicateurs": flags_lisibles(patient),
        "hors_plage": hors_plage,
        "modele": {
            "nom": METRIQUES["modele"], "roc_auc": METRIQUES["roc_auc"],
            "accuracy": METRIQUES["accuracy"], "n_total": METRIQUES["n_total"],
        },
    })


@app.route("/api/predict", methods=["GET"])
def methode_non_autorisee():
    return jsonify({"detail": "Utilisez POST avec un JSON patient."}), 405
