"""
Entraîne le pipeline complet (prétraitement + PCA + SMOTE + Random Forest) sur le
dataset brut, l'évalue sur un jeu de test indépendant, puis sauvegarde l'artefact
`heart_model/model_pipeline.joblib` utilisé par l'API (Docker) et par la fonction
serverless (Vercel).

Usage : python train_model.py
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report,
)

from heart_model.pipeline import COLONNES_BRUTES, RANDOM_STATE
from heart_model.train_pipeline import build_pipeline

CHEMIN_DONNEES = "heart_disease_cleveland_ML.csv"
CHEMIN_MODELE = "heart_model/model_pipeline.joblib"


def main():
    # --- Chargement des données brutes ---
    df = pd.read_csv(CHEMIN_DONNEES)
    df.columns = [c.strip() for c in df.columns]

    X = df[COLONNES_BRUTES]
    y = df["num"]

    # --- Séparation train/test stratifiée (le test ne sert qu'à l'évaluation finale) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # --- Entraînement du pipeline complet ---
    # L'imputation, le scaler, l'encodeur, la PCA et SMOTE sont tous appris UNIQUEMENT
    # sur X_train/y_train à l'intérieur du pipeline -> aucune fuite de données.
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # --- Évaluation sur le jeu de test, jamais vu pendant l'entraînement ---
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=== Évaluation sur le jeu de test ===")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Précision : {precision_score(y_test, y_pred):.4f}")
    print(f"Rappel    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Sain (0)", "Malade (1)"]))

    # --- Ré-entraînement final sur 100% des données disponibles ---
    # Une fois la performance validée sur le test ci-dessus, on ré-entraîne le pipeline
    # sur l'ensemble du dataset pour maximiser les données utilisées par le modèle
    # réellement déployé (pratique standard une fois le test de généralisation effectué).
    pipeline_final = build_pipeline()
    pipeline_final.fit(X, y)

    # --- Extraction d'un pipeline d'INFÉRENCE sans SMOTE ---
    # SMOTE ne sert qu'à sur-échantillonner pendant l'entraînement ; il est inactif à la
    # prédiction. On le retire donc du pipeline sauvegardé pour servir un artefact plus
    # léger, qui ne dépend plus de `imbalanced-learn` au moment de la prédiction
    # (important pour respecter les limites de taille d'une fonction serverless Vercel).
    etapes_inference = [(nom, etape) for nom, etape in pipeline_final.steps if nom != "smote"]
    pipeline_inference = SkPipeline(steps=etapes_inference)

    joblib.dump(pipeline_inference, CHEMIN_MODELE)
    print(f"\nModèle final (inférence, sans SMOTE) sauvegardé -> {CHEMIN_MODELE}")


if __name__ == "__main__":
    main()
