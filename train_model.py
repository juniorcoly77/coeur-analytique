"""
Entraîne le modèle final (régression logistique) sur le dataset NHANES, l'évalue sur un
jeu de test indépendant, puis sauvegarde les artefacts utilisés par la fonction Vercel :
    heart_model/model_pipeline.joblib   pipeline complet (prétraitement + modèle)
    heart_model/explication.json        noms des variables + moyennes (explications locales)
    heart_model/metrics.json            performances mesurées sur le jeu de test

Usage : python train_model.py      (nécessite scikit-learn, pandas, numpy, joblib)
"""
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from heart_model.pipeline import RANDOM_STATE, build_preprocessing_steps, charger_donnees

CHEMIN_DONNEES = "data/nhanes_risque_cardiaque.csv"


def construire(C=1.0):
    return Pipeline([
        ("preparation_transformation", Pipeline(build_preprocessing_steps(use_pca=False))),
        ("modele", LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE)),
    ])


def main():
    X, y = charger_donnees(CHEMIN_DONNEES)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Recherche de l'hyperparamètre C (force de la régularisation L2), optimisée sur le ROC-AUC
    grille = {"modele__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    gs = GridSearchCV(construire(), grille, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    print("Meilleurs paramètres :", gs.best_params_, "| ROC-AUC CV :", round(gs.best_score_, 4))

    # Évaluation sur le jeu de test (jamais vu pendant la recherche)
    y_pred = gs.predict(X_te)
    y_proba = gs.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    metrics = {
        "modele": "Régression logistique", "n_total": int(len(X)), "n_test": int(len(X_te)),
        "accuracy": accuracy_score(y_te, y_pred), "precision": precision_score(y_te, y_pred),
        "rappel": recall_score(y_te, y_pred), "f1": f1_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_proba), "pr_auc": average_precision_score(y_te, y_proba),
        "brier": brier_score_loss(y_te, y_proba),
        "matrice": {"vrais_negatifs": int(tn), "faux_positifs": int(fp),
                    "faux_negatifs": int(fn), "vrais_positifs": int(tp)},
        "hyperparametres": {"C": gs.best_params_["modele__C"], "penalisation": "L2"},
    }
    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    print(classification_report(y_te, y_pred, target_names=["Risque faible (Non)", "Risque (Oui)"]))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Ré-entraînement final sur 100 % des données avec les hyperparamètres retenus
    final = construire(C=gs.best_params_["modele__C"])
    final.fit(X, y)
    joblib.dump(final, "heart_model/model_pipeline.joblib")

    # Moyennes des variables transformées (référence pour expliquer chaque prédiction)
    prepa = final.named_steps["preparation_transformation"]
    Xt = prepa.transform(X)
    # Plages observées à l'entraînement : sert à signaler les saisies où le modèle extrapole
    cols = ["age", "ta_systolique", "ta_diastolique", "ldl", "glycemie", "poids", "taille",
            "acide_urique", "freq_cardiaque"]
    plages = {c: [float(X[c].min()), float(X[c].max())] for c in cols}
    with open("heart_model/explication.json", "w") as f:
        json.dump({"moyennes": Xt.mean(axis=0).tolist(), "plages": plages}, f)
    with open("heart_model/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Artefacts sauvegardés dans heart_model/")


if __name__ == "__main__":
    main()
