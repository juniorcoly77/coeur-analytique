"""
Pipeline complet d'ENTRAÎNEMENT (prétraitement + SMOTE + Random Forest).

Ce module dépend de `imbalanced-learn`, contrairement à heart_model/pipeline.py.
Il n'est importé que par train_model.py — jamais par l'API de production (Docker)
ni par la fonction serverless Vercel — afin que ces déploiements n'aient pas besoin
d'installer `imbalanced-learn`.
"""

from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from heart_model.pipeline import build_preprocessing_steps, RANDOM_STATE


def build_pipeline():
    """Construit le pipeline complet d'entraînement, non entraîné (prêt pour .fit(X, y))."""
    etapes = build_preprocessing_steps()

    etape_smote = SMOTE(random_state=RANDOM_STATE)

    # Hyperparamètres issus du GridSearchCV du notebook de modélisation
    etape_modele = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=1, random_state=RANDOM_STATE
    )

    etapes += [
        ("smote", etape_smote),
        ("modele", etape_modele),
    ]

    return ImbPipeline(steps=etapes)
