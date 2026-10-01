"""
Briques de prétraitement pour le dataset Heart Disease (Cleveland).

IMPORTANT : ce module est importé au moment du chargement (unpickling) du modèle
par joblib, aussi bien dans l'API FastAPI (Docker) que dans la fonction serverless
Vercel. Il ne doit donc dépendre QUE des bibliothèques disponibles en PRODUCTION
(scikit-learn, pandas, numpy) — surtout pas de `imbalanced-learn`, qui n'est
nécessaire qu'à l'entraînement (voir heart_model/train_pipeline.py) et alourdirait
inutilement le déploiement (notamment les limites de taille de Vercel).

Étapes couvertes ici (identiques au notebook de preprocessing) :
    1. Imputation de 'ca' et 'thal' (valeurs manquantes -> mode appris sur le train)
    2. Enrichissement (feature engineering) + création des colonnes flags
    3. Normalisation des variables numériques + encodage One-Hot des catégorielles
    4. Réduction de dimension par PCA (13 composantes)
La suite (SMOTE + Random Forest) est assemblée dans heart_model/train_pipeline.py,
utilisé uniquement par le script d'entraînement.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.decomposition import PCA

RANDOM_STATE = 42

# Les 13 variables cliniques brutes attendues en entrée (celles saisies dans le formulaire)
COLONNES_BRUTES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

COLONNES_NUMERIQUES = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca",
    "fc_max_theorique", "reserve_cardiaque", "ratio_chol_age",
    "charge_vasculaire", "score_risque",
]
COLONNES_CATEGORIELLES = ["cp", "restecg", "slope", "thal"]
COLONNES_BINAIRES_BASE = ["sex", "fbs", "exang"]
COLONNES_FLAGS = [
    "flag_hypertension", "flag_hypercholesterolemie", "flag_glycemie_elevee",
    "flag_angine_effort", "flag_ecg_anormal", "flag_depression_st_importante",
    "flag_senior",
]
COLONNES_BINAIRES = COLONNES_BINAIRES_BASE + COLONNES_FLAGS

N_COMPOSANTES_PCA = 13


class ImputeCaThal(BaseEstimator, TransformerMixin):
    """
    Convertit 'ca' et 'thal' en numérique (les valeurs non convertibles -> NaN),
    puis impute les valeurs manquantes par le MODE appris sur le jeu d'entraînement
    (même logique que le notebook de preprocessing).
    """

    def fit(self, X, y=None):
        X = X.copy()
        ca = pd.to_numeric(X["ca"], errors="coerce")
        thal = pd.to_numeric(X["thal"], errors="coerce")
        # .mode() peut renvoyer plusieurs valeurs ex-aequo -> on garde la première
        self.mode_ca_ = ca.mode(dropna=True).iloc[0]
        self.mode_thal_ = thal.mode(dropna=True).iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        X["ca"] = pd.to_numeric(X["ca"], errors="coerce").fillna(self.mode_ca_)
        X["thal"] = pd.to_numeric(X["thal"], errors="coerce").fillna(self.mode_thal_)
        return X


def enrichir_et_flaguer(X):
    """
    Fonction déterministe (sans apprentissage) qui reproduit les étapes 4 et 5 du
    notebook de preprocessing : ajout des variables enrichies + des colonnes flags.
    Appliquée aussi bien à l'entraînement qu'à la prédiction d'un seul patient.
    """
    X = X.copy()

    # --- Enrichissement clinique ---
    X["fc_max_theorique"] = 220 - X["age"]
    X["reserve_cardiaque"] = X["fc_max_theorique"] - X["thalach"]
    X["ratio_chol_age"] = X["chol"] / X["age"]
    X["charge_vasculaire"] = X["trestbps"] * (1 + X["oldpeak"] / 10)
    X["score_risque"] = (
        (X["age"] > 55).astype(int)
        + (X["sex"] == 1).astype(int)
        + (X["chol"] > 240).astype(int)
        + (X["trestbps"] >= 140).astype(int)
        + (X["fbs"] == 1).astype(int)
    )

    # --- Flags cliniques (seuils médicaux usuels) ---
    X["flag_hypertension"] = (X["trestbps"] >= 140).astype(int)
    X["flag_hypercholesterolemie"] = (X["chol"] > 240).astype(int)
    X["flag_glycemie_elevee"] = (X["fbs"] == 1).astype(int)
    X["flag_angine_effort"] = (X["exang"] == 1).astype(int)
    X["flag_ecg_anormal"] = (X["restecg"] != 0).astype(int)
    X["flag_depression_st_importante"] = (X["oldpeak"] > 2).astype(int)
    X["flag_senior"] = (X["age"] >= 60).astype(int)

    return X


def build_preprocessing_steps():
    """
    Construit les étapes de prétraitement communes (imputation -> enrichissement ->
    transformation -> PCA), réutilisées à la fois par le pipeline d'entraînement
    (avec SMOTE + modèle, voir train_pipeline.py) et par le pipeline d'inférence
    sauvegardé pour la production.
    """
    etape_imputation = ImputeCaThal()
    etape_enrichissement = FunctionTransformer(enrichir_et_flaguer, validate=False)

    etape_transformation = ColumnTransformer(transformers=[
        ("num", StandardScaler(), COLONNES_NUMERIQUES),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), COLONNES_CATEGORIELLES),
        ("bin", "passthrough", COLONNES_BINAIRES),
    ])

    etape_pca = PCA(n_components=N_COMPOSANTES_PCA, random_state=RANDOM_STATE)

    return [
        ("imputation", etape_imputation),
        ("enrichissement", etape_enrichissement),
        ("transformation", etape_transformation),
        ("pca", etape_pca),
    ]


def flags_lisibles(patient: dict) -> list:
    """
    Recalcule, pour l'affichage côté interface, les indicateurs cliniques lisibles
    par un humain à partir des données brutes saisies par l'utilisateur.
    Indépendant du modèle : sert uniquement à expliquer le contexte du résultat.
    """
    libelles = []
    if patient["trestbps"] >= 140:
        libelles.append("Pression artérielle au repos élevée (≥ 140 mmHg)")
    if patient["chol"] > 240:
        libelles.append("Cholestérol élevé (> 240 mg/dl)")
    if patient["fbs"] == 1:
        libelles.append("Glycémie à jeun élevée (> 120 mg/dl)")
    if patient["exang"] == 1:
        libelles.append("Angine de poitrine déclenchée par l'effort")
    if patient["restecg"] != 0:
        libelles.append("ECG au repos anormal")
    if patient["oldpeak"] > 2:
        libelles.append("Sous-décalage ST important à l'effort")
    if patient["age"] >= 60:
        libelles.append("Patient d'âge avancé (≥ 60 ans)")
    return libelles
