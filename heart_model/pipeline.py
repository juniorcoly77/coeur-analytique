"""
Prétraitement V2 — dataset NHANES, cible : Risque_de_crise_cardiaque (Oui/Non).

Ce module est importé au chargement du modèle en production (Vercel) : il ne doit
dépendre que de scikit-learn / pandas / numpy (pas d'imbalanced-learn, voir
train_pipeline_v2.py pour la partie entraînement).

Entrées brutes attendues (16 variables, noms internes en snake_case) :
    age, sexe, type_douleur, ta_systolique, ta_diastolique, ldl, glycemie,
    ecg_repos, poids, taille, acide_urique, tabagisme, sedentarite,
    freq_cardiaque, angine_effort, sous_decalage_st
L'IMC n'est PAS saisi : il est recalculé à partir du poids et de la taille.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# Correspondance colonnes du fichier CSV -> noms internes
RENOMMAGE = {
    "Age": "age",
    "Sexe": "sexe",
    "Type_douleur_thoracique": "type_douleur",
    "TA_repos_systolique_mmHg": "ta_systolique",
    "TA_repos_diastolique_mmHg": "ta_diastolique",
    "LDL_mg_dL": "ldl",
    "Glycemie_mg_dL": "glycemie",
    "ECG_repos": "ecg_repos",
    "Poids_kg": "poids",
    "Taille_cm": "taille",
    "Acide_urique_mg_dL": "acide_urique",
    "Tabagisme": "tabagisme",
    "Sedentarite_activite_sportive": "sedentarite",
    "Frequence_cardiaque_bpm": "freq_cardiaque",
    "Angine_declenchee_effort": "angine_effort",
    "Sous_decalage_ST": "sous_decalage_st",
}
# IMC_kg_m2 est volontairement ignoré : recalculé depuis poids/taille

COLONNES_BRUTES = list(RENOMMAGE.values())

TYPES_DOULEUR = ["Typique angineuse", "Atypique", "Non angineuse", "Asymptomatique"]
TYPES_ECG = ["Normal", "Anomalie ST-T", "Hypertrophie ventriculaire"]

# Variables utilisées par le modèle. pression_pulsee, pam et score_risque sont calculées par
# `preparer` (utiles pour des modèles non linéaires ou l'exploration) mais volontairement
# exclues ici : ce sont des combinaisons linéaires d'autres variables, donc sans apport pour
# une régression logistique, et elles brouillent l'interprétation des coefficients.
COLONNES_NUMERIQUES = [
    "age", "ta_systolique", "ta_diastolique", "ldl", "glycemie", "imc",
    "acide_urique", "freq_cardiaque",
]
COLONNES_CATEGORIELLES = ["type_douleur", "ecg_repos"]
COLONNES_BINAIRES_BASE = ["sexe", "tabagisme", "sedentarite", "angine_effort", "sous_decalage_st"]
COLONNES_FLAGS = [
    "flag_hypertension", "flag_ldl_eleve", "flag_hyperglycemie", "flag_obesite",
    "flag_acide_urique_eleve", "flag_freq_anormale", "flag_senior",
]
COLONNES_BINAIRES = COLONNES_BINAIRES_BASE + COLONNES_FLAGS


def _oui_non(serie):
    """Oui/Non -> 1/0 (insensible à la casse et aux espaces)."""
    return serie.astype(str).str.strip().str.lower().eq("oui").astype(int)


def preparer(X):
    """
    Étape déterministe (sans apprentissage) : nettoyage, encodage binaire,
    enrichissement (IMC, pression pulsée, PAM) et création des flags cliniques.
    """
    X = X.copy()

    # --- Nettoyage / encodage des variables binaires ---
    X["sexe"] = X["sexe"].astype(str).str.strip().str.lower().eq("homme").astype(int)
    for col in ["tabagisme", "angine_effort", "sous_decalage_st"]:
        X[col] = _oui_non(X[col])
    # 'Sédentaire' peut arriver avec un accent corrompu dans le CSV source -> on teste 'dentaire'
    X["sedentarite"] = X["sedentarite"].astype(str).str.lower().str.contains("dentaire").astype(int)
    for col in ["type_douleur", "ecg_repos"]:
        X[col] = X[col].astype(str).str.strip()

    # --- Enrichissement ---
    X["imc"] = X["poids"] / (X["taille"] / 100) ** 2
    X["pression_pulsee"] = X["ta_systolique"] - X["ta_diastolique"]
    X["pam"] = (X["ta_systolique"] + 2 * X["ta_diastolique"]) / 3

    # --- Flags cliniques (seuils médicaux usuels) ---
    X["flag_hypertension"] = ((X["ta_systolique"] >= 140) | (X["ta_diastolique"] >= 90)).astype(int)
    X["flag_ldl_eleve"] = (X["ldl"] >= 160).astype(int)
    X["flag_hyperglycemie"] = (X["glycemie"] >= 126).astype(int)
    X["flag_obesite"] = (X["imc"] >= 30).astype(int)
    seuil_urique = np.where(X["sexe"] == 1, 7.0, 6.0)  # seuils différents homme / femme
    X["flag_acide_urique_eleve"] = (X["acide_urique"] > seuil_urique).astype(int)
    X["flag_freq_anormale"] = ((X["freq_cardiaque"] > 100) | (X["freq_cardiaque"] < 50)).astype(int)
    X["flag_senior"] = (X["age"] >= 60).astype(int)

    # Score cumulatif : nombre de facteurs de risque présents
    X["score_risque"] = (
        X["flag_hypertension"] + X["flag_ldl_eleve"] + X["flag_hyperglycemie"]
        + X["flag_obesite"] + X["flag_acide_urique_eleve"] + X["flag_senior"]
        + X["tabagisme"] + X["sedentarite"]
    )
    return X


def build_preprocessing_steps(use_pca=False, n_composantes=None):
    """Étapes communes : préparation -> imputation/scaling/encodage -> (PCA optionnelle)."""
    numerique = Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categoriel = Pipeline([
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            categories=[TYPES_DOULEUR, TYPES_ECG], handle_unknown="ignore", sparse_output=False
        )),
    ])
    transformation = ColumnTransformer([
        ("num", numerique, COLONNES_NUMERIQUES),
        ("cat", categoriel, COLONNES_CATEGORIELLES),
        ("bin", "passthrough", COLONNES_BINAIRES),
    ])
    etapes = [
        ("preparation", FunctionTransformer(preparer, validate=False)),
        ("transformation", transformation),
    ]
    if use_pca:
        etapes.append(("pca", PCA(n_components=n_composantes, random_state=RANDOM_STATE)))
    return etapes


def flags_lisibles(p: dict) -> list:
    """Facteurs de risque détectés, formulés pour l'affichage (indépendant du modèle)."""
    imc = p["poids"] / (p["taille"] / 100) ** 2
    out = []
    if p["ta_systolique"] >= 140 or p["ta_diastolique"] >= 90:
        out.append("Hypertension artérielle (≥ 140/90 mmHg)")
    if p["ldl"] >= 160:
        out.append("LDL-cholestérol élevé (≥ 160 mg/dl)")
    if p["glycemie"] >= 126:
        out.append("Glycémie élevée (≥ 126 mg/dl)")
    if imc >= 30:
        out.append(f"Obésité (IMC {imc:.1f})")
    seuil = 7.0 if str(p["sexe"]).strip().lower() == "homme" else 6.0
    if p["acide_urique"] > seuil:
        out.append(f"Acide urique élevé (> {seuil:g} mg/dl)")
    if p["freq_cardiaque"] > 100 or p["freq_cardiaque"] < 50:
        out.append("Fréquence cardiaque anormale")
    if str(p["tabagisme"]).strip().lower() == "oui":
        out.append("Tabagisme")
    if "dentaire" in str(p["sedentarite"]).lower():
        out.append("Mode de vie sédentaire")
    if str(p["angine_effort"]).strip().lower() == "oui":
        out.append("Angine déclenchée par l'effort")
    if str(p["sous_decalage_st"]).strip().lower() == "oui":
        out.append("Sous-décalage du segment ST")
    if p["ecg_repos"] != "Normal":
        out.append(f"ECG au repos : {p['ecg_repos'].lower()}")
    if p["age"] >= 60:
        out.append("Âge avancé (≥ 60 ans)")
    return out


def charger_donnees(chemin):
    """Charge le CSV NHANES (séparateur ';', virgule décimale) et renomme les colonnes."""
    df = pd.read_csv(chemin, sep=";", decimal=",", encoding="utf-8", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    y = df["Risque_de_crise_cardiaque"].eq("Oui").astype(int)
    X = df.rename(columns=RENOMMAGE)[COLONNES_BRUTES]
    return X, y


# ---------------------------------------------------------------------------
# Explication locale d'une prédiction (modèle linéaire : contribution = coef x écart à la moyenne)
# ---------------------------------------------------------------------------
# Les contributions sont agrégées par variable clinique : une valeur continue et son indicateur
# de seuil (ex. LDL et « LDL élevé ») portent le même signal et ne doivent pas être lus séparément.
GROUPES_NUMERIQUES = {
    "age": ["num__age", "bin__flag_senior"],
    "tension": ["num__ta_systolique", "num__ta_diastolique", "bin__flag_hypertension"],
    "ldl": ["num__ldl", "bin__flag_ldl_eleve"],
    "glycemie": ["num__glycemie", "bin__flag_hyperglycemie"],
    "imc": ["num__imc", "bin__flag_obesite"],
    "acide_urique": ["num__acide_urique", "bin__flag_acide_urique_eleve"],
    "freq_cardiaque": ["num__freq_cardiaque", "bin__flag_freq_anormale"],
}
GROUPES_BINAIRES = {  # feature -> (libellé si présent, libellé si absent)
    "bin__sexe": ("Sexe masculin", "Sexe féminin"),
    "bin__tabagisme": ("Tabagisme", "Non-fumeur"),
    "bin__sedentarite": ("Sédentarité", "Activité physique régulière"),
    "bin__angine_effort": ("Angine à l'effort", "Pas d'angine à l'effort"),
    "bin__sous_decalage_st": ("Sous-décalage ST", "Pas de sous-décalage ST"),
}


def _libelle_numerique(cle, r):
    imc = r["poids"] / (r["taille"] / 100) ** 2
    return {
        "age": f"Âge ({r['age']:g} ans)",
        "tension": f"Tension artérielle ({r['ta_systolique']:g}/{r['ta_diastolique']:g} mmHg)",
        "ldl": f"LDL-cholestérol ({r['ldl']:g} mg/dl)",
        "glycemie": f"Glycémie ({r['glycemie']:g} mg/dl)",
        "imc": f"IMC ({imc:.1f})".replace(".", ","),
        "acide_urique": f"Acide urique ({r['acide_urique']:g} mg/dl)",
        "freq_cardiaque": f"Fréquence cardiaque ({r['freq_cardiaque']:g} bpm)",
    }[cle]


def facteurs_contributifs(pipeline, moyennes, X_patient, top=4):
    """
    Variables qui poussent le plus le score du modèle vers le HAUT (risque) ou vers le BAS
    (protecteur), par rapport au patient moyen du jeu d'entraînement. Valable car le modèle
    final est linéaire. Les libellés décrivent l'état réel du patient.
    """
    prepa = pipeline.named_steps["preparation_transformation"]
    modele = pipeline.named_steps["modele"]
    x = prepa.transform(X_patient)[0]
    noms = list(prepa.named_steps["transformation"].get_feature_names_out())
    contrib = dict(zip(noms, modele.coef_[0] * (x - np.asarray(moyennes))))
    valeur = dict(zip(noms, x))
    r = X_patient.iloc[0]

    items = []
    for cle, feats in GROUPES_NUMERIQUES.items():
        items.append((_libelle_numerique(cle, r), sum(contrib[f] for f in feats)))
    for f, (present, absent) in GROUPES_BINAIRES.items():
        items.append((present if valeur[f] == 1 else absent, contrib[f]))
    for col, titre in [("type_douleur", "Douleur thoracique"), ("ecg_repos", "ECG de repos")]:
        total = sum(c for n, c in contrib.items() if n.startswith(f"cat__{col}_"))
        items.append((f"{titre} : {str(r[col]).lower()}", total))

    items = [(n, c) for n, c in items if abs(c) > 0.05]
    plus = sorted([t for t in items if t[1] > 0], key=lambda t: -t[1])[:top]
    moins = sorted([t for t in items if t[1] < 0], key=lambda t: t[1])[:top]
    maxi = max([abs(c) for _, c in plus + moins] + [1e-9])
    fmt = lambda lst, sens: [
        {"libelle": n, "sens": sens, "poids": round(abs(c) / maxi, 3)} for n, c in lst
    ]
    return {"risque": fmt(plus, "augmente"), "protecteur": fmt(moins, "diminue")}
