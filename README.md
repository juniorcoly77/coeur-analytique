# Coeur Analytique — risque de crise cardiaque

Site web qui estime la probabilité de risque de crise cardiaque d'un patient à partir de
16 informations (clinique, biologie, mode de vie). Modèle : régression logistique entraînée
sur le dataset NHANES (1 000 patients). Déploiement : Vercel (site statique + fonction Python).

## Structure

```
public/                 frontend (index.html, style.css, script.js)
api/predict.py          fonction serverless Vercel (Flask) : POST /api/predict
api/requirements.txt    dépendances de production
heart_model/
  pipeline.py           prétraitement, enrichissement, flags, explications (module partagé)
  model_pipeline.joblib modèle entraîné
  explication.json      moyennes d'entraînement + plages observées
  metrics.json          performances mesurées sur le jeu de test
data/                   dataset d'entraînement
train_model.py          ré-entraîne le modèle et régénère les 3 fichiers de heart_model/
heart_attack_nhanes_modelisation.ipynb   démarche complète commentée (EDA -> évaluation)
vercel.json
```

## Variables saisies (16)

Âge, sexe, poids, taille (l'IMC est recalculé), type de douleur thoracique, angine à l'effort,
ECG au repos, sous-décalage ST, fréquence cardiaque, pression systolique et diastolique,
LDL, glycémie, acide urique, tabagisme, activité physique.

## Ré-entraîner le modèle

```bash
pip install -r requirements-train.txt
PYTHONPATH=. python train_model.py
```

Régénère `model_pipeline.joblib`, `explication.json` et `metrics.json`. **Commit ces trois
fichiers** : Vercel les embarque dans la fonction (`includeFiles` dans `vercel.json`).
Les versions de `api/requirements.txt` doivent rester identiques à celles utilisées pour
entraîner (un `.joblib` n'est pas garanti compatible entre versions de scikit-learn).

## Tester en local

```bash
pip install -r api/requirements.txt
PYTHONPATH=. python -c "from api.predict import app; app.run(port=5001)"
curl -X POST localhost:5001/api/predict -H "Content-Type: application/json" -d '{
  "age":68,"sexe":"Homme","type_douleur":"Typique angineuse","ta_systolique":164,"ta_diastolique":95,
  "ldl":150,"glycemie":127,"ecg_repos":"Normal","poids":74,"taille":169,"acide_urique":7.0,
  "tabagisme":"Oui","sedentarite":"Sédentaire","freq_cardiaque":79,"angine_effort":"Oui","sous_decalage_st":"Non"}'
```

Pour voir le site complet en local, utiliser `vercel dev` (la CLI Vercel sert `public/` et `api/`).

## Déployer sur Vercel

1. Pousser le dépôt sur GitHub, puis l'importer sur vercel.com/new.
2. Répertoire racine : `./`. Préréglage : **Other** (ne pas utiliser « Services »).
3. Deploy. Vérifier ensuite `POST /api/predict` avec la commande curl ci-dessus.

## Limites

* Le jeu de données semble simulé à partir de règles simples : les scores obtenus
  (ROC-AUC ≈ 0,99) ne se transposeraient pas à des patients réels.
* Entraîné sur des patients de 30 à 80 ans : l'API signale les saisies hors de cette plage.
* Outil pédagogique, pas un dispositif médical.
