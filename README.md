# Coeur Analytique

Outil web d'aide à la décision qui estime, à partir de 13 variables cliniques, une
probabilité de maladie cardiaque à l'aide d'un modèle Random Forest entraîné sur le
dataset Heart Disease (Cleveland).

```
heart_model/
  pipeline.py          # prétraitement (imputation, enrichissement, flags, scaling, PCA) — SANS imbalanced-learn
  train_pipeline.py     # ajoute SMOTE + RandomForest — utilisé UNIQUEMENT à l'entraînement
  model_pipeline.joblib # artefact entraîné, chargé en production (Docker ET Vercel)
app/
  main.py                # API FastAPI — sert /api/predict + le frontend statique (Docker)
api/
  predict.py              # fonction serverless Flask équivalente, pour Vercel
  requirements.txt        # dépendances Python propres à la fonction Vercel
public/
  index.html, style.css, script.js   # frontend (identique pour les deux déploiements)
train_model.py             # ré-entraîne le modèle et régénère model_pipeline.joblib
Dockerfile, docker-compose.yml
vercel.json
```

## ⚠️ À savoir avant de déployer

**Vercel n'exécute pas d'image Docker.** Le `Dockerfile` fourni sert à un déploiement
conteneurisé classique (local, ou sur une plateforme qui exécute des conteneurs comme
Render, Fly.io, Railway ou AWS App Runner). Pour Vercel, le même modèle et la même
logique de prétraitement sont réutilisés dans une **fonction Python serverless**
(`api/predict.py`) — ce sont deux chemins de déploiement indépendants qui partagent le
même code (`heart_model/`) et le même frontend (`public/`), mais ce n'est pas le
conteneur Docker qui tourne sur Vercel.

## Ré-entraîner le modèle

```bash
pip install -r requirements-train.txt
python train_model.py
```

Régénère `heart_model/model_pipeline.joblib`. Le pipeline sauvegardé exclut SMOTE
(actif seulement à l'entraînement) pour rester léger et ne dépendre que de
scikit-learn/pandas/numpy en production.

## Lancer en local sans Docker

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# -> http://localhost:8000
```

## Conteneuriser avec Docker

```bash
docker compose up --build
# -> http://localhost:8000
```

Ou directement avec Docker :

```bash
docker build -t coeur-analytique .
docker run -p 8000:8000 coeur-analytique
```

Le `Dockerfile` :
- installe les dépendances avant de copier le code (cache Docker efficace) ;
- tourne avec un utilisateur non-root ;
- expose un `HEALTHCHECK` sur `/api/health`.

Pour déployer ce conteneur sur une plateforme qui accepte Docker (Render, Fly.io,
Railway, AWS App Runner...), pousser l'image construite et exposer le port 8000 ;
aucune modification du code n'est nécessaire.

## Déployer le frontend + l'API sur Vercel

1. Installer la CLI si besoin : `npm i -g vercel`
2. Depuis la racine du projet : `vercel deploy` (ou connecter le dépôt Git dans le
   tableau de bord Vercel pour un déploiement automatique à chaque push).
3. Vercel détecte `vercel.json` :
   - sert `public/` comme site statique (`outputDirectory`) ;
   - construit `api/predict.py` comme fonction Python, avec `heart_model/` (y compris
     le fichier `.joblib`) embarqué via `includeFiles` ;
   - redirige `/api/predict` vers cette fonction.

Les dépendances de la fonction (`api/requirements.txt`) sont volontairement réduites
(Flask, scikit-learn, pandas, numpy, joblib — pas FastAPI/uvicorn, inutiles ici) pour
rester sous la limite Vercel de **500 Mo non compressés pour les fonctions Python**
(≈264 Mo mesurés pour ce projet, cf. `vercel.com/docs/functions/limitations`).

Si un déploiement futur dépasse cette limite (ex. après une mise à jour de
dépendance), les options sont : épingler des versions plus légères, ou activer les
*large functions* de Vercel (jusqu'à 5 Go, nécessite Fluid Compute).

### Architecture alternative

Si vous préférez garder l'API Docker comme unique backend (par ex. déployée sur
Render) et n'utiliser Vercel que pour le frontend statique, il suffit de :
- déployer `public/` seul sur Vercel (`outputDirectory: public`, pas de `functions`) ;
- changer l'URL appelée dans `public/script.js` (`/api/predict`) par l'URL complète de
  l'API Docker déployée ailleurs ;
- activer CORS côté FastAPI (déjà fait dans `app/main.py`, ouvert à `*` — à restreindre
  au domaine Vercel réel en production).

## Vérifications effectuées

- Pipeline testé avec et sans `imbalanced-learn` installé (le pipeline d'inférence ne
  requiert QUE scikit-learn/pandas/numpy).
- API FastAPI testée en local : cas sain, cas à risque, validation des champs invalides
  (422), fichiers statiques servis.
- Fonction Flask (`api/predict.py`) testée via `app.test_client()` : mêmes résultats
  numériques que l'API FastAPI pour les mêmes patients (les deux chargent le même
  `model_pipeline.joblib`).
- Le build et l'exécution de l'image Docker elle-même n'ont pas pu être testés dans cet
  environnement (Docker non disponible) — le `Dockerfile` reprend cependant exactement
  les mêmes dépendances et la même commande de lancement (`uvicorn app.main:app`) déjà
  validées en dehors du conteneur. Vérifiez avec `docker compose up --build` avant un
  déploiement en production.

## Limite du modèle

Dataset d'entraînement réduit (303 patients), usage pédagogique. Ne remplace aucun
avis médical professionnel — rappelé explicitement dans l'interface.
