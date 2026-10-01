// Coeur Analytique — logique du formulaire et affichage du résultat
// Appelle POST /api/predict (servi soit par FastAPI/Docker, soit par la fonction Vercel).

const form = document.getElementById("patient-form");
const submitBtn = document.getElementById("submit-btn");
const errorMsg = document.getElementById("error-msg");

const placeholder = document.getElementById("placeholder");
const loading = document.getElementById("loading");
const result = document.getElementById("result");

const verdictBadge = document.getElementById("verdict-badge");
const probaValue = document.getElementById("proba-value");
const flagsList = document.getElementById("flags-list");
const pulseResult = document.getElementById("pulse-result");

// Champs à convertir en nombre avant l'envoi (le reste part en chaîne de caractères)
const CHAMPS_NUMERIQUES = [
  "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
  "thalach", "exang", "oldpeak", "slope", "ca", "thal",
];

function afficherEtat(etat) {
  // etat : 'attente' | 'chargement' | 'resultat'
  placeholder.hidden = etat !== "attente";
  loading.hidden = etat !== "chargement";
  result.hidden = etat !== "resultat";
}

function collecterDonnees() {
  const data = new FormData(form);
  const patient = {};
  for (const champ of CHAMPS_NUMERIQUES) {
    patient[champ] = Number(data.get(champ));
  }
  return patient;
}

function afficherResultat(reponse) {
  const proba = reponse.probabilite; // 0..1
  const estMalade = reponse.prediction === 1;

  verdictBadge.textContent = estMalade ? "Risque élevé détecté" : "Profil favorable";
  verdictBadge.className = "verdict-badge " + (estMalade ? "malade" : "sain");

  probaValue.textContent = Math.round(proba * 100) + "%";

  flagsList.innerHTML = "";
  if (reponse.indicateurs.length === 0) {
    flagsList.classList.add("empty");
    const li = document.createElement("li");
    li.textContent = "Aucun facteur de risque clinique usuel détecté.";
    flagsList.appendChild(li);
  } else {
    flagsList.classList.remove("empty");
    for (const indicateur of reponse.indicateurs) {
      const li = document.createElement("li");
      li.textContent = indicateur;
      flagsList.appendChild(li);
    }
  }

  // Un seul mouvement orchestré : le tracé du pouls se dessine une fois le résultat prêt
  pulseResult.style.stroke = estMalade ? "#F2A68E" : "#9FE3C7";
  pulseResult.classList.remove("draw");
  // force reflow pour pouvoir rejouer l'animation à chaque nouvelle analyse
  void pulseResult.getBoundingClientRect();
  pulseResult.classList.add("draw");

  afficherEtat("resultat");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMsg.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Analyse en cours…";
  afficherEtat("chargement");

  try {
    const patient = collecterDonnees();
    const reponse = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patient),
    });

    if (!reponse.ok) {
      const detail = await reponse.json().catch(() => ({}));
      throw new Error(detail.detail || "Le serveur n'a pas pu traiter cette demande.");
    }

    const donnees = await reponse.json();
    afficherResultat(donnees);
  } catch (err) {
    afficherEtat("attente");
    errorMsg.textContent = err.message || "Une erreur est survenue. Réessayez.";
    errorMsg.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Analyser le patient";
  }
});
