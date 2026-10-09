// Coeur Analytique — formulaire, appel à /api/predict, affichage du résultat

const form = document.getElementById("patient-form");
const submitBtn = document.getElementById("submit-btn");
const errorMsg = document.getElementById("error-msg");
const placeholder = document.getElementById("placeholder");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const pulseResult = document.getElementById("pulse-result");

const CHAMPS_NUMERIQUES = [
  "age", "poids", "taille", "freq_cardiaque", "ta_systolique",
  "ta_diastolique", "ldl", "glycemie", "acide_urique",
];

const NIVEAUX = {
  faible: { texte: "Risque faible", classe: "sain", pouls: "#9FE3C7" },
  intermediaire: { texte: "Risque intermédiaire", classe: "moyen", pouls: "#F1D38A" },
  eleve: { texte: "Risque élevé", classe: "malade", pouls: "#F2A68E" },
};

function afficherEtat(etat) {
  placeholder.hidden = etat !== "attente";
  loading.hidden = etat !== "chargement";
  result.hidden = etat !== "resultat";
}

// IMC recalculé en direct à partir du poids et de la taille
const champ = (nom) => form.querySelector(`[name="${nom}"]`);

function majIMC() {
  const poids = parseFloat(champ("poids").value);
  const taille = parseFloat(champ("taille").value);
  const el = document.getElementById("imc-value");
  el.textContent = poids > 0 && taille > 0
    ? (poids / Math.pow(taille / 100, 2)).toFixed(1).replace(".", ",")
    : "—";
}
champ("poids").addEventListener("input", majIMC);
champ("taille").addEventListener("input", majIMC);
majIMC();

function collecterDonnees() {
  const data = new FormData(form);
  const patient = {};
  for (const [cle, valeur] of data.entries()) {
    patient[cle] = CHAMPS_NUMERIQUES.includes(cle) ? Number(valeur) : valeur;
  }
  return patient;
}

// Le modèle est très tranché : on n'affiche jamais « 0 % » ni « 100 % »
function formaterProba(p) {
  if (p > 0.99) return "> 99 %";
  if (p < 0.01) return "< 1 %";
  return Math.round(p * 100) + " %";
}

function remplirListe(ul, items, vide) {
  ul.innerHTML = "";
  if (items.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = vide;
    ul.appendChild(li);
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = item.libelle;
    const track = document.createElement("span");
    track.className = "track";
    const fill = document.createElement("span");
    fill.className = "fill";
    fill.style.width = Math.max(6, Math.round(item.poids * 100)) + "%";
    track.appendChild(fill);
    li.append(label, track);
    ul.appendChild(li);
  }
}

function afficherResultat(r) {
  const niveau = NIVEAUX[r.niveau];
  const badge = document.getElementById("verdict-badge");
  badge.textContent = niveau.texte;
  badge.className = "verdict-badge " + niveau.classe;

  document.getElementById("proba-value").textContent = formaterProba(r.probabilite);

  const warn = document.getElementById("range-warning");
  warn.hidden = r.hors_plage.length === 0;
  warn.textContent = r.hors_plage.join(" ");

  remplirListe(document.querySelector("#factors-up .bars"), r.facteurs.risque,
    "Aucun facteur ne dépasse le patient moyen.");
  remplirListe(document.querySelector("#factors-down .bars"), r.facteurs.protecteur,
    "Aucun facteur protecteur marqué.");

  const flags = document.getElementById("flags-list");
  flags.innerHTML = "";
  flags.classList.toggle("empty", r.indicateurs.length === 0);
  const liste = r.indicateurs.length ? r.indicateurs : ["Aucun facteur de risque clinique usuel détecté."];
  for (const t of liste) {
    const li = document.createElement("li");
    li.textContent = t;
    flags.appendChild(li);
  }

  const m = r.modele;
  document.getElementById("model-info").textContent =
    `${m.nom} · ROC-AUC ${String(m.roc_auc).replace(".", ",")} et exactitude ${Math.round(m.accuracy * 100)} % sur des patients de test (base de ${m.n_total} patients).`;

  // Un seul mouvement orchestré : le tracé du pouls se dessine quand le résultat est prêt
  pulseResult.style.stroke = niveau.pouls;
  pulseResult.classList.remove("draw");
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
    const reponse = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collecterDonnees()),
    });
    const donnees = await reponse.json().catch(() => ({}));
    if (!reponse.ok) throw new Error(donnees.detail || "Le serveur n'a pas pu traiter cette demande.");
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
