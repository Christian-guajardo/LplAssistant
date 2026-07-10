# star-hengxing/deep-research — le harnais d'orchestration piloté par l'agent

**Source :** dossier local `repos_storage/star-hengxing_deep-research/` (POC Python d'une connaissance)
**Type :** CLI Python (~1 650 lignes : `app.py`, `project.py`, `state.py`, `models.py`, `quality.py`, `pdf.py`) + `SKILL.md`, géré par pixi | **Analysé le :** 2026-07-10 (lecture intégrale du code)

## En une phrase

Une inversion de contrôle astucieuse : le CLI ne contient **aucun appel LLM** — c'est un *harnais* (état, qualité, PDF) conçu comme une skill Claude Code, où l'agent IA fait la recherche et le CLI garantit la discipline. La logique d'orchestration est dans `SKILL.md`, pas dans le code.

## Architecture & pipeline

- **8 commandes** : `init` → `plan "dir1" "dir2"…` → `add` (direction tardive) → `agents` (prompts par direction) → `status` → `validate` → `generate` (PDF/HTML).
- **Machine à états JSON** (`.deep-research-state.json`, `state.py` lu en entier) :
  - phases strictement ordonnées `initialized → planned → researching → synthesized → reviewed → generated → complete`, transitions **monotones** (retour en arrière refusé), état `error` accessible de partout et réinitialisable ;
  - par-phase : `{status, started_at, completed_at, error}` ; par-agent (= direction) : `{id, topic, description, output_dir, status, report_path}` ;
  - `next_phase()` rend la reprise triviale : première phase incomplète = où reprendre. Versionné (`version: "1"`), phase inconnue ⇒ fallback + warning (migration douce).
- **`.meta.json` obligatoire par direction** : chaque agent chercheur doit remplir modèle, tokens totaux, liens cherchés, durée — sinon la direction **n'apparaît pas dans l'annexe du PDF**. La traçabilité est forcée par la chaîne de build, pas par la bonne volonté.
- **Quality gates** (`quality.py` lu en entier) : extraction regex des URLs du rapport → HEAD puis GET fallback (URLs percent-encodées, UA dédié) ; **check de connectivité préalable** (offline ⇒ skip = pass, pas de faux rouge) ; auto-fix prettier + markdownlint ; vérif du format de la bibliographie (lignes `[n]` séparées par ligne vide sinon pandoc les fusionne).
- **Génération PDF** : pipeline « kami » (weasyprint) par défaut, fallback pandoc+typst ; annexe auto (métadonnées modèles, tokens, liens) ; filtres Lua (`---` → saut de page).

## Les règles opérationnelles de SKILL.md (de l'or)

- **Assemblage progressif obligatoire** : « NEVER write the full README in one shot » — sections ≤ ~2 000 mots écrites une à une en append, parce que les limites de tokens de sortie tronquent les longs rapports. Directement applicable à tout rédacteur LLM local.
- L'agent de synthèse doit vérifier les `.meta.json` manquants (`for d in agent-*/; do test -f …`) avant de rédiger.
- Contraintes markdown-pour-PDF explicites (pas de `---` dans le corps, pas d'emoji dans les tableaux, max 4 colonnes).
- « When NOT to use » assumé : lookups simples, débogage, réponses en 1-2 recherches.

## Limites

- Pas autonome : il FAUT un agent (Claude Code) au-dessus ; le CLI ne cherche pas, ne synthétise pas.
- Deux environnements pixi (dev/runtime) — soigné mais lourd pour un déploiement Laplace.

## À réutiliser pour Laplace

- **La machine à états + checkpoint JSON telle quelle** (portage C++ direct : enum de phases, transitions monotones, `next_phase()`) — c'est la réponse au besoin n°1 : un run de recherche de 20 min qui survit à un crash/reboot.
- Le contrat `.meta.json` : notre pipeline doit **refuser** de publier un rapport dont une branche n'a pas ses métadonnées (tokens, sources, durée).
- Les quality gates URL (HEAD→GET, offline-aware) en C++ : trivial avec libcurl, à exécuter avant toute synthèse finale.
- La rédaction progressive par sections pour le rédacteur final de Laplace.
- L'idée méta : **séparer harnais déterministe (C++) et intelligence (LLM)** — le C++ tient l'état, les gates et les budgets ; le LLM ne tient que le raisonnement.

## Liens à creuser (besoins)

- `docs/kami-pdf-pipeline.md` (pipeline weasyprint) si on veut des sorties PDF un jour.
