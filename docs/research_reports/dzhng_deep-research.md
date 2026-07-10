# dzhng/deep-research — la boucle récursive minimale

**Source :** <https://github.com/dzhng/deep-research>
**Type :** CLI/API Node.js (TypeScript) | **Taille :** ~560 lignes de code cœur | **Analysé le :** 2026-07-10 (lecture intégrale de `src/deep-research.ts`, `src/prompt.ts`)

## En une phrase

La preuve qu'un deep research efficace tient en ~300 lignes : un arbre de recherche récursif paramétré par `breadth` (largeur) et `depth` (profondeur), qui accumule des « learnings » et les réinjecte à chaque niveau.

## Architecture & pipeline

1. **Questions de clarification** (`feedback.ts`) : avant de lancer quoi que ce soit, le LLM pose 2-3 questions à l'utilisateur pour cadrer l'intention.
2. **`generateSerpQueries`** : le LLM produit jusqu'à `breadth` requêtes SERP en JSON structuré (schéma Zod). Chaque requête porte deux champs : `query` **et** `researchGoal` (l'objectif + les directions futures) — c'est ce second champ qui pilote la récursion.
3. **Recherche + scraping** : Firecrawl (`search` avec `scrapeOptions: {formats: ['markdown']}`), 5 résultats max, timeout 15 s, concurrence bornée par `p-limit` (défaut 2).
4. **`processSerpResult`** : chaque contenu est tronqué à **25 000 caractères** (`trimPrompt`), puis le LLM en extrait `learnings` (max 3, denses, avec entités/chiffres/dates exacts) et `followUpQuestions` (max = nouvelle largeur).
5. **Récursion** : `newBreadth = ceil(breadth/2)`, `newDepth = depth-1`. La nouvelle requête = `researchGoal` précédent + follow-ups. Les learnings et URLs s'accumulent (dédupliqués par `Set`).
6. **Sortie** : `writeFinalReport` (rapport markdown 3+ pages, tous les learnings, section Sources) ou `writeFinalAnswer` (réponse courte exacte, format imposé — mode « bench »).

## Techniques à retenir

- **La géométrie breadth/depth** : largeur divisée par 2 à chaque niveau ⇒ le coût est contrôlé (b + b/2 + b/4… par branche) tout en explorant large au départ et précis en profondeur.
- **Learnings = mémoire de la boucle.** Pas de résumé glissant : une liste plate de faits denses, dédupliqués, passés au prompt suivant pour générer des requêtes *plus spécifiques*.
- **Chaque étage de la boucle exige du JSON structuré** (Zod + `generateObject`). En local, ça se traduit directement par des grammaires GBNF.
- **Timeout par appel LLM** (`AbortSignal.timeout(60_000)`) et erreurs avalées par branche (une branche qui échoue rend `{learnings:[], urls:[]}` sans tuer l'arbre).
- Le system prompt est court et réutilisable : « expert researcher », date du jour injectée, « be as detailed as possible », « flag speculation ».

## Dépendances externes & limites

- **Firecrawl** fait 100 % du travail recherche+extraction (API payante, ou self-host via `FIRECRAWL_BASE_URL`). Sans équivalent local, ce design ne tourne pas.
- Modèles : OpenAI o3-mini / DeepSeek R1 via Fireworks / tout endpoint OpenAI-compatible (`OPENAI_ENDPOINT` custom) — donc branchable sur `llama-server`.
- Pas de checkpoint/resume : un crash au niveau 3 perd tout.
- Pas d'évaluation de fiabilité des sources, pas de citations inline (juste une liste d'URLs à la fin).

## À réutiliser pour Laplace

- **C'est le squelette idéal du module C++** : l'algorithme complet (2 prompts + 1 boucle récursive) est portable en une après-midi. Structs `ResearchProgress`/`ResearchResult`, `p-limit` → pool de threads borné.
- Le pattern `researchGoal` (objectif + directions) comme fil conducteur de la récursion est ce qui rend les requêtes de niveau N+1 intelligentes — le conserver tel quel.
- Le paramètre de troncature (25k chars/source) et learnings max 3/source sont de bons défauts pour un modèle 8-14B local.
- Remplacer Firecrawl par : SearXNG local (recherche) + lecteur HTML→markdown local (extraction). C'est LE besoin bloquant identifié.

## Liens à creuser (besoins)

- Firecrawl self-host (<https://github.com/mendableai/firecrawl>) — option lourde ; comparer avec un lecteur maison.
- L'appli sœur qui l'héberge : <https://deep-research.exp.dzhng.com>.
