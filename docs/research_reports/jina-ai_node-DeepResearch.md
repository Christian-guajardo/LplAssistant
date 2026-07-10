# jina-ai/node-DeepResearch — la machine à états Search/Read/Reason

**Source :** <https://github.com/jina-ai/node-DeepResearch>
**Type :** Agent Node.js/TypeScript + serveur OpenAI-compatible | **Cœur :** `src/agent.ts` (~1 200 lignes) + ~20 outils dans `src/tools/` | **Analysé le :** 2026-07-10

## En une phrase

Un agent « answer-first » (trouver LA bonne réponse, pas écrire un long rapport) qui boucle Search → Read → Reason jusqu'à épuisement d'un **budget de tokens**, avec un « Beast Mode » qui force une réponse finale quand le budget est cramé.

## Architecture & pipeline

- **Boucle unique à actions typées** : à chaque étape le LLM choisit une action parmi `search`, `visit` (lire une URL), `reflect` (décomposer en sous-questions), `answer`, `coding`. Le prompt est **reconstruit à chaque étape** (`getPrompt`) en n'incluant que les sections des actions *autorisées*.
- **Gating dynamique des actions** (le vrai savoir-faire du repo, `agent.ts:484-570`) :
  - `allowReflect` coupé si la file de gaps dépasse `MAX_REFLECT_PER_STEP` ;
  - `allowSearch` coupé quand > 50 URLs déjà collectées (anti-explosion) ;
  - `allowRead` seulement si la liste d'URLs pondérées est non vide ;
  - après une réponse jugée mauvaise, `allowAnswer=false` pour forcer plus de recherche.
- **File de « gaps »** : `reflect` pousse des sous-questions dans une file ; l'agent traite la question courante depuis cette file (l'originale y retourne pour être re-visitée).
- **Budget & Beast Mode** : suivi du total de tokens (`tokenTracker`) ; **10 % du budget est réservé** ; une fois dépassé, prompt spécial « beast mode » = réponse immédiate obligatoire avec la connaissance accumulée.
- **Évaluateur** : la réponse candidate passe par `evaluator.ts` (définitude, attribution, fraîcheur, exhaustivité) avant d'être acceptée — sinon la boucle repart avec le feedback d'échec.
- **Pipeline de raffinage des SERP** : `query-rewriter` (réécriture des requêtes), `serp-cluster`, `jina-dedup` (dédup sémantique par embeddings), `jina-rerank` (re-ranking), `jina-latechunk` (late chunking à la lecture), `build-ref` (citations), `finalizer`, `broken-ch-fixer`.
- **Exposition serveur** : endpoint `/v1/chat/completions` OpenAI-compatible qui streame le raisonnement dans des balises `<think>…</think>` — n'importe quel client de chat existant devient une UI de deep research.

## Techniques à retenir

- **Le budget en tokens comme condition d'arrêt** est plus robuste qu'un nombre d'itérations : il capture le vrai coût (grosses pages lues = moins d'étapes).
- **Beast Mode** : garde-fou indispensable contre l'agent qui « n'ose jamais répondre ». À implémenter tel quel.
- **Interdire des actions plutôt que les pénaliser** : retirer la section du prompt est plus fiable qu'une consigne « évite de » — surtout pour des petits modèles.
- La mémoire de l'agent = 4 listes rejouées dans le prompt : connaissances acquises, questions déjà posées, URLs visitées, tentatives de réponse échouées (avec la raison de l'échec).
- Dédup sémantique des requêtes de recherche (cosinus sur embeddings) pour ne jamais re-chercher deux fois la même chose — Laplace a déjà bge-m3 pour ça.

## Dépendances externes & limites

- Verrouillé sur l'écosystème Jina : Reader API (`r.jina.ai`) pour lire les pages, `s.jina.ai`/Brave/Serper pour chercher, embeddings/reranker Jina. Tout est remplaçable mais c'est du travail.
- LLM : Gemini par défaut, OpenAI possible — et un mode documenté pour LLM local OpenAI-compatible.
- Pas de persistance d'état entre sessions (in-memory).

## À réutiliser pour Laplace

- **L'architecture cible du module recherche** : boucle à actions typées + gating + gaps queue + budget + beast mode + évaluateur. C'est le meilleur modèle « agent » (vs le modèle « workflow » de dzhng) et il se marie naturellement avec une grammaire GBNF qui n'expose *que* les actions autorisées à l'étape courante (grammaire régénérée par étape).
- Le streaming `<think>` : Laplace peut streamer ses étapes de raisonnement vers le satellite (TXT:) sans polluer la réponse finale.
- Les 4 listes de mémoire de boucle sont un format compact qui borne naturellement le contexte.

## Liens à creuser (besoins)

- **Guide d'implémentation DeepSearch/DeepResearch de Han Xiao** (la vraie mine d'or du repo) : partie I <https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch>, partie II (sélection de snippets & ranking d'URLs) <https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch>.
- Démo hébergée du code exact : <https://search.jina.ai>.
- Besoin local : remplacer Reader/serp/rerank Jina par SearXNG + lecteur maison + bge-m3 (embeddings + rerank par cosinus).
