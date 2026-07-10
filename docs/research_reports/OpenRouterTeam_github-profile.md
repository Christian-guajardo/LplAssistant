# OpenRouterTeam (profil) — le routeur de modèles et sa boucle d'agent

**Source :** <https://github.com/OpenRouterTeam/.github> (profile/README.md)
**Type :** Profil d'organisation — passerelle API unifiée vers 340+ modèles / 90+ providers | **Analysé le :** 2026-07-10

## En une phrase

Une seule API OpenAI-compatible devant tous les modèles du marché, avec routage automatique prix/perf et fallbacks — pour Laplace c'est surtout : (1) la confirmation que **l'API OpenAI-compatible est LE standard d'interop**, (2) un banc d'essai cloud pour comparer des modèles avant de les télécharger, (3) une boucle d'agent de référence très lisible.

## Ce que propose l'écosystème

- **API unifiée** : change le `baseURL` et la clé, garde ton code OpenAI. Routage intelligent (`provider: { sort: "price" }`), fallbacks automatiques, pas de lock-in.
- **SDKs** : TypeScript (`@openrouter/sdk`), Python, Go.
- **`@openrouter/agent`** (typescript-agent) : mini-boucle d'agent où les outils déclarés (schéma Zod + `execute`) sont **auto-exécutés** en multi-tours avec streaming — l'archétype minimal « boucle tool-calling » à imiter en C++ (déclaration d'outil = nom + description + schéma + fonction ; la boucle appelle, exécute, réinjecte, jusqu'à réponse finale).
- Repos annexes : `tool-calling` (démo), `skills`, `ai-sdk-provider`, `awesome-openrouter`.

## Pertinence pour Laplace

- **Compatibilité gratuite** : si le moteur de Laplace parle OpenAI-compatible en interne (ce que fait `llama-server`), alors basculer une tâche ponctuelle vers un modèle cloud via OpenRouter = changer une URL. Utile en phase de dev pour A/B-tester (ex. Tongyi-DeepResearch-30B y est servi) sans rien installer.
- Le champ `provider.sort: price` illustre une idée transposable en local : **un routeur interne Laplace** qui choisit petit modèle / gros modèle / (option cloud) selon la tâche, le budget latence et la confidentialité.
- Attention vie privée : router vers OpenRouter = envoyer les prompts à un tiers. Pour Laplace, ce doit rester un mode debug/bench explicite, jamais un fallback silencieux (cohérent avec la doctrine 100 % local).

## Liens à creuser (besoins)

- <https://openrouter.ai/models> (catalogue + prix, utile pour la veille modèles) ; <https://openrouter.ai/rankings> (quels modèles les agents utilisent réellement).
- `typescript-agent` : <https://github.com/OpenRouterTeam/typescript-agent> — lire `callModel` comme spéc de notre boucle d'outils.
