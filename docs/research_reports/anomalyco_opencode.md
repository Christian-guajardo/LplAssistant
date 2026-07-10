# anomalyco/opencode — l'ingénierie de session d'un agent de production

**Source :** <https://github.com/anomalyco/opencode>
**Type :** Agent de code open source (TypeScript/Bun, monorepo ~30 packages : core, tui, desktop, server, sdk…) | **Analysé le :** 2026-07-10 (README, `CONTEXT.md` intégral, `AGENTS.md`, inventaire `packages/core/src/tool/`)

## En une phrase

Au-delà du produit (CLI/TUI de coding agent), la valeur pour nous est dans `CONTEXT.md` : le vocabulaire le plus rigoureux que j'aie vu pour la **gestion du contexte d'un agent long-vécu** — et il se mappe presque un-pour-un sur les contraintes KV-cache de llama.cpp.

## La boîte à outils de l'agent (packages/core/src/tool/)

Registre minimal et suffisant : `bash`, `read`, `write`, `edit`, `apply-patch`, `glob`, `grep` (ripgrep vendorisé), `webfetch`, `websearch`, `todowrite`, `question`, `skill`. Douze outils. Un agent de code de production n'a pas besoin de plus — étalon de sobriété pour l'ensemble d'outils de Laplace.

## Les concepts de CONTEXT.md à voler

- **Context Source** : chaque fait contextuel (date, instructions projet, skills dispo…) est un producteur typé avec clé stable, codec JSON, et **renderers purs** baseline/update/removal. Le registre les évalue en parallèle et les compose en ordre déterministe.
- **Context Epoch / Baseline System Context** : le contexte système rendu en début d'époque est **immuable** et sert de « provider-cache baseline » — persistant sur disque, réutilisé verbatim après redémarrage. *C'est exactement un préfixe KV-cache stable.* Une époque se termine à la compaction, qui replie tout dans une nouvelle baseline.
- **Mid-Conversation System Message** : quand une Context Source change en cours de route, on n'invalide PAS la baseline — on injecte chronologiquement un message système durable « nouvel état effectif de X ». Le préfixe caché reste intact.
- **Safe Provider-Turn Boundary** : les changements de contexte sont échantillonnés **paresseusement** juste avant chaque appel modèle, jamais poussés en asynchrone. Ordre garanti : input promu → résultats d'outils réglés → message de contexte combiné.
- **Prompt Promotion / Admitted Prompt** : les inputs utilisateur entrent dans une inbox durable et ne deviennent visibles au modèle qu'à une frontière sûre — le « barge-in » de Laplace formalisé (steering prompts promus pendant le drain, prompts en file promus à l'idle).
- **Model Tool Output / Managed Tool Output File** : la sortie d'un outil est une **projection bornée** ; l'intégralité part dans un fichier temporaire géré. (= CCR de honey, version runtime.)
- **Session Drain** : l'exécution locale « jusqu'à plus de continuation » n'a *pas* d'identité durable — la reprise après crash se raisonne depuis les prompts/l'historique/l'état des outils, pas depuis un objet « run ».

## Autres enseignements

- Style guide `AGENTS.md` : tout dans une fonction sauf réutilisable, pas d'helpers à usage unique, inliner les valeurs à usage unique — hygiène directement applicable à notre C++.
- Architecture en couches strictes (Schema → Core/Protocol → Server ; le client ne dépend jamais de Core) — modèle pour séparer `laplace-core` / `laplace-server` / clients satellites.
- Multi-frontends (TUI, desktop, web, Slack) sur un seul serveur : la façade HTTP/SDK est ce qui rend ça possible.

## Dépendances externes & limites

- Bun/TypeScript, cloud opt-in (console, share) ; le produit vise le coding, pas l'assistant personnel.
- Le runtime lui-même ne nous sert pas ; on transpose les invariants, pas le code.

## À réutiliser pour Laplace

- **Implémenter les Context Epochs en C++** : baseline système immuable = préfixe KV décodé une fois (voire sauvé sur disque via l'API de state llama.cpp) ; changements → messages système chronologiques ; compaction = nouvelle époque + nouveau préfixe.
- La discipline « Safe Boundary » pour intégrer les événements maison (domotique, nouveaux locuteurs) sans casser le cache ni la cohérence.
- Le pattern inbox/promotion pour unifier notre gestion du barge-in vocal multi-locuteurs.
- La projection bornée des sorties d'outils + fichier géré (avec `retrieve`).

## Liens à creuser (besoins)

- `CONTEXT.md` du repo (déjà cloné) à relire au moment du design du runtime.
- Docs produit : <https://opencode.ai/docs> ; specs internes dans `specs/`.
