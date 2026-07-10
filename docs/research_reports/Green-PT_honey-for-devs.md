# Green-PT/honey-for-devs — l'économie de tokens comme discipline

**Source :** <https://github.com/Green-PT/honey-for-devs>
**Type :** Skill multi-outils (Claude Code, Copilot, Cursor, Cline, OpenClaw…) + formats + benchmarks reproductibles | **Analysé le :** 2026-07-10 (README, `eso/SPEC.md`, `agents/hive-*.md`, `bench/` vérifiés dans le repo)

## En une phrase

« Write less code and say less about it » : trois leviers indépendants (moins de code, moins de prose, handoffs agent-à-agent denses) qui coupent ~50 % des tokens de sortie **sans perte de correction**, prouvé par un banc de 23 tâches jugé par un panel de 4 modèles.

## Les trois leviers

1. **Less code** : échelle YAGNI — (le truc doit-il exister ? → stdlib → natif du langage → dépendance existante → une ligne → bloc minimal), s'arrêter au premier barreau qui marche.
2. **Less prose** : supprimer wind-up, hedging, narration du code. Réponse d'abord.
3. **Handoffs denses (Lever 3)** : quand le lecteur est un *agent* et pas un humain, sortie en JSON compact/columnar ou **ESON** — ne s'applique JAMAIS aux réponses utilisateur.

Garde-fous : validation d'entrée, gestion d'erreurs, auth, secrets, migrations, deletes ne sont **jamais** compressés (« Lazy ≠ broken »). Intensité auto (lite/full/ultra) choisie par réflexe, sans dépenser de tokens de raisonnement à décider.

## Les mécanismes vérifiés dans le repo

- **ESON** (`eso/SPEC.md`, implémentation JS + tests) : notation structurée schema-first — clés déclarées une fois, comptes de lignes déclarés (détection de troncature), cellules compatibles JSON. ~-50 % sur les handoffs, récupération 100 % lossless là où Caveman/Ponytail perdent (67 %/50 % de recovery au bench).
- **CCR** (`eso/ccr.js`, skill `honey-ccr`) : **Compress-Cache-Retrieve** pour les sorties massives (logs, scrapes) — déduplique/échantillonne, met l'original en cache disque avec un hash, donne à l'agent la vue compressée + un outil `retrieve(hash)` pour ravoir l'original au besoin. **L'information n'est jamais détruite, juste déplacée hors contexte.**
- **PX** (`bench/px/`) : rendre des fichiers texte denses en PNG pour les lire via un modèle vision — les tokens image croissent avec les pixels, pas les caractères (jusqu'à -85 % en lecture). Nécessite un modèle multimodal ; pertinence future pour Laplace (Qwen-VL local).
- **Hive** (`agents/hive-scout|builder|reviewer.md`) : sous-agents **read-only Haiku-class** qui retournent des handoffs compacts adressés par `id` stable avec un compteur `n` comme checksum — l'orchestrateur ne dépense pas son contexte à lire des fichiers.
- **Leçon de banc** : la version « rule » légère renvoyée à chaque tour bat le gros SKILL.md re-injecté (le prompt d'optimisation trop lourd coûte plus qu'il n'économise) — même critique que celle des collègues de l'utilisateur sur Caveman.

## Dépendances externes & limites

- Conçu pour des harnais existants (skills/rules), pas une bibliothèque runtime — on transpose les *protocoles*, pas le code.
- PX exige un modèle vision et un rendu d'images ; à garder pour plus tard.
- Les benchs sont sur Opus/GPT : les ratios exacts sur un 8-14B local restent à mesurer.

## À réutiliser pour Laplace

- **CCR est le protocole de scraping de Laplace** : page web → cache disque intégral → vue « écrémée » bornée dans le prompt → outil `retrieve` pour les sections précises. Résout à la fois le contexte court des modèles locaux et la perte d'info par troncature (le POC masterlaplace tronquait à 360 chars : exactement l'erreur que CCR évite).
- **ESON (ou JSON columnar) pour tous les états internes** de la machine à états et les handoffs planner→chercheurs→rédacteur.
- Le style « réponse d'abord, zéro remplissage » dans le system prompt de Laplace : moins de tokens générés = latence vocale réduite (le TTS commence plus tôt).
- Le pattern hive : nos sous-tâches de lecture/résumé sur le *petit* modèle, l'orchestration sur le grand.

## Liens à creuser (besoins)

- `eso/SPEC.md` (spec ESON complète, dans le repo cloné) ; `bench/results/cross-provider.md`.
- Les ancêtres cités : Ponytail (<https://github.com/DietrichGebert/ponytail>), Caveman (<https://github.com/JuliusBrussee/caveman>) — contexte, pas à installer (bloqués par la sécu entreprise).
