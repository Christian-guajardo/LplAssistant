# masterlaplace/deep-research — ton POC VS Code : les leçons durement acquises

**Source :** dossier local `repos_storage/masterlaplace_deep-research/` (extension VS Code + 11 documents historiques, ~4 400 lignes de md **toutes lues**)
**Type :** Extension TypeScript (Language Model Tool Copilot Chat, ~2 800 LOC) + discussions Gemini/Copilot de mars→juin 2026 | **Analysé le :** 2026-07-10

## En une phrase

Un backend de collecte multi-sources déjà très abouti (10 providers, ranking, map-reduce Ollama) dont la vraie valeur pour la suite est double : **le pipeline de providers** à porter, et **la liste des murs rencontrés** — chacun devient une exigence de conception de l'outil Laplace.

## Le pipeline construit (v0.0.1, « GO for production »)

- **10 providers parallèles** (localBackend.ts, 1 200+ LOC) : Wikipedia (API MediaWiki), StackOverflow (StackExchange API), GitHub Repos (stars) / Issues (updated — excellent pour le troubleshooting) / Docs (README+releases via API), Domain Docs (DDG Lite + `site:khronos.org|nvidia.com|MDN`), Web général (DDG Lite), Reddit JSON public, OpenAlex (`is_oa:true`), arXiv (API XML) + fallback web si sources primaires < 50 % de la cible.
- **Ranking composite** (`weight*0.7 + lengthScore*0.3 + recencyBoost*0.2`, puis `+ overlap lexical*0.6`) : poids par provider, boost fraîcheur, overlap requête/contenu, **rééquilibrage par catégories** (cap sur les papers hors intent académique), diversité de providers imposée dans le top-N.
- **Enrichissement** : proxy Jina Reader (`r.jina.ai`) pour pages dynamiques, détection PDF (dont conversion arXiv abs→pdf), extraits tronqués à 360 chars/source.
- **Synthèse map-reduce Ollama** : map = 2-3 findings par source, reduce = rapport structuré (~300 mots) ; auto-détection des modèles (`GET /api/tags`, choix du plus gros), QuickPick de sélection, fallback statistique si Ollama absent.
- **Orchestration asynchrone** : fire-and-forget (invoke retourne un Job ID immédiatement), jobs persistés (workspaceState), annulation, re-run, rapport écrit dans `.github/research_reports/` (« Data Sink » — jamais réinjecter le rapport dans le contexte du chat).

## Les murs rencontrés (chaque bug = une exigence pour Laplace)

1. **Résumé hardcodé découvert tardivement** → un « rapport » sans synthèse LLM ne vaut rien ; la collecte n'était pas le problème, la synthèse si. *Exigence : la synthèse LLM est le cœur, pas une option.*
2. **Ollama sérialise les requêtes** : 8 appels map parallèles × timeout 15 s = tout timeout ; correctif = séquentiel à 30 s/source. *Exigence : `llama-server` avec vrais slots parallèles + timeouts calibrés sur le débit mesuré.*
3. **Troncature à 360 chars/source** = l'information détruite avant la synthèse. *Exigence : protocole CCR (cache intégral disque + vue compressée + retrieve).*
4. **Bruit lexical** : « 2026 » dans la requête matchait les issues GitHub #2026. *Exigence : matching sémantique (embeddings bge-m3) en plus du lexical.*
5. **Reddit 403, DDG Lite parsing fragile, Cloudflare** : les sources HTML sans API cassent en premier. *Exigence : dégradation par source classée (ok/empty/error) + diagnostics dans le rapport — déjà bien fait dans le POC, à conserver.*
6. **GitHub 60 req/h sans token** (5 000 avec). *Exigence : gestion de tokens optionnels par provider.*
7. Le comparatif honnête avec Gemini Deep Research : notre agrégation ≈ collecte OK, mais Gemini **lit les pages en entier et itère** (planning → search → read → reflect → synthèse, 5-8 min). *Exigence : boucle itérative avec lecture complète, pas du one-shot sur snippets.*

## Les documents historiques (contexte stratégique, tous lus)

- `discussion-mars-2026.md` + `Integrer_la_recherche_Gemini_dans_VS_Code.md` : rapport Gemini détaillant l'API Interactions (background:true, polling, `previous_interaction_id`), le choix Language Model Tool vs MCP vs Chat Participant, le pattern Data Sink, SecretStorage. Le pattern **fire-and-forget + polling + notification** reste la bonne UX pour les recherches longues, y compris vocales (« je te préviens quand c'est prêt »).
- `Roadmap & Prompt Copilot.md` : méthode « Master Prompt puis étapes une à une » pour faire coder un agent — méthodologie réutilisable.
- `discussion-juin-2026.md` : la genèse de la vision — modèle moyen local + ingénierie > gros modèle cloud générique ; le savoir ouvert (OpenAlex, arXiv, wikis) accessible aux IA locales ; MCP comme chaînon ; et l'épisode sécurité entreprise (Caveman/OpenClaw bloqués par CrowdStrike, critique des « optimiseurs de tokens » mal conçus). Fixe deux contraintes permanentes : **rien d'exotique à télécharger sur la machine pro**, et **la qualité du backend prime sur le vibe-coding**.
- `discussion_copilot.md` : journal complet du build par l'agent Copilot (plan en 8 phases, V1→V5 du backend) — et la préférence explicite : *« un backend de qualité plutôt que de laisser l'IA tout gérer »*.
- Les 3 rapports générés dans `.github/research_reports/` : preuve du format de sortie (executive summary, top findings attribués, sources, diagnostics, limitations).

## À réutiliser pour Laplace

- **Porter la table des 10 providers en C++** (libcurl + parsing JSON) : c'est déjà la bonne liste de sources dev/science, avec leurs endpoints exacts et leurs pièges documentés.
- Le ranking composite + rééquilibrage par catégories (fonctions courtes, portage direct), en y ajoutant le score sémantique bge-m3.
- Les diagnostics par provider et la section Limitations dans chaque rapport (transparence = confiance).
- Le Data Sink : les rapports de Laplace sont des fichiers markdown datés, jamais du contexte de chat.
- Remplacer les deux dépendances non-locales restantes : Jina Reader → lecteur HTML local, DDG Lite scraping → SearXNG.
