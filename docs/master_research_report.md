# Rapport maître — Deep Research & agentique local pour Laplace

**Projet :** LplAssistant (Jarvis local, C++20, llama.cpp/whisper.cpp/pgvector)
**Date :** 2026-07-10 · **Sources :** 15 repos/liens analysés de première main (rapports détaillés dans [research_reports/](research_reports/))
**Rôle de ce document :** source de vérité pour construire le plan d'implémentation de l'outil d'agentique/deep research basé sur llama.cpp.

---

## 1. Ce que « Deep Research » veut dire en 2026

Tous les systèmes étudiés convergent vers la même boucle, au-delà des différences de framework :

> **Planifier → Chercher → Lire (en entier) → Réfléchir (nommer les lacunes) → Re-chercher → Synthétiser**, sous contrainte de **budget**, avec **état persistant**.

Ce qui sépare un gadget d'un vrai outil (leçon directe de ton POC : la collecte marchait, le rapport était creux) :

1. **La synthèse LLM itérative est le cœur** — l'agrégation de snippets ne produit pas de savoir. Gemini Deep Research lit les pages intégralement et itère pendant 5-8 min ; c'est la barre.
2. **La lecture complète des pages**, pas les extraits (ton POC tronquait à 360 chars : information détruite avant synthèse).
3. **Un budget explicite** (tokens, pas itérations) avec sortie forcée — le « Beast Mode » de Jina : 10 % du budget réservé pour contraindre une réponse finale.
4. **Un état sur disque** qui survit aux crashs (star-hengxing : machine à états JSON à transitions monotones + `next_phase()`).

## 2. Les trois archétypes d'architecture (et notre choix)

| Archétype | Représentant | Forces | Faiblesses |
|---|---|---|---|
| **Workflow récursif** | dzhng (breadth/depth, learnings accumulés) | ~300 lignes, coût contrôlé (largeur ÷2 par niveau), trivial à porter en C++ | Pas de décision fine, pas de resume |
| **Agent à machine à états** | jina node-DeepResearch (search/visit/reflect/answer + gating + gaps queue + budget + évaluateur) | S'adapte à la question, anti-boucle par construction | Plus de prompts à régler |
| **Superviseur multi-agents** | open_deep_research, gpt-researcher, IterResearch « Heavy » (Alibaba) | Parallélisme (latence ÷5), contextes isolés par chercheur | Coûteux ; cohérence à re-fabriquer à la rédaction |

**Choix pour Laplace : l'agent à machine à états (jina) comme cœur, avec fan-out parallèle borné (gpt-researcher) pour les sous-questions, et deux vitesses (Alibaba) : mode ReAct rapide / mode Heavy explicite.** Le workflow dzhng sert de v0 de validation du pipeline.

Décisions transverses volées aux meilleurs :
- **Clarifier avant de chercher** (dzhng/u14app) : 2-3 questions à l'utilisateur, puis plan annoncé et validé (en vocal : « voilà mon plan, je te préviens quand c'est prêt » — fire-and-forget hérité de ton POC).
- **Réflexion explicite** (local-deep-researcher) : un état de la boucle écrit noir sur blanc la *lacune de connaissance* avant la requête suivante.
- **Compression aux frontières** (open_deep_research + honey) : jamais de source brute entre étages, uniquement des findings compressés (format ESON/JSON columnar).
- **Évaluateur avant réponse** (jina) : définitude, attribution, fraîcheur ; réponse rejetée ⇒ la boucle repart avec la raison de l'échec.
- **Quality gates déterministes** (star-hengxing) : vérif HTTP des URLs citées (HEAD→GET, offline-aware), format de bibliographie, `.meta.json` obligatoire (modèle, tokens, sources, durée) sinon publication refusée.
- **Rédaction progressive par sections** (star-hengxing) : jamais un rapport long en un seul appel (troncature de sortie garantie sur les longs formats).

## 3. Le moteur : llama.cpp fait déjà 80 % du runtime

Découverte centrale de l'analyse (voir [research_reports/ggml-org_llama.cpp.md](research_reports/ggml-org_llama.cpp.md)) : `llama-server` n'est pas juste un endpoint de complétion —

- **API OpenAI ET Anthropic-compatibles**, embeddings **et reranking** servis par le même daemon (bge-m3 peut y vivre ; le reranker externe devient inutile).
- **Slots parallèles** (`-np`) + continuous batching : N chercheurs simultanés sur un seul modèle en RAM — la réponse au mur « Ollama sérialise » de ton POC.
- **`--cache-reuse`** : réutilisation KV par shifting, généralisation serveur de notre cache de préfixe maison.
- **GBNF/JSON-Schema par requête** : le JSON de la machine à états est *physiquement* garanti. L'échec des petits modèles sur le JSON strict (documenté par local-deep-researcher) disparaît. Idée clé : **grammaire régénérée à chaque étape n'exposant que les actions autorisées** (le gating de Jina, compilé dans le sampler).
- **Function calling universel** (`--jinja`) avec format natif Qwen 2.5 (notre famille de modèles) et fallback générique.
- **Décodage spéculatif** (`--spec-draft-*`, draft 0.5-1.5B devant le 8-14B) + lookup decoding ; **budget de raisonnement natif** (`--reasoning-budget`).
- Mode `--offline`, quantif du KV, `--mlock`, affinités CPU, offload MoE (`--cpu-moe`).

**Conséquence d'architecture :** basculer vers un split process — daemon `llama-server` + Laplace client HTTP localhost. Bénéfices : batching multi-locuteurs, tool-calls parsés côté serveur, grammaires par requête, et disparition du conflit de link ggml whisper/llama (notre bug historique qui impose le binaire STT séparé). Le lien direct libllama peut survivre pour le chat courant si le banc d'essai le justifie.

**Modèles :** garder qwen2.5-1.5b pour le dialogue instantané ; viser un **8-14B (Qwen 2.5/3.x)** pour le raisonnement de recherche ; évaluer **Tongyi-DeepResearch-30B-A3B** (MoE : coût d'un ~3B, capacité d'un 30B, 128K contexte, *entraîné pour* cette tâche — testable via OpenRouter avant téléchargement). Rôles séparés (open_deep_research) : petit modèle pour résumer les pages, grand pour décider/rédiger.

## 4. Ingénierie du contexte (la vraie difficulté du local)

Les patterns d'opencode (`CONTEXT.md`) se mappent un-pour-un sur le KV-cache :

- **Context Epochs** : contexte système de base immuable par époque = **préfixe KV stable**, décodé une fois, persistable sur disque ; les changements (date, domotique, nouveau locuteur) deviennent des *messages système chronologiques*, jamais une invalidation du préfixe ; la **compaction ouvre une nouvelle époque**.
- **Safe boundaries** : les changements de contexte sont intégrés paresseusement juste avant un appel modèle — jamais poussés en asynchrone au milieu d'un tour.
- **Inbox / promotion des prompts** : formalisation exacte de notre barge-in vocal multi-locuteurs (steer/followup/collect/interrupt chez OpenClaw).
- **Sorties d'outils bornées** : projection limitée dans l'historique + fichier intégral géré à côté.

Complété par la **compaction OpenClaw** : résumer les vieux tours en préservant les paires tool-call/résultat, déclenchée près de la limite *ou* sur erreur d'overflow (retry après compaction), avec **rappel préalable à l'agent de sauver ses notes en mémoire**. Et ses **guardrails anti-boucle** : triplet (outil, args, résultat) répété après compaction ⇒ abort.

**Économie de tokens (honey-for-devs, vérifié au banc)** : trois protocoles à intégrer nativement —
- **CCR (Compress-Cache-Retrieve)** : toute page scrapée va intégralement en cache disque ; le prompt ne reçoit qu'une vue écrémée + un hash ; l'agent a un outil `retrieve` pour les sections précises. *Réponse définitive au mur des 360 chars.*
- **ESON / JSON compact** pour tous les handoffs inter-étapes (~-50 % lossless).
- **Style « réponse d'abord »** dans le system prompt : moins de tokens générés = le TTS démarre plus tôt.

## 5. Mémoire long terme

- Notre **Composite Retrieval** (SQL filtre, vecteurs rapprochent) est validé par l'état de l'art ; le GraphRAG reste rejeté comme *store* — le graphe n'est pertinent que comme **artefact de sortie** (u14app, WikiGraphs).
- **DeepMind Continual Learning** (arXiv 2105.13327) fournit l'évolution théorique : mémoire k-NN task-free/online où la sortie est **pondérée par la distance** (pas de top-k sec), et où l'on peut stocker des *comportements* (préférences, profils par locuteur) et pas seulement des faits — apprentissage localisé, zéro oubli catastrophique, zéro fine-tuning.
- **OpenClaw** apporte l'étage lisible : `MEMORY.md` durable injecté au bootstrap + notes datées `memory/YYYY-MM-DD.md` indexées mais non injectées, **distillation périodique** du quotidien vers le durable — et **Dreaming** : consolidation nocturne en 3 phases (trier → réfléchir → promouvoir), parfaite pour un serveur domestique idle la nuit.

**Synthèse Laplace :** pgvector reste le store ; on y superpose le duo markdown durable/quotidien + un job nocturne de distillation, et on passe le retrieval en pondération par distance.

## 6. La stack de récupération 100 % locale (le chantier n°1)

Toutes les références dépendent d'APIs commerciales (Firecrawl, Tavily, Serper, Jina Reader, Dashscope). La boucle besoin→solution donne :

| Besoin | Solution locale retenue | Source de la préconisation |
|---|---|---|
| Recherche web | **SearXNG** en Docker (méta-moteur, sans clé, air-gap) ; DuckDuckGo en secours | local-deep-researcher, u14app, OpenClaw |
| Lecture de pages (JS, Cloudflare) | **Lecteur HTML→markdown maison** (extraction `<body>`, strip nav/footer) + headless browser en 2e rideau pour les sites durs ; dégradation classée par source | POC masterlaplace (diagnostics), Jina (guide) |
| Sources structurées API-first | **Porter la table des 10 providers du POC** (Wikipedia, StackExchange, GitHub ×3, OpenAlex `is_oa:true`, arXiv, Reddit, domain docs) — endpoints et pièges déjà documentés | POC masterlaplace |
| PDF (papers, manuels) | Extracteur PDF local (poppler/pdfium) + CCR pour les gros documents | besoin récurrent (Alibaba Tech_Report, arXiv, manuels Intel) |
| Ranking | Composite du POC (poids provider + fraîcheur + overlap) **+ score sémantique bge-m3** (règle le bruit lexical type « 2026 » → issues #2026) + rerank via llama-server | POC + jina |
| Sandbox d'exécution | Interpréteur Python/shell sandboxé (plus tard) | Alibaba (SandboxFusion), opencode (bash) |

Anti-biais (gpt-researcher) : minimum de sources et de diversité de domaines imposés par les quality gates.

## 7. Leçons durement acquises de ton POC (exigences non négociables)

1. La synthèse est le produit, la collecte n'est que la matière première.
2. Vrais slots parallèles + timeouts calibrés sur le débit *mesuré* (jamais 8×15 s sur un runtime séquentiel).
3. Ne jamais tronquer une source : CCR.
4. Matching sémantique en plus du lexical.
5. Diagnostics par provider + section Limitations dans chaque rapport (la transparence du POC était sa meilleure qualité — la garder).
6. Data Sink : les rapports sont des fichiers markdown datés, jamais du contexte de chat.
7. Tokens optionnels par provider (GitHub 60→5 000 req/h).
8. Contrainte d'environnement : rien d'exotique à télécharger sur la machine pro (OpenClaw/Caveman interdits — analyse par docs seulement).

## 8. Architecture cible du module `deep_research` de Laplace

```
┌──────────────────────── laplace (C++20) ────────────────────────┐
│ REPL/vocal → clarification → plan (validé) → fire-and-forget    │
│                                                                 │
│  Machine à états (checkpoint JSON, transitions monotones)       │
│   PLAN → [fan-out N chercheurs // slots llama-server]           │
│           chaque chercheur : search → read(CCR) → reflect       │
│           (gaps) → … budget tokens, beast mode à 90 %           │
│   → compress (ESON) → evaluate (gates URL/citations/meta)       │
│   → write (par sections) → rapport .md daté + notification      │
│                                                                 │
│  Outils locaux : searxng · http_reader · providers API ·        │
│  pdf_reader · pgvector(retrieve/memorize) · retrieve(CCR)       │
└───────────────┬─────────────────────────────────────────────────┘
                │ HTTP localhost (OpenAI-compat, GBNF par requête)
        ┌───────▼────────┐   ┌──────────────┐   ┌───────────────┐
        │ llama-server    │   │ SearXNG      │   │ PostgreSQL    │
        │ 8-14B + draft   │   │ (Docker)     │   │ + pgvector    │
        │ + embed + rerank│   └──────────────┘   └───────────────┘
        └─────────────────┘
```

Dual-retrieval systématique (u14app) : chaque requête frappe pgvector ET le web. Le pipeline expose sa progression sur le protocole UDP existant (équivalent du SSE), et pourra être façadé en MCP plus tard (u14app, gpt-researcher).

## 9. Feuille de route proposée

1. **Fondation moteur** : `llama-server` en service (modèle 8-14B + draft), client HTTP C++ (libcurl), grammaire GBNF de la machine à états. Critère : 100 sorties JSON valides consécutives.
2. **v0 workflow (dzhng porté)** : SearXNG + lecteur HTML minimal + boucle breadth/depth + rapport md. Critère : rapport sourcé de 3+ pages en < 10 min sur le mini-banc (10-20 questions maison — à constituer d'abord, leçon open_deep_research).
3. **v1 agent (jina)** : actions typées, gaps queue, budget/beast mode, évaluateur, checkpoint/resume (star-hengxing), quality gates.
4. **v2 échelle** : fan-out sur slots, providers API du POC portés, CCR + ESON, ranking sémantique, mode Heavy.
5. **v3 intégration Laplace** : dual-retrieval pgvector, fire-and-forget vocal, mémoire markdown + Dreaming nocturne, Context Epochs dans le runtime de dialogue.

## 10. Besoins ouverts (boucle besoin→solution, itération suivante)

- **Lecteur PDF local** : Tech_Report.pdf (Alibaba), papiers arXiv (2105.13327, 2107.09556, 2410.04343, 2402.14207), guide CNRS OpenAlex — non exploitables aujourd'hui par notre outillage.
- **Guides de conception à lire** (fetch web simple suffit) : Jina « A Practical Guide to Implementing DeepSearch/DeepResearch » parties I & II ; blog « bitter lesson » de Lance Martin ; blog fondateur gpt-researcher.
- **Bancs** : GGUF de Tongyi-30B-A3B à évaluer ; gain réel du décodage spéculatif sur la machine cible ; ratios ESON/CCR sur un 8-14B.
- **Mise à jour whisper.cpp** 1.6.2 → 1.8.x.
- Sites anti-bot durs : décider du 2e rideau (headless browser local) après mesure du taux d'échec réel du lecteur simple.

---

*Rapports détaillés par source : [research_reports/](research_reports/) — un fichier par repo/lien, tous relus depuis le code et les docs d'origine.*
