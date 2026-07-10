# OpenClaw — l'assistant personnel agentique (analysé sur docs uniquement)

**Source :** <https://github.com/openclaw/openclaw> · docs <https://docs.openclaw.ai>
**Type :** Assistant personnel Node.js multi-canaux (gateway + agents), MIT | **Analysé le :** 2026-07-10
**⚠️ Contrainte** : outil interdit par la sécurité de l'entreprise (flag CrowdStrike). Analyse faite sur la documentation seulement ; le clone temporaire a été supprimé, rien n'a été installé ni exécuté. Ne jamais le télécharger sur cette machine.

## En une phrase

Le projet le plus proche de la vision Laplace côté *produit* (assistant personnel toujours-on, mono-utilisateur, sur tes machines, tes canaux, tes règles — lignée Warelay→Clawdbot→Moltbot→OpenClaw), avec des solutions éprouvées pour la mémoire, la compaction et les boucles d'outils — mais cloud-model-first et non optimisé bare-metal.

## Architecture

- **Gateway** (daemon launchd/systemd) = plan de contrôle : canaux (WhatsApp, Telegram, Slack, Discord, Signal… ~20), routage vers des **agents** isolés (workspace + store de sessions + auth par agent, liés aux canaux par *bindings*).
- **Boucle agent** sérialisée par session : RPC `agent` → file par-session (+ lane globale) → résolution modèle/skills → `runEmbeddedAgent` → streaming (tool/assistant/lifecycle) → persistance transcript sous **verrou d'écriture fichier inter-processus**. Modes de file : steer/followup/collect/interrupt (≈ notre barge-in).
- **Hooks** à chaque couture : `before_prompt_build`, `before_tool_call`/`after_tool_call`, `tool_result_persist`, `before/after_compaction`, `session_start/end`… Un runtime extensible sans toucher au cœur.
- **Harness pluggable** : le runtime intégré (`openclaw`) coexiste avec des harness externes par provider — l'agent est une interface, pas une implémentation.

## Les systèmes remarquables

- **Mémoire = fichiers Markdown, zéro état caché** : `MEMORY.md` (durable, injecté au bootstrap, tronqué au-delà d'un budget), `memory/YYYY-MM-DD.md` (notes de travail, indexées pour `memory_search`/`memory_get` mais pas injectées), distillation périodique du quotidien vers le durable. Philosophie « le modèle ne se souvient que de ce qui est écrit sur disque ».
- **Dreaming** (opt-in) : consolidation nocturne en 3 phases — *light* (trier/stager les signaux récents), *REM* (thèmes et réflexions), *deep* (scorer et **promouvoir** vers MEMORY.md) — avec journal humain `DREAMS.md`. Idée magnifique pour un assistant domestique : consolider pendant que la maison dort.
- **Compaction** : résumé des tours anciens en préservant les paires tool-call/tool-result à la coupe, déclenchée près de la limite OU sur erreur d'overflow (dizaines de patterns d'erreurs providers reconnus, retry après compaction) ; **rappel automatique à l'agent de sauver ses notes en mémoire avant de compacter**.
- **Guardrails de boucles d'outils** : détection de motifs répétés dans l'historique d'appels + garde post-compaction qui avorte si le même triplet (tool, args, result) revient après un retry d'overflow — le cycle « overflow→compaction→même boucle » est cassé explicitement.
- **Recherche web par plugins** : Brave, DuckDuckGo (sans clé), **SearXNG self-host**, Perplexity, Exa… + `web_fetch` local. Confirme SearXNG comme choix local de référence.

## Limites (pour notre usage)

- Node.js, cloud-first (OpenAI/Anthropic en OAuth), pas de voix locale native au cœur, très grosse surface (canaux, apps compagnes) — l'anti-thèse de « bare metal minimal ».
- Interdit sur la machine de dev de l'utilisateur : source d'inspiration, jamais une dépendance.

## À réutiliser pour Laplace

- Le **modèle mémoire en Markdown à deux étages + distillation** (MEMORY.md ↔ notes datées) par-dessus notre pgvector : lisible, versionnable, auditable — et le LLM-wiki du rapport Jarvis, concrétisé.
- **Dreaming** comme job cron nocturne de Laplace (llama-server idle la nuit) : promotion mémoire + résumés de la journée par locuteur.
- Le rappel pré-compaction (« sauve tes notes ») et la préservation des paires tool-call/result.
- Les guardrails anti-boucle (triplet répété ⇒ abort) dans notre machine à états.
- La sérialisation par session + verrou fichier : notre serveur UDP multi-locuteurs a le même problème, leur solution est simple et robuste.

## Liens à creuser (besoins)

- Docs concepts (lecture web only) : <https://docs.openclaw.ai/concepts/agent-loop>, `/concepts/compaction`, `/concepts/memory`, `/concepts/dreaming`, `/tools/searxng-search`.
