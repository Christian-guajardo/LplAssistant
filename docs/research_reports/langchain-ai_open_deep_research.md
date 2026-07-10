# langchain-ai/open_deep_research — l'agent superviseur multi-modèles

**Source :** <https://github.com/langchain-ai/open_deep_research>
**Type :** Agent LangGraph Python (production, #6 du Deep Research Bench, score RACE 0.4344) | **Analysé le :** 2026-07-10

## En une phrase

La version « industrielle » du deep research open source : un superviseur qui délègue des sous-sujets à des chercheurs parallèles, avec **un modèle différent par rôle** (résumé / recherche / compression / rédaction) et une leçon d'histoire précieuse dans `src/legacy/`.

## Architecture & pipeline

Trois phases :

1. **Scoping** : clarification avec l'utilisateur puis écriture d'un brief de recherche.
2. **Research** : un **superviseur** décompose le brief et lance des **sous-agents chercheurs en parallèle** (chacun sur un sous-sujet, avec ses propres appels de recherche itératifs) ; chaque chercheur **compresse** ses trouvailles avant remontée.
3. **Writing** : un modèle dédié rédige le rapport final à partir des findings compressés.

Quatre rôles de modèles configurables séparément : `summarization` (défaut gpt-4.1-mini — petit et rapide), `research` (gpt-4.1), `compression` (gpt-4.1), `final_report` (gpt-4.1). Recherche : Tavily par défaut, moteurs alternatifs, et **serveurs MCP** comme sources d'outils.

## Le contexte historique (dossier `src/legacy/` + blog « bitter lesson »)

Le repo contient ses deux anciennes architectures, abandonnées :
- **Plan-and-Execute** (`graph.py`) : plan de sections validé par un humain, exécution séquentielle section par section. Qualité prévisible, mais lent et rigide.
- **Multi-agent supervisor** (`multi_agent.py`) : plus rapide (parallélisme), mais rapports décousus.

La version actuelle est une synthèse : superviseur + parallélisme **mais** rédaction finale en un seul passage global (cohérence), et compression agressive entre les étages. Leur blog (rlancemartin, « bitter lesson », 30/07/2025) explique la migration : moins de structure imposée, plus de capacité déléguée au modèle, à mesure que les modèles s'améliorent.

## Techniques à retenir

- **Un modèle par rôle** : en local, ça se traduit par : petit modèle (1-3B) pour résumer les pages, modèle principal (8-14B) pour raisonner/décider, et éventuellement le même en température basse pour rédiger. `llama-server` sait servir plusieurs modèles ou on lance 2 instances.
- **Compression entre étages** : ne jamais faire remonter les sources brutes au superviseur — uniquement des findings compressés. C'est ce qui borne le contexte du superviseur.
- **L'évaluation systématique** : Deep Research Bench (100 tâches PhD-level, score RACE par LLM-juge). Pour Laplace : se constituer un mini-banc de 10-20 questions pour mesurer chaque itération.
- Isolation du contexte par sous-agent : chaque chercheur a son propre historique — pas de pollution croisée. En llama.cpp : un slot / une séquence KV par chercheur.

## Dépendances externes & limites

- LangGraph + écosystème LangChain (init_chat_model, Studio UI) — la valeur est dans le *design*, pas dans le framework.
- Tavily par défaut (API payante) ; MCP supporté nativement pour brancher d'autres outils.
- Coût élevé du mode superviseur (beaucoup d'appels parallèles).

## À réutiliser pour Laplace

- La séparation scoping → research (parallèle, compressé) → writing (global) comme forme générale de notre pipeline, avec le nombre de chercheurs parallèles mappé sur les slots de `llama-server`.
- Le principe « compression aux frontières » entre chaque étage (voir aussi honey-for-devs : ESON/handoffs).
- Le mini-banc d'évaluation avant d'optimiser quoi que ce soit.

## Liens à creuser (besoins)

- Deep Research Bench leaderboard : <https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard>.
- Blog « bitter lesson » de Lance Martin : <https://rlancemartin.github.io/2025/07/30/bitter_lesson/> — lecture stratégique recommandée.
- Cours gratuit LangChain « Deep Research from scratch » : <https://github.com/langchain-ai/deep_research_from_scratch>.
