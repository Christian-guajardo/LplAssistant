# langchain-ai/local-deep-researcher — la boucle réflexive 100 % locale

**Source :** <https://github.com/langchain-ai/local-deep-researcher>
**Type :** Graphe LangGraph Python (module `ollama_deep_researcher`) | **Analysé le :** 2026-07-10 (lecture de `prompts.py`, structure `graph.py`/`state.py`)

## En une phrase

Le seul projet LangChain pensé *d'abord* pour les LLM locaux (Ollama/LMStudio) : une boucle courte requête → recherche → résumé glissant → **réflexion explicite sur les lacunes** → nouvelle requête, répétée N fois.

## Architecture & pipeline

Graphe à 5 nœuds, inspiré d'IterDRAG :

1. **`generate_query`** : le LLM produit UNE requête web ciblée (JSON `{query, rationale}`), date du jour injectée dans le prompt.
2. **`web_research`** : DuckDuckGo (sans clé) ou SearXNG (auto-hébergé) ; sources dédupliquées.
3. **`summarize_sources`** : **résumé glissant** — le prompt distingue explicitement « créer un NOUVEAU résumé » et « ÉTENDRE un résumé existant » (intégrer chaque info nouvelle dans le paragraphe pertinent, sinon nouveau paragraphe, sinon ignorer). C'est la mémoire de la boucle.
4. **`reflect_on_summary`** : le LLM analyse le résumé et **écrit noir sur blanc la lacune de connaissance** (`knowledge_gap`) + la question de suivi. La requête suivante ne cible *que* cette lacune.
5. Boucle jusqu'à `max_web_research_loops`, puis rapport markdown final avec toutes les sources citées.

## Techniques à retenir

- **La réflexion explicite (« knowledge gap ») est LE mécanisme anti-boucle** : forcer le modèle à nommer ce qui manque avant de re-chercher évite de refaire la même requête.
- **Deux modes de sortie structurée** dans les prompts : `json_mode` et `tool_calling` — parce que les petits modèles distillés (DeepSeek R1 1.5B/7B) échouent souvent sur le JSON strict. Le README documente ce problème et le fallback.
  → **En C++/llama.cpp ce problème disparaît** : la grammaire GBNF contraint physiquement la sortie. Notre avantage compétitif direct.
- Le prompt de résumé glissant (« extend an existing summary ») maintient un contexte O(1) au lieu d'accumuler les sources — crucial à petit contexte.
- Stack réellement sans clé API : DuckDuckGo ou SearXNG + Ollama. Prouve la faisabilité du 100 % local.

## Dépendances externes & limites

- LangGraph (runtime Python, serveur `langgraph dev`) — lourd pour ce que fait la boucle ; la logique elle-même est triviale à porter.
- Une seule requête par itération (pas de parallélisme) : plus lent mais plus simple et plus économe.
- Le résumé glissant perd du détail au fil des itérations (pas de cache des sources brutes).

## À réutiliser pour Laplace

- Le **nœud de réflexion** (nommer la lacune) à insérer dans notre machine à états, entre synthèse et génération de la requête suivante.
- Le prompt de résumé glissant, quasi copiable tel quel pour notre modèle 8-14B.
- SearXNG en Docker comme moteur de recherche par défaut de Laplace (clé-free, air-gap possible).
- Leur constat d'échec JSON des petits modèles = argument massue pour l'architecture GBNF native.

## Liens à creuser (besoins)

- IterDRAG (l'inspiration citée) : recherche du papier « Inference Scaling for Long-Context RAG » (Google, arXiv 2410.04343).
- SearXNG : <https://docs.searxng.org> — à déployer en local pour Laplace.
