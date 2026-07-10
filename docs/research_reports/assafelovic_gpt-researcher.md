# assafelovic/gpt-researcher — le pionnier Planner/Executor/Publisher

**Source :** <https://github.com/assafelovic/gpt-researcher>
**Type :** Bibliothèque + appli Python (LangGraph/AG2 pour le mode multi-agents) | **Analysé le :** 2026-07-10

## En une phrase

Le projet fondateur (2023) du deep research open source : un **planner** qui décompose la question en sous-questions objectives, des **execution agents** (crawlers) parallèles, un **publisher** qui agrège en rapport long (2 000+ mots) avec suivi des sources — inspiré des papiers Plan-and-Solve et STORM.

## Architecture & pipeline

1. Création d'un agent spécifique à la tâche à partir de la requête (prompt d'agent choisi dynamiquement selon le domaine).
2. **Génération de sous-questions** qui, ensemble, forment « une opinion objective » sur le sujet — la formulation anti-biais est explicite : sources multiples (20+), diversité de domaines.
3. Un **crawler agent par sous-question** : recherche (Tavily par défaut, ~10 retrievers supportés), scraping (avec exécution JavaScript), résumé + traçage de la source pour chaque ressource.
4. **Filtrage et agrégation** des résumés → rapport final structuré ; export PDF/Word/MD ; images pertinentes scrapées et filtrées ; option d'images générées inline (Gemini).
5. Mode **Deep Research** (arbre récursif largeur/profondeur, « tree-like exploration with configurable depth/breadth ») et mode **multi-agents** LangGraph (équipe : chief editor, researcher, reviewer, revisor, writer, publisher).

## Les points uniques dans le paysage

- **Recherche hybride web + documents locaux** (`DOC_PATH` : PDF, docx, csv, md, pptx…) mature depuis longtemps.
- **MCP dans les deux sens** : client MCP (chercher dans GitHub, bases custom, en *complément* du web) ET serveur MCP (`gptr-mcp`) pour être piloté par Claude Desktop & co. Distribution en « Claude Skill » (`npx skills add assafelovic/gpt-researcher`).
- Vaste documentation d'ingénierie (docs.gptr.dev, blog « How we built GPT Researcher ») : gestion des coûts (~0,40 $/rapport en 2023, ~0,10 $ aujourd'hui), lutte contre les hallucinations par agrégation massive, prompts par type de rapport (research, resource, outline).

## Techniques à retenir

- **La parallélisation des sous-questions est le levier n°1 de latence** : rapport complet en ~3 min là où l'approche séquentielle en met 15+. Pour Laplace : slots parallèles de `llama-server` + pool de threads réseau.
- **L'objectivité par le volume** : 20+ sources agrégées réduisent mécaniquement le biais d'une source unique — d'où l'importance d'un scraper qui ne meurt pas sur 30 % des sites.
- **Un prompt d'agent choisi selon le domaine de la question** (finance, tech, voyage…) : bon compromis spécialisation/simplicité, applicable à Laplace avec une table de personas.
- Le rôle « reviewer/revisor » du mode multi-agents (relecture avant publication) préfigure nos quality gates.

## Dépendances externes & limites

- OpenAI par défaut (choix « stabilité ») ; LLM alternatifs configurables mais les prompts sont calibrés GPT.
- Tavily par défaut ; le scraping JS s'appuie sur des headless browsers (lourd).
- Python + LangGraph pour le mode avancé ; beaucoup de surface produit (frontend NextJS, etc.).

## À réutiliser pour Laplace

- La **taxonomie des rôles** (planner / crawlers parallèles / reviewer / publisher) comme vocabulaire de notre machine à états.
- La stratégie anti-biais : imposer un minimum de sources et de diversité de domaines dans nos quality gates.
- Le pattern « summarize + source-track chaque ressource » (résumé attaché à son URL) : c'est ce qui rend les citations finales fiables.

## Liens à creuser (besoins)

- Blog fondateur : <https://docs.gptr.dev/blog/building-gpt-researcher> ; papiers cités : Plan-and-Solve (arXiv 2305.04091), RAG (arXiv 2005.11401), STORM (arXiv 2402.14207).
- `gptr-mcp` : <https://github.com/assafelovic/gptr-mcp>.
- Deep Research de gptr : <https://docs.gptr.dev/docs/gpt-researcher/gptr/deep_research>.
