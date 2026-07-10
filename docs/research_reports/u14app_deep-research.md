# u14app/deep-research — le deep research « produit » : dual-retrieval, SSE et MCP

**Source :** <https://github.com/u14app/deep-research>
**Type :** Application Next.js full-stack (web UI + API) | **Analysé le :** 2026-07-10

## En une phrase

Un deep research complet packagé en produit : plan de recherche validé par l'utilisateur, recherche simultanée **web + base de connaissances locale**, sortie rapport + **graphe de connaissances**, le tout exposé en API SSE et en serveur MCP.

## Architecture & pipeline

1. **Topic + contexte** : question de l'utilisateur + documents locaux uploadés (texte, Office, PDF) qui forment une base de connaissances locale.
2. **Idéation** : génération d'un plan de recherche + questions de clarification, que l'utilisateur peut amender avant lancement.
3. **Collecte (dual-retrieval)** : les requêtes SERP générées interrogent en parallèle la base locale (RAG) et le web.
4. **Recherche récursive** : évaluation des findings, génération de requêtes plus profondes jusqu'à satisfaction (héritage direct de dzhng/deep-research).
5. **Sortie** : rapport markdown + génération optionnelle d'un **knowledge graph** du contenu.

Deux modèles distincts : « Thinking model » (raisonnement/planification) et « Task model » (exécution) — même philosophie de rôles que open_deep_research, en plus simple.

## Intégrations (la vraie valeur du repo)

- **API SSE** (`/api/sse`, POST et GET) : streaming temps réel de l'avancement (étapes, pensées, findings) vers n'importe quel client.
- **Serveur MCP** (`/api/mcp`, transports StreamableHTTP et SSE) : le deep research devient un *outil* consommable par n'importe quel agent (Claude, Copilot…), protégeable par `ACCESS_PASSWORD`.
- Moteurs de recherche : SearXNG (self-host), Tavily, Firecrawl, etc. LLM : Gemini, OpenAI, Anthropic, Ollama, DeepSeek…
- Mode « local API » : tout tourne côté navigateur, aucune clé serveur — la confidentialité comme argument produit.

## Techniques à retenir

- **Dual-retrieval systématique** : chaque requête générée frappe le savoir local ET le web. Pour Laplace, qui a déjà pgvector : le module recherche doit interroger `memories` (et plus tard le LLM-wiki) avec les mêmes requêtes que le web, et fusionner les résultats avant synthèse.
- **Le plan validé par l'utilisateur avant de brûler du compute** (héritage Plan-and-Execute) — en REPL vocal, Laplace peut énoncer son plan et attendre un « vas-y ».
- **Exposer le pipeline en SSE + MCP** : sépare le moteur de l'interface. Pour Laplace : le protocole UDP existant joue le rôle du SSE ; une façade MCP rendrait le moteur réutilisable par d'autres clients.
- Knowledge graph comme *artefact de sortie* (compréhension systémique), pas comme store de retrieval — cohérent avec le rejet du GraphRAG du rapport Jarvis, et avec WikiGraphs.

## Dépendances externes & limites

- Next.js/Vercel-centrique ; la logique de recherche est mêlée au produit web.
- La qualité dépend entièrement des providers configurés ; pas d'évaluateur de réponse.
- Pas de checkpoint/resume persistant du run de recherche.

## À réutiliser pour Laplace

- Le contrat d'API du SSE (états : plan → queries → findings → report) comme spécification de nos messages de progression vers les satellites.
- Le pattern MCP-server-du-moteur pour plus tard (Laplace outil des autres agents).
- La liste de ses providers de recherche = catalogue des options à évaluer (SearXNG retenu).

## Liens à creuser (besoins)

- Doc SearXNG : <https://docs.searxng.org>.
- Leur doc API MCP/SSE dans le README (section « API documentation ») comme référence de conception.
