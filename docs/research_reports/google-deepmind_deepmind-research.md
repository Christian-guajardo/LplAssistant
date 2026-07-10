# google-deepmind/deepmind-research — WikiGraphs & Continual Learning

**Sources :** <https://github.com/google-deepmind/deepmind-research> (README racine), sous-projets `wikigraphs/` et `continual_learning/`
**Type :** Archive de code de recherche (implémentations accompagnant des papiers, non maintenues) | **Analysé le :** 2026-07-10

## En une phrase

Deux fondations théoriques pour la mémoire de Laplace : un dataset/baseline **texte ⇄ graphe de connaissances** (WikiGraphs) et une architecture de **mémoire k-NN à classifieurs locaux** qui élimine l'oubli catastrophique sans connaître les frontières de tâches (Continual Learning).

## 1. WikiGraphs (`wikigraphs/`, papier arXiv 2107.09556)

- **Dataset** : chaque article de WikiText-103 apparié à un sous-graphe du knowledge graph Freebase. Conçu pour deux tâches : générer du texte long *conditionné par un graphe* (graph2text) et récupérer/générer le graphe depuis le texte.
- **Baselines** : TransformerXL conditionné par le graphe encodé en BoW, ou par un vrai GNN (Jraph) ; stack JAX/Haiku/Optax.
- **Leçon pour Laplace** : la direction « représenter le savoir en graphe, puis générer le texte depuis le graphe » améliore la factualité des synthèses longues — mais le coût (extraction, maintenance du graphe) est exactement ce que le rapport Jarvis rejette dans le GraphRAG. Position retenue : **graphe comme artefact de sortie optionnel** (comme u14app), pas comme store de retrieval.

## 2. Continual Learning (`continual_learning/`, papier arXiv 2105.13327, Shanahan et al.)

Le modèle « Encoders and Ensembles » :
- un **encodeur pré-entraîné figé** (sur un autre dataset que la cible) ;
- une **mémoire à clés fixes aléatoires** avec lookup **k-NN** ;
- chaque emplacement mémoire stocke les paramètres d'un **petit classifieur local entraînable** ;
- la sortie = moyenne des k classifieurs sélectionnés, **pondérée par la distance** entre leur clé et l'encodage de l'entrée.

Réglage démontré : **task-free** (pas de frontières de tâches), **online** (données vues une seule fois), apprentissage incrémental de classes. Résultat : l'apprentissage est localisé — seuls les classifieurs proches de l'entrée bougent — donc pas d'oubli catastrophique.

- **Leçon pour Laplace** : c'est la justification théorique de notre Composite Retrieval, et sa généralisation :
  - notre pgvector + bge-m3 = l'encodeur figé + la mémoire adressée par similarité ;
  - l'idée à voler : attacher aux entrées mémoire non pas seulement du *texte* mais des **comportements** (préférences, règles de réponse, profils par locuteur) pondérés par la distance cosinus au contexte courant — une « mémoire de personnalité » qui s'apprend en ligne, une interaction à la fois, sans fine-tuning.
  - la pondération par distance (au lieu du top-k sec) lisse les réponses quand la frontière entre souvenirs est floue.

## Dépendances externes & limites

- Code de recherche 2021, JAX ancien, non maintenu — on prend les *idées*, pas le code.
- Freebase est mort (remplacé par Wikidata) ; WikiGraphs reste utile comme référence de conception.

## À réutiliser pour Laplace

- Pondération par distance dans le retrieval pgvector (score = f(cosinus) au lieu de coupure top-k brute).
- Mémoire « comportementale » k-NN par-dessus la mémoire factuelle existante.
- Éval de la mémoire en régime online/task-free : le banc d'essai réaliste d'un assistant domestique.

## Liens à creuser (besoins)

- arXiv 2105.13327 (Encoders and Ensembles) et 2107.09556 (WikiGraphs) — **besoin : récupération + lecture de PDF arXiv en local**.
- WikiText-103 et Wikidata pour toute future expérimentation graphe.
