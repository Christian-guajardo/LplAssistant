# Alibaba-NLP/DeepResearch (Tongyi) — le modèle *entraîné pour* la recherche agentique

**Source :** <https://github.com/Alibaba-NLP/DeepResearch>
**Type :** Modèle ouvert (Tongyi-DeepResearch-30B-A3B) + stack d'inférence/évaluation Python | **Analysé le :** 2026-07-10

## En une phrase

L'approche « modèle d'abord » : un MoE 30,5 B paramètres dont **3,3 B activés par token** (coût d'inférence d'un ~3B, capacité d'un 30B), 128K de contexte, entraîné de bout en bout (pré-entraînement continu agentique + SFT + RL on-policy type GRPO) spécifiquement pour les tâches de recherche longues — SOTA sur Humanity's Last Exam, BrowseComp, WebWalkerQA, FRAMES, SimpleQA.

## Ce que contient le repo

- **Inference** : scripts ReAct (`run_react_infer.sh`) multi-workers ; entrée JSONL `{"question","answer"}` ; possibilité de préfixer un nom de fichier à la question pour tester le traitement de documents.
- **Deux paradigmes d'inférence** :
  - **ReAct pur** : boucle Thought→Action→Observation classique, pour mesurer les capacités intrinsèques du modèle (pas de prompt engineering).
  - **IterResearch « Heavy » mode** : *test-time scaling* — plusieurs agents de recherche en parallèle, puis synthèse itérative de leurs rapports ; maximise le plafond de performance au prix de beaucoup de tokens.
- **WebAgent** : la lignée de travaux amont (WebWalker, WebDancer, WebSailor…) du même labo.
- **Évaluation** : harnais de bench sur les suites citées.

## Les leçons transférables (sans réentraîner quoi que ce soit)

- **La géométrie MoE A3B est exactement le profil qu'il faut viser en local** : 30B-A3B en Q4 ≈ 18-20 Go, activations d'un 3B ⇒ tourne sur la config de l'utilisateur (36 Go de mémoire partagée Intel) à une vitesse acceptable. llama.cpp gère les MoE (`--cpu-moe`, `--n-cpu-moe` pour délester les experts sur CPU). **Candidat sérieux de modèle pour le mode « recherche profonde » de Laplace**, à côté du petit modèle rapide pour le dialogue.
- **IterResearch** : l'idée de reconstruire un espace de travail *frais* à chaque tour (au lieu d'empiler tout l'historique) pour éviter la « suffocation cognitive » du contexte — converge avec la compaction d'opencode/OpenClaw.
- La **stack d'outils** qu'ils branchent au modèle : recherche (Serper), lecture de pages (Jina Reader), parsing de fichiers (Dashscope), interpréteur Python sandboxé (SandboxFusion). C'est la liste canonique des 4 outils d'un agent de recherche ; Laplace doit fournir les mêmes en local.
- Le format de leurs prompts système ReAct est public dans `inference/` — réutilisable pour un modèle local.

## Dépendances externes & limites

- Toute la stack d'inférence fournie dépend d'APIs commerciales (Serper, Jina, Dashscope, OpenAI pour le juge) — à remplacer intégralement en local.
- Python 3.10 strict, dépendances lourdes ; le repo est un artefact de recherche, pas un produit.
- Le « Heavy mode » est dispendieux : à réserver aux questions difficiles (mode explicite).

## À réutiliser pour Laplace

- Tester **Tongyi-DeepResearch-30B-A3B en GGUF** comme « gros cerveau » de recherche de Laplace (llama.cpp le supporte ; dispo aussi via OpenRouter pour comparer avant de télécharger).
- Le duo de modes ReAct (rapide) / Heavy (test-time scaling explicite) comme les deux vitesses du module recherche.
- Leur liste de benchmarks = étalon pour notre mini-banc d'éval.

## Liens à creuser (besoins)

- Tech blog : <https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/> ; Tech_Report.pdf inclus dans le repo (non lu — **besoin : lecteur PDF local**).
- Modèle : <https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B> (chercher les conversions GGUF communautaires).
- Lignée WebAgent (WebSailor, WebDancer) dans le dossier `WebAgent/` du repo.
