# ggml-org/llama.cpp — le moteur (et il fait plus qu'on ne croyait)

**Sources :** <https://github.com/ggml-org/llama.cpp> + profil org <https://github.com/ggml-org/.github> (README profil lu)
**Type :** Moteur d'inférence C/C++ (ggml) + serveur HTTP + outillage | **Analysé le :** 2026-07-10 (lecture `tools/server/README.md` 2 057 l., `docs/function-calling.md`, `grammars/`, `examples/`)

## En une phrase

Laplace n'a pas besoin d'écrire un runtime d'inférence : `llama-server` fournit déjà l'API OpenAI **et** Anthropic-compatible, le tool-calling universel, les grammaires, le décodage spéculatif, les slots parallèles et le cache de prompt intelligent — en pur C++ (httplib + nlohmann::json), notre stack exacte.

## L'écosystème (profil ggml-org)

ggml (lib ML) → llama.cpp (LLM) + whisper.cpp (STT — déjà dans Laplace). Autour : llama.vscode/vim, LlamaBarn, llama.app. Depuis février 2026, **ggml.ai a rejoint Hugging Face** (pérennité du projet). Partenaires : NVIDIA, HF. whisper.cpp v1.8.5 (mai 2026) — notre 1.6.2 est en retard.

## Capacités de `llama-server` critiques pour Laplace

- **API** : `/v1/chat/completions`, `/v1/responses`, embeddings, **reranking** (PR 9510 — remplace un reranker externe !), Anthropic Messages compat, endpoints de monitoring `/slots`, `/metrics`.
- **Parallélisme** : `-np N` slots, continuous batching, KV unifié partageable entre séquences (`--kv-unified`) — plusieurs « chercheurs » simultanés sur un seul modèle en RAM.
- **Cache de prompt** : `--cache-prompt` (défaut on) + **`--cache-reuse N`** : réutilisation de chunks du KV par *shifting* même quand le préfixe n'est pas identique. Notre réutilisation manuelle de préfixe dans `llm.cpp` existe déjà côté serveur, en mieux.
- **Sorties contraintes** : `--grammar`/`--grammar-file` (GBNF), `-j/--json-schema` (conversion JSON-Schema→GBNF intégrée, aussi par requête via le champ `json_schema`), `grammars/*.gbnf` d'exemple + `examples/json_schema_to_grammar.py`. **Garantie physique de JSON valide pour la machine à états.**
- **Function calling universel** (`--jinja`, `docs/function-calling.md`) : formats natifs pour Qwen 2.5 (notre modèle !), Hermes, Llama 3.x, Mistral Nemo, Command R7B, DeepSeek R1 + **handler générique** pour tout autre template. `parallel_tool_calls` optionnel. Le parsing des tool calls est fait côté serveur.
- **Raisonnement contrôlé** : `--reasoning-format` (extraction des `<think>` en `reasoning_content`), `--reasoning-budget N` (+ message d'épuisement) — budget de réflexion natif, utile pour borner la latence vocale.
- **Décodage spéculatif** serveur : `--spec-draft-hf`/`-md` modèle draft + `--spec-draft-n-max` (défaut 3), types KV draft séparés, affinités CPU dédiées. Aussi : **lookup decoding** (`-lcs`/`-lcd`, cache de n-grams statique/dynamique) et `examples/lookahead`.
- **Divers utiles** : multimodal, préfill de réponse assistant (style Claude), LoRA à chaud (`--lora`), control vectors, `--offline`, quantif KV (`-ctk/-ctv q8_0`…), `--mlock`, affinités/priorités threads (`--cpu-mask`, `--prio`), MoE offload (`--cpu-moe`, `-ncmoe`).
- `examples/reason-act.sh` : mini boucle ReAct de référence en shell ; `examples/retrieval`, `examples/parallel`, `examples/embedding`.

## Conséquences d'architecture pour Laplace

1. **Basculer d'un lien direct libllama vers un split process** : un daemon `llama-server` (modèle principal + slots + spéculatif) et le binaire Laplace en client HTTP localhost. Bénéfices immédiats : batching multi-locuteurs propre, tool-calls parsés, grammaires par requête, plus de conflit de link ggml entre whisper.cpp et llama.cpp (notre bug historique), embeddings + rerank servis par le même daemon.
   (Alternative conservée : garder libllama en in-process pour la latence du chat court, et le serveur pour la recherche — à trancher au banc.)
2. La boucle ReAct de Laplace = client HTTP + grammaire GBNF des actions autorisées, régénérée à chaque étape.
3. Draft model : Qwen 0.5B en draft du 14B ; mesurer le gain réel sur notre matériel.

## Limites

- Pas d'orchestration d'agent, pas de mémoire, pas d'outils : c'est un moteur — tout le reste (la valeur de Laplace) est à nous.
- Bande passante mémoire = plafond dur en local (config utilisateur : mémoire partagée Intel ~80-100 Go/s → un 14B Q4 ≈ 10-15 tok/s, un 30B-A3B MoE bien plus rapide que sa taille ne le suggère).

## Liens à creuser (besoins)

- Changelog serveur (issue 9291) et `docs/multimodal.md` ; llama.app (<https://llama.app>).
- Mise à jour whisper.cpp vers 1.8.x (gains perf STT + corrections).
- HF ggml-org : <https://huggingface.co/ggml-org> (GGUF officiels).
