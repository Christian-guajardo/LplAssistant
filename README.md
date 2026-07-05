# Laplace — Assistant personnel local (LplAssistant)

Assistant vocal/textuel 100 % local (C++20), basé sur l'architecture du rapport
d'implémentation : llama.cpp (LLM), whisper.cpp (STT), PostgreSQL + pgvector
(mémoire long terme en **Composite Retrieval** : SQL pour filtrer, vecteurs
pour rapprocher).

## Architecture

```
┌────────────── LplAssistant (binaire principal) ──────────────┐
│  main.cpp   REPL / --ask / --wav / --listen (UDP)            │
│  udp_audio  serveur UDP multi-satellites (démux par adresse) │
│  voiceprint empreinte vocale (timbre+pitch, 1 session/pers.) │
│  tts        synthèse vocale (Piper, espeak-ng en secours)    │
│  laplace-mic (binaire) satellite micro+haut-parleur local    │
│  agent      retrieval → prompt → génération → mémorisation   │
│  llm        llama.cpp + KV cache réutilisé (préfixe commun)  │
│  embedder   bge-m3 (1024d, multilingue), normalisé L2        │
│  db         pqxx → PostgreSQL + pgvector (index HNSW)        │
└──────────────────────────────────────────────────────────────┘
        │ popen
        ▼
  laplace-stt (binaire isolé)  ← whisper.cpp
```

Le STT vit dans un binaire séparé : le ggml embarqué de whisper.cpp 1.6.2 est
incompatible au link avec celui de llama.cpp b3775 (segfault sinon).

## Modèles (dossier `models/`)

| Rôle | Fichier | Taille |
| --- | --- | --- |
| LLM | qwen2.5-1.5b-instruct-q4_k_m.gguf | ~1 Go |
| Embeddings | bge-m3-q4_k_m.gguf | ~420 Mo |
| STT | ggml-base.bin (whisper base) | ~142 Mo |

## Prérequis

- PostgreSQL démarré (`service postgresql start` sous WSL) avec la base/rôle
  `laplace` (mot de passe `laplace`) et l'extension pgvector.
- Le schéma (`memories` + index HNSW) est créé automatiquement au lancement.

## Build & run

```sh
xmake --root            # build (LplAssistant + laplace-stt)
./build/linux/x86_64/release/LplAssistant                 # REPL (commandes: /mem /forget /quit)
./build/linux/x86_64/release/LplAssistant --ask "..."     # question unique
./build/linux/x86_64/release/LplAssistant --wav voix.wav  # entrée vocale (WAV PCM16, mono/stéréo, tout taux)
./build/linux/x86_64/release/LplAssistant --listen 7777   # serveur UDP audio (satellites)
xmake f --root -m debug # passer en mode debug
```

## Streaming audio UDP

`--listen [port]` (défaut 7777) attend des datagrammes UDP de **PCM16LE mono
16 kHz** bruts. Fin d'énoncé : datagramme `END` ou 800 ms sans paquet.

- **Multi-satellites** : chaque adresse source (ip:port) a son propre buffer
  d'énoncé — plusieurs personnes peuvent parler à des satellites différents
  en même temps (traitement séquentiel : un seul LLM en RAM).
- **Sessions par locuteur** : chaque énoncé reçoit une empreinte vocale
  (timbre spectral + hauteur de voix, cosinus, seuil
  `LAPLACE_VOICE_THRESHOLD` défaut 0.80). Une voix reconnue garde **son**
  historique de dialogue, quel que soit le satellite : on peut commencer
  dans la cuisine et continuer dans la salle à manger. Voix inconnue →
  contexte partagé « global ».
- **Calibrage vocal** : dire « Laplace, calibration *prénom*, ... » (finir la
  phrase pour donner assez de voix) ; répéter 2-3 fois pour affiner. Profils
  dans `voiceprints.tsv`. Ce n'est pas de la biométrie de sécurité : c'est
  fait pour distinguer les personnes d'un foyer.
- **Mot d'appel** : Laplace ne répond que si l'énoncé commence par
  « laplace » (vérifié sur la transcription, variantes « la place »/« la
  passe » acceptées ; sur ESP32 ce sera du TinyML on-device). Sinon :
  datagramme `NOP`, aucune réponse.
- **Réponse (vers le satellite émetteur uniquement)** : `TXT:<texte>`, puis
  la voix synthétisée (PCM16 mono 16 kHz, paquets de 40 ms), puis `AEND`.
  Seul le satellite qui a entendu la question reçoit — et parle.

### Synthèse vocale

`tts.cpp` cherche **Piper** (`third_party/piper/piper` + modèle
`models/fr_FR-siwis-medium.onnx`, voir `LAPLACE_PIPER_DIR` /
`LAPLACE_TTS_VOICE`) : voix naturelle. À défaut, repli automatique sur
**espeak-ng** (robotique). Le moteur utilisé est affiché au premier énoncé
(`[laplace] voix: ...`).

### Satellite micro local (`laplace-mic`)

En attendant les ESP32, `laplace-mic` sert de satellite de secours : il capte
le micro par défaut (PulseAudio ; sous WSL c'est le micro Windows via WSLg),
détecte la parole par énergie (VAD à hystérésis + 700 ms de *hangover*), et
streame le PCM16 mono 16 kHz en UDP — paquets de 40 ms puis `END`, exactement
le protocole cible de l'ESP32. Il affiche la réponse texte, **joue la voix**
sur le haut-parleur par défaut, puis purge le micro (anti-larsen) et repasse
en écoute : un seul binaire alterne écoute/parole sur la même socket UDP.

```sh
# Terminal 1 : l'assistant
./build/linux/x86_64/release/LplAssistant --listen 7777
# Terminal 2 : le satellite micro (puis dites « Laplace, ... »)
./build/linux/x86_64/release/laplace-mic --host 127.0.0.1 --port 7777
```

## Configuration (variables d'environnement)

`LAPLACE_DB`, `LAPLACE_LLM_MODEL`, `LAPLACE_EMBED_MODEL`, `LAPLACE_STT_MODEL`,
`LAPLACE_N_CTX`, `LAPLACE_N_THREADS`, `LAPLACE_MAX_TOKENS`
(voir [src/config.h](src/config.h)).

## Optimisations implémentées

- **KV cache persistant** : à chaque tour, seul le suffixe du prompt qui
  diffère du tour précédent est re-décodé (préfixe commun conservé).
- **Économie de tokens** : seuls les 8 derniers tours restent dans le contexte ;
  le reste est archivé en base et réinjecté à la demande via le retrieval
  (seuil de similarité cosinus 0.45, top-5).
- **Index HNSW** pgvector pour une recherche vectorielle sub-linéaire.
- **Sampling stabilisé** : pénalité de répétition 1.15 + top-k/min-p
  (indispensable sur un modèle 1.5B).

## Étapes suivantes (rapport, non implémentées)

- Serveurs/clients MCP, boucle ReAct avec outils.
- Firmware ESP32 (côté serveur, le streaming UDP est prêt : `--listen`).
- Tailscale/RustDesk pour l'accès distant.
- Passage CUDA (Jetson) : recompiler llama.cpp/whisper.cpp avec le support GPU.
