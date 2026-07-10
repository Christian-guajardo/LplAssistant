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
service postgresql start #lance le postgresql
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
  en même temps (un seul LLM en RAM : les réponses sont générées une à une,
  mais la réception et le triage ne bloquent jamais).
- **Pipeline live (3 étages)** : le serveur ne fabrique plus toute la réponse
  avant de l'envoyer. Il diffuse **au fil de l'eau** — dès la première phrase
  générée : `LLM → file de phrases → TTS → anneau de trames → envoi cadencé`.
  Chaque étage tourne dans son thread avec des tampons bornés (contre-pression :
  le LLM ne prend jamais plus de ~2 s d'avance sur la voix). Résultat : le son
  démarre en ~1 s au lieu d'attendre 10 s le paragraphe entier.
- **Interruption / priorité au dernier énoncé** : le triage (STT + routage)
  tourne **pendant** la génération. Un « Laplace, stop » coupe la réponse en
  cours (annulation du LLM + `STOP` au satellite). Une **nouvelle question**
  du même locuteur **écrase** la réponse en cours (barge-in) : Laplace laisse
  tomber l'ancienne et répond à la nouvelle.
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
- **Gestion des profils** (à la voix) : « Laplace, **supprime tous les
  profils** » (ou « réinitialise les profils ») efface tout le registre ;
  « Laplace, **supprime le profil *prénom*** » n'enlève qu'un profil
  (insensible à la casse). Laplace confirme de vive voix.
- **Mot d'appel** : la transcription est **entièrement mise en minuscules**,
  puis Laplace ne répond que si l'énoncé commence par « laplace » (variantes
  « la place »/« la passe »/« laplasse » acceptées, car Whisper base les
  confond ; sur ESP32 ce sera du TinyML on-device). Sinon : aucune réponse.
- **Réponse (vers le satellite émetteur uniquement)** : un `TXT:<phrase>` par
  segment au fil de la génération, entremêlé de la voix synthétisée (PCM16
  mono 16 kHz, paquets de 40 ms cadencés temps réel), puis `AEND` en fin —
  ou `STOP` si la réponse est coupée. Seul le satellite qui a entendu la
  question reçoit — et parle.
- **Full-duplex (écoute + parole simultanées)** : le satellite écoute en
  permanence, même pendant qu'il joue une réponse. On peut donc l'**interrompre
  à la voix** : « Laplace, stop » (aussi « arrête », « tais-toi », « chut »,
  « silence ») → le serveur renvoie `STOP`, le satellite coupe la lecture net.
- **Anti-écho par le contenu** : quand un satellite joue la réponse, son micro
  peut la capter. Plutôt qu'une annulation d'écho DSP, le serveur retient ce
  qu'il vient de dire à chaque satellite : un énoncé reçu dans la fenêtre de
  lecture dont ≥ 70 % des mots sont dans la dernière réponse est reconnu comme
  son propre écho et ignoré. Laplace ne se répond donc pas à lui-même, et
  reste à l'écoute pendant qu'il parle.

### Synthèse vocale

`tts.cpp` cherche **Piper** (`third_party/piper/piper` + modèle
`models/fr_FR-siwis-medium.onnx`, voir `LAPLACE_PIPER_DIR` /
`LAPLACE_TTS_VOICE`) : voix naturelle. À défaut, repli automatique sur
**espeak-ng** (robotique). Le moteur utilisé est affiché au premier énoncé
(`[laplace] voix: ...`). Le texte est **nettoyé du markdown** avant synthèse
(puces, `*gras*`, `` `code` ``, liens `[texte](url)`) : la voix ne prononce
pas les caractères de mise en forme, seulement le contenu.

### Satellite micro local (`laplace-mic`)

En attendant les ESP32, `laplace-mic` sert de satellite de secours : il capte
le micro par défaut (PulseAudio ; sous WSL c'est le micro Windows via WSLg),
détecte la parole par énergie (VAD à hystérésis + 700 ms de *hangover*), et
streame le PCM16 mono 16 kHz en UDP — paquets de 40 ms puis `END`, exactement
le protocole cible de l'ESP32. Il affiche la réponse texte et **joue la voix**
sur le haut-parleur par défaut. **Écoute et parole tournent en parallèle**
(threads séparés micro / réseau / lecture, une seule socket UDP) : on peut lui
parler pendant qu'il répond, et « Laplace, stop » (`STOP`) coupe la lecture
immédiatement. Pendant la lecture les seuils VAD sont relevés (le HP excite le
micro) et le serveur filtre l'écho par le contenu (voir plus haut).

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

## Deep research (module `src/research/`)

Recherche profonde autonome : machine à états à actions typées (inspirée de
jina node-DeepResearch), grammaire GBNF contraignant chaque décision JSON,
budget de tokens avec « beast mode », checkpoint reprenable et rapport
markdown sourcé. Voir [master_research_report.md](docs/master_research_report.md)
et [research_reports/](docs/research_reports/) pour la genèse (15 sources analysées).

```
plan → [ boucle: search → read(CCR) → reflect(gaps) → answer(évalué) ] → rapport
```

- **search** : SearXNG (local, sans clé) + providers API-first (Wikipedia,
  OpenAlex, arXiv, StackExchange, GitHub) — chacun classé ok/empty/error.
- **read** : lecteur HTML→texte maison (retire script/nav/footer) puis **CCR**
  (Compress-Cache-Retrieve) : le texte intégral va en cache disque, seule une
  vue écrémée bornée entre dans le prompt — aucune source n'est tronquée.
- **reflect** : nomme explicitement les lacunes et pousse des sous-questions.
- **answer** : réponse évaluée (définitude/attribution/couverture) ; rejetée →
  la boucle repart avec la critique.
- **rapport** : rédaction progressive par sections + quality gates URL
  (HEAD→GET, offline-aware) + diagnostics providers + limites.

```sh
# LLM via llama-server (cible : slots parallèles + grammaire par requête)
LAPLACE_RESEARCH_LLM_URL=http://127.0.0.1:8080 \
LAPLACE_SEARXNG_URL=http://127.0.0.1:8888 \
./build/linux/x86_64/release/laplace-research "sujet à creuser"

# ou modèle local in-process (grammaire GBNF native) — sans llama-server
./build/linux/x86_64/release/laplace-research "sujet" --guidance "précisions"
./build/linux/x86_64/release/laplace-research --resume research_runs/<run>/

# depuis le REPL de l'assistant
Vous> /research pourquoi mon init Vulkan plante sur Intel Arc
# ou en une passe
./build/linux/x86_64/release/LplAssistant --research "sujet"
```

Variables : `LAPLACE_RESEARCH_LLM_URL`, `LAPLACE_SEARXNG_URL`, `GITHUB_TOKEN`,
`LAPLACE_RESEARCH_{BUDGET,MAX_STEPS,PROVIDERS,DIR}`. Tests bout-en-bout
hors-ligne (LLM + web + SearXNG simulés) : `tests/mock_research_stack.py`.

## Étapes suivantes (rapport, non implémentées)

- Serveurs/clients MCP, boucle ReAct avec outils.
- Firmware ESP32 (côté serveur, le streaming UDP est prêt : `--listen`).
- Tailscale/RustDesk pour l'accès distant.
- Passage CUDA (Jetson) : recompiler llama.cpp/whisper.cpp avec le support GPU.
