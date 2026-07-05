add_rules("mode.debug", "mode.release")

set_languages("c++20")

add_requires("llama.cpp", {configs = {shared = false}})
add_requires("whisper.cpp", {configs = {shared = false}})

-- Binaire principal : LLM + embeddings + PostgreSQL.
-- Ne linke PAS whisper.cpp : son ggml embarqué entre en conflit de symboles
-- avec celui de llama.cpp (segfault au chargement du modèle).
target("LplAssistant")
    set_kind("binary")
    add_files("src/*.cpp")
    remove_files("src/stt.cpp", "src/stt_main.cpp", "src/mic_main.cpp")
    add_packages("llama.cpp")
    add_syslinks("pqxx", "pq", "pthread")

-- Micro-service STT isolé (whisper.cpp seul).
target("laplace-stt")
    set_kind("binary")
    add_files("src/stt.cpp", "src/stt_main.cpp")
    add_packages("whisper.cpp")
    add_syslinks("pthread")

-- Satellite micro de test (fallback local WSL) : capte le micro et streame
-- en UDP vers `LplAssistant --listen`. Reproduit le futur ESP32.
target("laplace-mic")
    set_kind("binary")
    add_files("src/mic_main.cpp")
    add_syslinks("pulse-simple", "pulse", "pthread")
