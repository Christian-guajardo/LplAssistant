add_rules("mode.debug", "mode.release")

set_languages("c++20")

add_requires("nlohmann_json")
add_requires("cpp-httplib")

add_requires("llama.cpp", {configs = {shared = false}})
add_requires("whisper.cpp", {configs = {shared = false}, optional = true})

target("LplAssistant")
    set_kind("binary")
    add_files("src/*.cpp", "src/research/*.cpp")
    remove_files("src/stt.cpp", "src/stt_main.cpp", "src/mic_main.cpp",
                 "src/research/research_main.cpp")
    add_packages("llama.cpp", "nlohmann_json", "cpp-httplib")
    add_syslinks("pqxx", "pq", "pthread")

-- CLI autonome du deep research : même moteur, sans base ni audio.
-- LLM via llama-server (LAPLACE_RESEARCH_LLM_URL) ou modèle local in-process.
target("laplace-research")
    set_kind("binary")
    add_files("src/research/*.cpp", "src/llm.cpp", "src/config.cpp")
    add_packages("llama.cpp", "nlohmann_json", "cpp-httplib")
    add_syslinks("pthread")

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
