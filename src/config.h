#pragma once
#include <string>

// Configuration centrale de Laplace. Valeurs surchargées par variables
// d'environnement (LAPLACE_*) pour éviter toute recompilation.
namespace laplace {

struct Config {
    std::string db_conn      = "host=127.0.0.1 port=5432 dbname=laplace user=laplace password=laplace";
    std::string llm_model    = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    std::string embed_model  = "models/bge-m3-q4_k_m.gguf";
    std::string stt_model    = "models/ggml-base.bin";
    int         n_ctx        = 4096;   // contexte LLM
    int         n_threads    = 8;      // 16 coeurs logiques -> 8 threads physiques
    int         max_new_tokens = 512;
    int         top_k_memories = 5;    // souvenirs injectés dans le prompt
    float       min_similarity = 0.45f; // seuil cosinus pour le retrieval

    static Config from_env();
};

} // namespace laplace
