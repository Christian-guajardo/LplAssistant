/**
 * @file Settings.hpp
 * @brief Central configuration, overridden by the environment.
 *
 * Every field is overridable through an environment variable so that changing a
 * model or a threshold never requires a recompile. Defaults describe a working
 * development machine, not a minimum — a field left at its default should still
 * produce a usable assistant.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_BACKEND_SETTINGS_HPP
#    define LPL_BACKEND_SETTINGS_HPP

#    include <string>

// Configuration centrale de Laplace. Valeurs surchargées par variables
// d'environnement (LAPLACE_*) pour éviter toute recompilation.
namespace lpl::backend {

struct Settings {
    std::string databaseConnection      = "host=127.0.0.1 port=5432 dbname=laplace user=laplace password=laplace";
    std::string languageModelPath    = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    std::string embeddingModelPath  = "models/bge-m3-q4_k_m.gguf";
    std::string speechModelPath    = "models/ggml-base.bin";
    int         contextLength        = 4096;   // contexte LLM
    int         threadCount    = 8;      // 16 coeurs logiques -> 8 threads physiques
    int         maximumNewTokens = 512;
    int         topMemoryCount = 5;    // souvenirs injectés dans le prompt
    float       minimumSimilarity = 0.45f; // seuil cosinus pour le retrieval

    static Settings fromEnvironment();
};

} // namespace lpl::backend

#endif // LPL_BACKEND_SETTINGS_HPP
