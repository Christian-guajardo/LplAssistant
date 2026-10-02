/**
 * @file Settings.cpp
 * @brief Implementation of environment-driven settings.
 *
 *  
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/backend/Settings.hpp>
#include <cstdlib>

namespace lpl::backend {

static void env_str(const char* name, std::string& out) {
    if (const char* v = std::getenv(name)) out = v;
}
static void env_int(const char* name, int& out) {
    if (const char* v = std::getenv(name)) out = std::atoi(v);
}

Settings Settings::fromEnvironment() {
    Settings c;
    env_str("LAPLACE_DB",          c.databaseConnection);
    env_str("LAPLACE_LLM_MODEL",   c.languageModelPath);
    env_str("LAPLACE_EMBED_MODEL", c.embeddingModelPath);
    env_str("LAPLACE_STT_MODEL",   c.speechModelPath);
    env_int("LAPLACE_N_CTX",       c.contextLength);
    env_int("LAPLACE_N_THREADS",   c.threadCount);
    env_int("LAPLACE_MAX_TOKENS",  c.maximumNewTokens);
    return c;
}

} // namespace lpl::backend
