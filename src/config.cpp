#include "config.h"
#include <cstdlib>

namespace laplace {

static void env_str(const char* name, std::string& out) {
    if (const char* v = std::getenv(name)) out = v;
}
static void env_int(const char* name, int& out) {
    if (const char* v = std::getenv(name)) out = std::atoi(v);
}

Config Config::from_env() {
    Config c;
    env_str("LAPLACE_DB",          c.db_conn);
    env_str("LAPLACE_LLM_MODEL",   c.llm_model);
    env_str("LAPLACE_EMBED_MODEL", c.embed_model);
    env_str("LAPLACE_STT_MODEL",   c.stt_model);
    env_int("LAPLACE_N_CTX",       c.n_ctx);
    env_int("LAPLACE_N_THREADS",   c.n_threads);
    env_int("LAPLACE_MAX_TOKENS",  c.max_new_tokens);
    return c;
}

} // namespace laplace
