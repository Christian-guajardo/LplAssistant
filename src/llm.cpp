#include "llm.h"
#include <llama.h>
#include <cstring>
#include <stdexcept>

namespace laplace {

struct Llm::Impl {
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    llama_sampler* smpl  = nullptr;
    int            n_ctx = 0;
    int            n_batch = 512;
    std::vector<llama_token> cache; // tokens déjà présents dans le KV cache

    ~Impl() {
        if (smpl)  llama_sampler_free(smpl);
        if (ctx)   llama_free(ctx);
        if (model) llama_free_model(model);
    }
};

Llm::Llm(const std::string& model_path, int n_ctx, int n_threads)
    : impl(new Impl()) {
    llama_model_params mp = llama_model_default_params();
    impl->model = llama_load_model_from_file(model_path.c_str(), mp);
    if (!impl->model)
        throw std::runtime_error("Llm: impossible de charger " + model_path);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = impl->n_batch;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    impl->ctx = llama_new_context_with_model(impl->model, cp);
    if (!impl->ctx)
        throw std::runtime_error("Llm: échec création du contexte");
    impl->n_ctx = n_ctx;

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    impl->smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(impl->smpl, llama_sampler_init_penalties(
        llama_n_vocab(impl->model), llama_token_eos(impl->model),
        llama_token_nl(impl->model),
        /*last_n*/ 256, /*repeat*/ 1.15f, /*freq*/ 0.0f, /*present*/ 0.0f,
        /*penalize_nl*/ false, /*ignore_eos*/ false));
    llama_sampler_chain_add(impl->smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(impl->smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(impl->smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(impl->smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

Llm::~Llm() = default;

static std::string apply_template(llama_model* model,
                                  const std::vector<ChatMessage>& history) {
    std::vector<llama_chat_message> msgs;
    msgs.reserve(history.size());
    size_t total = 0;
    for (const auto& m : history) {
        msgs.push_back({m.role.c_str(), m.content.c_str()});
        total += m.role.size() + m.content.size();
    }
    std::vector<char> buf(total * 2 + 1024);
    int32_t n = llama_chat_apply_template(model, nullptr, msgs.data(), msgs.size(),
                                          true, buf.data(), (int32_t)buf.size());
    if (n < 0) throw std::runtime_error("Llm: échec du chat template");
    if (n > (int32_t)buf.size()) {
        buf.resize(n);
        n = llama_chat_apply_template(model, nullptr, msgs.data(), msgs.size(),
                                      true, buf.data(), (int32_t)buf.size());
    }
    return std::string(buf.data(), n);
}

static std::string token_to_piece(llama_model* model, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(model, tok, buf, sizeof(buf), 0, false);
    return n > 0 ? std::string(buf, n) : std::string();
}

std::string Llm::generate(const std::vector<ChatMessage>& history,
                          int max_new_tokens,
                          const std::function<void(const std::string&)>& on_token) {
    const std::string prompt = apply_template(impl->model, history);

    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(impl->model, prompt.c_str(), (int)prompt.size(),
                           toks.data(), (int)toks.size(), true, true);
    if (n < 0) throw std::runtime_error("Llm: échec de tokenisation");
    toks.resize(n);
    if (n >= impl->n_ctx - max_new_tokens) {
        // Contexte plein : on repart de zéro (l'agent tronque l'historique en amont).
        impl->cache.clear();
        llama_kv_cache_clear(impl->ctx);
    }

    // Réutilisation du KV cache : ne décoder que le suffixe qui diffère.
    size_t common = 0;
    while (common < impl->cache.size() && common + 1 < (size_t)n &&
           impl->cache[common] == toks[common])
        ++common;
    if (common < impl->cache.size()) {
        llama_kv_cache_seq_rm(impl->ctx, 0, (llama_pos)common, -1);
        impl->cache.resize(common);
    }

    // Décodage du suffixe par lots de n_batch.
    for (int i = (int)common; i < n; i += impl->n_batch) {
        int chunk = std::min(impl->n_batch, n - i);
        llama_batch batch = llama_batch_get_one(toks.data() + i, chunk, i, 0);
        if (llama_decode(impl->ctx, batch) != 0)
            throw std::runtime_error("Llm: échec du decode (prompt)");
    }
    impl->cache = toks;

    // Boucle de génération autorégressive.
    std::string out;
    llama_token tok;
    for (int i = 0; i < max_new_tokens; ++i) {
        tok = llama_sampler_sample(impl->smpl, impl->ctx, -1);
        if (llama_token_is_eog(impl->model, tok)) break;
        std::string piece = token_to_piece(impl->model, tok);
        out += piece;
        if (on_token) on_token(piece);
        llama_batch batch = llama_batch_get_one(&tok, 1, (llama_pos)impl->cache.size(), 0);
        if (llama_decode(impl->ctx, batch) != 0)
            throw std::runtime_error("Llm: échec du decode (génération)");
        impl->cache.push_back(tok);
        if ((int)impl->cache.size() >= impl->n_ctx - 1) break;
    }
    return out;
}

} // namespace laplace
