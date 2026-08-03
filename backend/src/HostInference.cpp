/**
 * @file HostInference.cpp
 * @brief Implementation of hosted generation and cache reuse.
 *
 *  
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/backend/HostInference.hpp>
#include <llama.h>
#include <cstring>
#include <stdexcept>

namespace lpl::backend {

struct HostInference::Impl {
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    llama_sampler* smpl  = nullptr;
    int            contextLength = 0;
    int            n_batch = 512;
    std::vector<llama_token> cache; // tokens déjà présents dans le KV cache

    ~Impl() {
        if (smpl)  llama_sampler_free(smpl);
        if (ctx)   llama_free(ctx);
        if (model) llama_free_model(model);
    }
};

HostInference::HostInference(const std::string& modelPath, int contextLength, int threadCount)
    : impl(new Impl()) {
    llama_model_params mp = llama_model_default_params();
    impl->model = llama_load_model_from_file(modelPath.c_str(), mp);
    if (!impl->model)
        throw std::runtime_error("HostInference: impossible de charger " + modelPath);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = contextLength;
    cp.n_batch         = impl->n_batch;
    cp.n_threads       = threadCount;
    cp.n_threads_batch = threadCount;
    impl->ctx = llama_new_context_with_model(impl->model, cp);
    if (!impl->ctx)
        throw std::runtime_error("HostInference: échec création du contexte");
    impl->contextLength = contextLength;

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

HostInference::~HostInference() = default;

static std::string apply_template(llama_model* model,
                                  const std::vector<infer::ChatMessage>& history) {
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
    if (n < 0) throw std::runtime_error("HostInference: échec du chat template");
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

std::string HostInference::generateConstrained(const std::vector<infer::ChatMessage>& history,
                                      const std::string& gbnf_grammar,
                                      int maximumNewTokens) {
    const std::string prompt = apply_template(impl->model, history);

    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(impl->model, prompt.c_str(), (int)prompt.size(),
                           toks.data(), (int)toks.size(), true, true);
    if (n < 0) throw std::runtime_error("HostInference: échec de tokenisation");
    toks.resize(n);
    if (n >= impl->contextLength - maximumNewTokens) {
        impl->cache.clear();
        llama_kv_cache_clear(impl->ctx);
    }

    // Même réutilisation de préfixe KV que generate().
    size_t common = 0;
    while (common < impl->cache.size() && common + 1 < (size_t)n &&
           impl->cache[common] == toks[common])
        ++common;
    if (common < impl->cache.size()) {
        llama_kv_cache_seq_rm(impl->ctx, 0, (llama_pos)common, -1);
        impl->cache.resize(common);
    }
    for (int i = (int)common; i < n; i += impl->n_batch) {
        int chunk = std::min(impl->n_batch, n - i);
        llama_batch batch = llama_batch_get_one(toks.data() + i, chunk, i, 0);
        if (llama_decode(impl->ctx, batch) != 0)
            throw std::runtime_error("HostInference: échec du decode (prompt)");
    }
    impl->cache = toks;

    // Chaîne dédiée : grammaire (masque à état) puis greedy. Greedy prend
    // l'argmax des logits NON masqués — déterministe, exactement ce qu'on veut
    // pour du JSON contraint, et sans dépendre des champs de probabilité `p`
    // (que `dist` exigerait via un softmax préalable absent ici).
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_grammar(
        impl->model, gbnf_grammar.c_str(), "root"));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Échantillonnage manuel (apply + accept séparés) plutôt que
    // llama_sampler_sample() : ce dernier accepte AUSSI le token EOG dans la
    // grammaire, et en b3775 accepter EOG sur une grammaire dont le JSON vient
    // d'être complété touche GGML_ASSERT(!stacks.empty()) et abort. On casse la
    // boucle sur EOG AVANT tout accept, de sorte que la grammaire ne reçoit que
    // des tokens qu'elle peut réellement consumer.
    const int n_vocab = llama_n_vocab(impl->model);
    std::vector<llama_token_data> cand(n_vocab);
    std::string out;
    for (int i = 0; i < maximumNewTokens; ++i) {
        const float* logits = llama_get_logits_ith(impl->ctx, -1);
        for (int t = 0; t < n_vocab; ++t)
            cand[t] = llama_token_data{t, logits[t], 0.0f};
        llama_token_data_array cur_p{cand.data(), (size_t)n_vocab, -1, false};
        llama_sampler_apply(smpl, &cur_p);
        llama_token tok = cur_p.data[cur_p.selected].id;
        if (llama_token_is_eog(impl->model, tok)) break;
        llama_sampler_accept(smpl, tok);
        out += token_to_piece(impl->model, tok);
        llama_batch batch = llama_batch_get_one(&tok, 1, (llama_pos)impl->cache.size(), 0);
        if (llama_decode(impl->ctx, batch) != 0) {
            llama_sampler_free(smpl);
            throw std::runtime_error("HostInference: échec du decode (génération)");
        }
        impl->cache.push_back(tok);
        if ((int)impl->cache.size() >= impl->contextLength - 1) break;
    }
    llama_sampler_free(smpl);
    return out;
}

std::string HostInference::generate(const std::vector<infer::ChatMessage>& history,
                          int maximumNewTokens,
                          const std::function<void(const std::string&)>& onToken,
                          const std::function<bool()>& shouldCancel) {
    const std::string prompt = apply_template(impl->model, history);

    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(impl->model, prompt.c_str(), (int)prompt.size(),
                           toks.data(), (int)toks.size(), true, true);
    if (n < 0) throw std::runtime_error("HostInference: échec de tokenisation");
    toks.resize(n);
    if (n >= impl->contextLength - maximumNewTokens) {
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
            throw std::runtime_error("HostInference: échec du decode (prompt)");
    }
    impl->cache = toks;

    // Boucle de génération autorégressive.
    std::string out;
    llama_token tok;
    for (int i = 0; i < maximumNewTokens; ++i) {
        // Barge-in : une nouvelle demande (ou un « stop ») annule ce tour.
        if (shouldCancel && shouldCancel()) break;
        tok = llama_sampler_sample(impl->smpl, impl->ctx, -1);
        if (llama_token_is_eog(impl->model, tok)) break;
        std::string piece = token_to_piece(impl->model, tok);
        out += piece;
        if (onToken) onToken(piece);
        llama_batch batch = llama_batch_get_one(&tok, 1, (llama_pos)impl->cache.size(), 0);
        if (llama_decode(impl->ctx, batch) != 0)
            throw std::runtime_error("HostInference: échec du decode (génération)");
        impl->cache.push_back(tok);
        if ((int)impl->cache.size() >= impl->contextLength - 1) break;
    }
    return out;
}

} // namespace lpl::backend
