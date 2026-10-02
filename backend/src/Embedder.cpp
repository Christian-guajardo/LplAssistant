/**
 * @file Embedder.cpp
 * @brief Implementation of embedding generation.
 *
 *  
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/backend/Embedder.hpp>
#include <llama.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace lpl::backend {

struct Embedder::Impl {
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    int            n_embd = 0;

    ~Impl() {
        if (ctx)   llama_free(ctx);
        if (model) llama_free_model(model);
    }
};

Embedder::Embedder(const std::string& modelPath, int threadCount)
    : impl(new Impl()) {
    llama_model_params mp = llama_model_default_params();
    impl->model = llama_load_model_from_file(modelPath.c_str(), mp);
    if (!impl->model)
        throw std::runtime_error("Embedder: impossible de charger " + modelPath);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 2048;
    cp.n_batch         = 2048;
    cp.n_ubatch        = 2048;
    cp.n_threads       = threadCount;
    cp.n_threads_batch = threadCount;
    cp.embeddings      = true;
    // UNSPECIFIED -> pooling défini par les métadonnées du GGUF (CLS pour bge-m3)
    cp.pooling_type    = LLAMA_POOLING_TYPE_UNSPECIFIED;

    impl->ctx = llama_new_context_with_model(impl->model, cp);
    if (!impl->ctx)
        throw std::runtime_error("Embedder: échec création du contexte");
    impl->n_embd = llama_n_embd(impl->model);
}

Embedder::~Embedder() = default;

int Embedder::dim() const { return impl->n_embd; }

std::vector<float> Embedder::embed(const std::string& text) {
    std::vector<llama_token> tokens(2048);
    int n = llama_tokenize(impl->model, text.c_str(), (int)text.size(),
                           tokens.data(), (int)tokens.size(), true, true);
    if (n < 0) throw std::runtime_error("Embedder: texte trop long");
    tokens.resize(n);

    llama_kv_cache_clear(impl->ctx);
    llama_batch batch = llama_batch_get_one(tokens.data(), n, 0, 0);
    if (llama_decode(impl->ctx, batch) != 0)
        throw std::runtime_error("Embedder: échec du decode");

    const float* emb = llama_get_embeddings_seq(impl->ctx, 0);
    if (!emb) throw std::runtime_error("Embedder: pas d'embedding produit");

    std::vector<float> out(emb, emb + impl->n_embd);
    float norm = 0.f;
    for (float v : out) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0.f) for (float& v : out) v /= norm;
    return out;
}

} // namespace lpl::backend
