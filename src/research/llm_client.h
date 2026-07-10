#pragma once
#include <memory>
#include <string>

namespace laplace {
class Llm; // src/llm.h
}

// Accès LLM du module recherche, à deux dos :
//  - HttpLlmClient : llama-server (slots parallèles, grammaire par requête,
//    cache_prompt) — la cible de l'architecture, et le dos testable au mock ;
//  - LocalLlmClient : réutilise le Llm in-process déjà chargé par l'assistant
//    (grammaire via generate_constrained) — zéro process supplémentaire.
namespace laplace::research {

struct LlmReply {
    std::string text;
    int         approx_tokens = 0; // prompt + sortie, pour le budget
};

class LlmClient {
public:
    virtual ~LlmClient() = default;
    // `grammar` vide = génération libre (rédaction du rapport).
    virtual LlmReply complete(const std::string& system, const std::string& user,
                              const std::string& grammar, int max_tokens,
                              float temperature) = 0;
};

// llama-server : POST {base_url}/completion (endpoint natif, champ `grammar`).
class HttpLlmClient : public LlmClient {
public:
    explicit HttpLlmClient(std::string base_url);
    LlmReply complete(const std::string& system, const std::string& user,
                      const std::string& grammar, int max_tokens,
                      float temperature) override;

private:
    std::string base_url_;
};

class LocalLlmClient : public LlmClient {
public:
    explicit LocalLlmClient(laplace::Llm& llm);
    LlmReply complete(const std::string& system, const std::string& user,
                      const std::string& grammar, int max_tokens,
                      float temperature) override;

private:
    laplace::Llm& llm_;
};

// Fabrique : LAPLACE_RESEARCH_LLM_URL défini -> HTTP ; sinon LLM local fourni.
std::unique_ptr<LlmClient> make_llm_client(laplace::Llm* local_or_null);

} // namespace laplace::research
