/**
 * @file LanguageModelClient.hpp
 * @brief Two backs for the same model access.
 *
 * A remote server with parallel slots and per-request grammars, or the in-process
 * model the assistant already holds. Same interface, so the engine never knows which
 * one it has — and the remote back is also what makes the whole loop testable against
 * a simulated stack.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_LANGUAGEMODELCLIENT_HPP
#    define LPL_RESEARCH_LANGUAGEMODELCLIENT_HPP

#    include <memory>
#    include <string>

// Forward-declared rather than included: this module only needs to hold a
// reference, and the hosted inference header drags in the whole runtime.
namespace lpl::backend {
class HostInference;
} // namespace lpl::backend

// Accès LLM du module recherche, à deux dos :
//  - RemoteLanguageModelClient : llama-server (slots parallèles, grammaire par requête,
//    cache_prompt) — la cible de l'architecture, et le dos testable au mock ;
//  - LocalLanguageModelClient : réutilise le HostInference in-process déjà chargé par l'assistant
//    (grammaire via generateConstrained) — zéro process supplémentaire.
namespace lpl::research {

struct LanguageModelReply {
    std::string text;
    int         approximateTokens = 0; // prompt + sortie, pour le budget
};

class LanguageModelClient {
public:
    virtual ~LanguageModelClient() = default;
    // `grammar` vide = génération libre (rédaction du rapport).
    virtual LanguageModelReply complete(const std::string& system, const std::string& user,
                              const std::string& grammar, int max_tokens,
                              float temperature) = 0;
};

// llama-server : POST {base_url}/completion (endpoint natif, champ `grammar`).
class RemoteLanguageModelClient : public LanguageModelClient {
public:
    explicit RemoteLanguageModelClient(std::string base_url);
    LanguageModelReply complete(const std::string& system, const std::string& user,
                      const std::string& grammar, int max_tokens,
                      float temperature) override;

private:
    std::string base_url_;
};

class LocalLanguageModelClient : public LanguageModelClient {
public:
    explicit LocalLanguageModelClient(lpl::backend::HostInference& llm);
    LanguageModelReply complete(const std::string& system, const std::string& user,
                      const std::string& grammar, int max_tokens,
                      float temperature) override;

private:
    lpl::backend::HostInference& llm_;
};

// Fabrique : LAPLACE_RESEARCH_LLM_URL défini -> HTTP ; sinon LLM local fourni.
std::unique_ptr<LanguageModelClient> make_llm_client(lpl::backend::HostInference* local_or_null);

} // namespace lpl::research

#endif // LPL_RESEARCH_LANGUAGEMODELCLIENT_HPP
