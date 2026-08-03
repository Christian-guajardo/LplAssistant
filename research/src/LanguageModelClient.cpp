/**
 * @file LanguageModelClient.cpp
 * @brief Implementation of two backs for the same model access.
 *
 *  
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/research/LanguageModelClient.hpp>
#include <lpl/research/WebFetch.hpp>
#include <lpl/backend/HostInference.hpp>

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <stdexcept>

using nlohmann::json;

namespace lpl::research {

namespace {

// Estimation grossière mais stable pour le budget (≈ 4 chars/token fr+en).
int approximateTokens(size_t chars) { return (int)(chars / 4) + 1; }

// Gabarit ChatML (famille Qwen). llama-server /completion attend le prompt brut.
std::string chatml(const std::string& system, const std::string& user) {
    return "<|im_start|>system\n" + system + "<|im_end|>\n" +
           "<|im_start|>user\n" + user + "<|im_end|>\n" +
           "<|im_start|>assistant\n";
}

} // namespace

RemoteLanguageModelClient::RemoteLanguageModelClient(std::string base_url) : base_url_(std::move(base_url)) {
    while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();
}

LanguageModelReply RemoteLanguageModelClient::complete(const std::string& system, const std::string& user,
                                 const std::string& grammar, int max_tokens,
                                 float temperature) {
    json body{
        {"prompt", chatml(system, user)},
        {"n_predict", max_tokens},
        {"temperature", temperature},
        {"cache_prompt", true},
        {"stop", json::array({"<|im_end|>"})},
    };
    if (!grammar.empty()) body["grammar"] = grammar;

    auto r = http_post_json(base_url_ + "/completion", body.dump());
    if (r.status != 200)
        throw std::runtime_error("llama-server: HTTP " + std::to_string(r.status) +
                                 " " + r.error);
    auto j = json::parse(r.body);
    LanguageModelReply reply;
    reply.text = j.value("content", "");
    int evaluated = j.value("tokens_evaluated", 0);
    int predicted = j.value("tokens_predicted", 0);
    reply.approximateTokens = evaluated + predicted;
    if (reply.approximateTokens == 0)
        reply.approximateTokens =
            approximateTokens(system.size() + user.size() + reply.text.size());
    return reply;
}

LocalLanguageModelClient::LocalLanguageModelClient(lpl::backend::HostInference& llm) : llm_(llm) {}

LanguageModelReply LocalLanguageModelClient::complete(const std::string& system, const std::string& user,
                                  const std::string& grammar, int max_tokens,
                                  float /*temperature*/) {
    std::vector<infer::ChatMessage> history{{"system", system}, {"user", user}};
    LanguageModelReply reply;
    reply.text = grammar.empty()
                     ? llm_.generate(history, max_tokens, nullptr)
                     : llm_.generateConstrained(history, grammar, max_tokens);
    reply.approximateTokens =
        approximateTokens(system.size() + user.size() + reply.text.size());
    return reply;
}

std::unique_ptr<LanguageModelClient> make_llm_client(lpl::backend::HostInference* local_or_null) {
    if (const char* url = std::getenv("LAPLACE_RESEARCH_LLM_URL"); url && *url)
        return std::make_unique<RemoteLanguageModelClient>(url);
    if (local_or_null)
        return std::make_unique<LocalLanguageModelClient>(*local_or_null);
    throw std::runtime_error(
        "recherche : ni LAPLACE_RESEARCH_LLM_URL ni LLM local disponible");
}

} // namespace lpl::research
