#include "llm_client.h"
#include "http.h"
#include "../llm.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <stdexcept>

using nlohmann::json;

namespace laplace::research {

namespace {

// Estimation grossière mais stable pour le budget (≈ 4 chars/token fr+en).
int approx_tokens(size_t chars) { return (int)(chars / 4) + 1; }

// Gabarit ChatML (famille Qwen). llama-server /completion attend le prompt brut.
std::string chatml(const std::string& system, const std::string& user) {
    return "<|im_start|>system\n" + system + "<|im_end|>\n" +
           "<|im_start|>user\n" + user + "<|im_end|>\n" +
           "<|im_start|>assistant\n";
}

} // namespace

HttpLlmClient::HttpLlmClient(std::string base_url) : base_url_(std::move(base_url)) {
    while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();
}

LlmReply HttpLlmClient::complete(const std::string& system, const std::string& user,
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
    LlmReply reply;
    reply.text = j.value("content", "");
    int evaluated = j.value("tokens_evaluated", 0);
    int predicted = j.value("tokens_predicted", 0);
    reply.approx_tokens = evaluated + predicted;
    if (reply.approx_tokens == 0)
        reply.approx_tokens =
            approx_tokens(system.size() + user.size() + reply.text.size());
    return reply;
}

LocalLlmClient::LocalLlmClient(laplace::Llm& llm) : llm_(llm) {}

LlmReply LocalLlmClient::complete(const std::string& system, const std::string& user,
                                  const std::string& grammar, int max_tokens,
                                  float /*temperature*/) {
    std::vector<ChatMessage> history{{"system", system}, {"user", user}};
    LlmReply reply;
    reply.text = grammar.empty()
                     ? llm_.generate(history, max_tokens, nullptr)
                     : llm_.generate_constrained(history, grammar, max_tokens);
    reply.approx_tokens =
        approx_tokens(system.size() + user.size() + reply.text.size());
    return reply;
}

std::unique_ptr<LlmClient> make_llm_client(laplace::Llm* local_or_null) {
    if (const char* url = std::getenv("LAPLACE_RESEARCH_LLM_URL"); url && *url)
        return std::make_unique<HttpLlmClient>(url);
    if (local_or_null)
        return std::make_unique<LocalLlmClient>(*local_or_null);
    throw std::runtime_error(
        "recherche : ni LAPLACE_RESEARCH_LLM_URL ni LLM local disponible");
}

} // namespace laplace::research
