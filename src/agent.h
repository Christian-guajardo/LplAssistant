#pragma once
#include "db.h"
#include "embedder.h"
#include "llm.h"
#include <functional>
#include <string>
#include <vector>

namespace laplace {

// Orchestrateur : Composite Retrieval -> prompt -> génération -> mémorisation.
class Agent {
public:
    Agent(Db& db, Embedder& embedder, Llm& llm,
          int top_k, float min_similarity, int max_new_tokens);

    // Traite un tour de conversation. Streaming via on_token.
    std::string ask(const std::string& user_input,
                    const std::function<void(const std::string&)>& on_token);

private:
    Db&       db;
    Embedder& embedder;
    Llm&      llm;
    int       top_k;
    float     min_similarity;
    int       max_new_tokens;
    std::vector<ChatMessage> session; // historique du dialogue courant
};

} // namespace laplace
