#include "agent.h"

namespace laplace {

static const char* SYSTEM_PROMPT =
    "Tu es Laplace, un assistant personnel local, privé et francophone. "
    "Tu réponds de façon concise, utile et directe. "
    "Des souvenirs pertinents issus de ta mémoire long terme peuvent t'être "
    "fournis : utilise-les seulement s'ils sont pertinents pour la question.";

Agent::Agent(Db& db_, Embedder& embedder_, Llm& llm_,
             int top_k_, float min_similarity_, int max_new_tokens_)
    : db(db_), embedder(embedder_), llm(llm_),
      top_k(top_k_), min_similarity(min_similarity_),
      max_new_tokens(max_new_tokens_) {}

std::string Agent::ask(const std::string& user_input,
                       const std::function<void(const std::string&)>& on_token) {
    // 1. Retrieval : embedding de la question puis recherche composite.
    std::vector<float> qvec = embedder.embed(user_input);
    std::vector<MemoryRow> memories = db.search(qvec, top_k, min_similarity);

    // 2. Construction du prompt : système + souvenirs + dialogue courant.
    std::string sys = SYSTEM_PROMPT;
    if (!memories.empty()) {
        sys += "\n\nSouvenirs pertinents (mémoire long terme) :";
        for (const auto& m : memories)
            sys += "\n- [" + m.created_at.substr(0, 10) + "] " + m.content;
    }

    // Économie de tokens : ne garder que les 8 derniers tours en session,
    // le reste vit dans la mémoire long terme (Composite Retrieval).
    if (session.size() > 16)
        session.erase(session.begin(), session.end() - 16);

    std::vector<ChatMessage> history;
    history.push_back({"system", sys});
    history.insert(history.end(), session.begin(), session.end());
    history.push_back({"user", user_input});

    // 3. Génération.
    std::string answer = llm.generate(history, max_new_tokens, on_token);

    // 4. Mémorisation de l'échange (embed sur le tour complet).
    session.push_back({"user", user_input});
    session.push_back({"assistant", answer});
    std::string memory_text = "Utilisateur: " + user_input + "\nLaplace: " + answer;
    db.add_memory("conversation", memory_text, embedder.embed(memory_text));

    return answer;
}

} // namespace laplace
