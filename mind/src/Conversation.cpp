/**
 * @file Conversation.cpp
 * @brief Implementation of a conversation turn.
 *
 *  
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/mind/Conversation.hpp>

namespace lpl::mind {

static const char* SYSTEM_PROMPT =
    "Tu es Laplace, un assistant personnel local, privé et francophone. "
    "Tu réponds de façon concise, utile et directe. "
    "Des souvenirs pertinents issus de ta mémoire long terme peuvent t'être "
    "fournis : utilise-les seulement s'ils sont pertinents pour la question.";

Conversation::Conversation(backend::VectorStore &store, backend::Embedder &embedder,
                           backend::HostInference &model, int topMemoryCount,
                           float minimumSimilarity, int maximumNewTokens)
    : _store(store), _embedder(embedder), _model(model), _topMemoryCount(topMemoryCount),
      _minimumSimilarity(minimumSimilarity), _maximumNewTokens(maximumNewTokens)
{
}

std::string Conversation::ask(const std::string& userInput,
                       const std::function<void(const std::string&)>& onToken,
                       const std::function<bool()>& shouldCancel) {
    // 1. Retrieval : embedding de la question puis recherche composite.
    std::vector<float> qvec = _embedder.embed(userInput);
    std::vector<backend::MemoryRecord> memories = _store.search(qvec, _topMemoryCount, _minimumSimilarity);

    // 2. Construction du prompt : système + souvenirs + dialogue courant.
    std::string sys = SYSTEM_PROMPT;
    if (!memories.empty()) {
        sys += "\n\nSouvenirs pertinents (mémoire long terme) :";
        for (const auto& m : memories)
            sys += "\n- [" + m.createdAt.substr(0, 10) + "] " + m.content;
    }

    // Économie de tokens : ne garder que les 8 derniers tours en _session,
    // le reste vit dans la mémoire long terme (Composite Retrieval).
    if (_session.size() > 16)
        _session.erase(_session.begin(), _session.end() - 16);

    std::vector<infer::ChatMessage> history;
    history.push_back({"system", sys});
    history.insert(history.end(), _session.begin(), _session.end());
    history.push_back({"user", userInput});

    // 3. Génération.
    std::string answer = _model.generate(history, _maximumNewTokens, onToken,
                                      shouldCancel);

    // 4. Mémorisation de l'échange (embed sur le tour complet).
    _session.push_back({"user", userInput});
    _session.push_back({"assistant", answer});
    std::string memory_text = "Utilisateur: " + userInput + "\nLaplace: " + answer;
    _store.addMemory("conversation", memory_text, _embedder.embed(memory_text));

    return answer;
}

} // namespace lpl::mind
