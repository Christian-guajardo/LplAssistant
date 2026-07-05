#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace laplace {

struct ChatMessage {
    std::string role; // "system" | "user" | "assistant"
    std::string content;
};

// Moteur de génération llama.cpp avec réutilisation du KV cache :
// seul le suffixe nouveau du prompt est re-décodé à chaque tour.
class Llm {
public:
    Llm(const std::string& model_path, int n_ctx, int n_threads);
    ~Llm();

    // Génère la réponse assistant pour l'historique donné.
    // on_token est appelé pour chaque morceau de texte produit (streaming).
    std::string generate(const std::vector<ChatMessage>& history,
                         int max_new_tokens,
                         const std::function<void(const std::string&)>& on_token);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace laplace
