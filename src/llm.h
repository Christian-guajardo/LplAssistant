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
    // should_cancel (optionnel) est consulté avant chaque token : s'il renvoie
    // true, la génération s'arrête et le texte partiel est retourné (barge-in).
    std::string generate(const std::vector<ChatMessage>& history,
                         int max_new_tokens,
                         const std::function<void(const std::string&)>& on_token,
                         const std::function<bool()>& should_cancel = nullptr);

    // Génération contrainte par une grammaire GBNF (sortie JSON garantie pour
    // la machine à états du deep research). Chaîne de samplers fraîche par
    // appel : grammaire + température basse, sans pénalité de répétition
    // (elle casserait la syntaxe JSON).
    std::string generate_constrained(const std::vector<ChatMessage>& history,
                                     const std::string& gbnf_grammar,
                                     int max_new_tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace laplace
