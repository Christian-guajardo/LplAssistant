/**
 * @file HostInference.hpp
 * @brief Hosted text generation, with the attention cache reused across turns.
 *
 * Wraps a mature inference runtime, and earns its place by one optimisation: only
 * the suffix of the prompt that differs from the previous turn is re-decoded, the
 * common prefix being kept in the cache. On a small model that is the difference
 * between a conversation and a wait.
 * 
 * Also exposes grammar-constrained generation, which is what makes a small local
 * model usable as a decision maker: the sampler cannot emit a token the grammar
 * forbids, so malformed output is unrepresentable rather than merely unlikely. The
 * sampler chain is rebuilt per constrained call on purpose — a repetition penalty
 * would break JSON syntax.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_BACKEND_HOSTINFERENCE_HPP
#    define LPL_BACKEND_HOSTINFERENCE_HPP

#    include <lpl/infer/Types.hpp>

#    include <functional>
#    include <memory>
#    include <string>
#    include <vector>

namespace lpl::backend {

// Moteur de génération llama.cpp avec réutilisation du KV cache :
// seul le suffixe nouveau du prompt est re-décodé à chaque tour.
class HostInference {
public:
    HostInference(const std::string& modelPath, int contextLength, int threadCount);
    ~HostInference();

    // Génère la réponse assistant pour l'historique donné.
    // onToken est appelé pour chaque morceau de texte produit (streaming).
    // shouldCancel (optionnel) est consulté avant chaque token : s'il renvoie
    // true, la génération s'arrête et le texte partiel est retourné (barge-in).
    std::string generate(const std::vector<infer::ChatMessage>& history,
                         int maximumNewTokens,
                         const std::function<void(const std::string&)>& onToken,
                         const std::function<bool()>& shouldCancel = nullptr);

    // Génération contrainte par une grammaire GBNF (sortie JSON garantie pour
    // la machine à états du deep research). Chaîne de samplers fraîche par
    // appel : grammaire + température basse, sans pénalité de répétition
    // (elle casserait la syntaxe JSON).
    std::string generateConstrained(const std::vector<infer::ChatMessage>& history,
                                     const std::string& gbnf_grammar,
                                     int maximumNewTokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace lpl::backend

#endif // LPL_BACKEND_HOSTINFERENCE_HPP
