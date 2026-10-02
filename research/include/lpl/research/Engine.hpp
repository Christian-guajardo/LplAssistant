/**
 * @file Engine.hpp
 * @brief The agent loop: search, read, reflect, answer.
 *
 * Typed actions under a regenerated grammar, with dynamic gating, a queue of named
 * gaps, a token budget, a last-resort mode when the budget runs low, and an
 * evaluator that can reject an answer and send the loop back out. Every step
 * checkpoints.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_ENGINE_HPP
#    define LPL_RESEARCH_ENGINE_HPP

#    include <memory>
#    include <string>

#    include <lpl/research/CompressCacheRetrieve.hpp>
#    include <lpl/research/LanguageModelClient.hpp>
#    include <lpl/research/Search.hpp>
#    include <lpl/research/State.hpp>

// Moteur de deep research : boucle agent à actions typées
// (search / read / reflect / answer) sous contrainte de grammaire GBNF,
// avec gating dynamique, file de lacunes, budget de tokens, beast mode,
// évaluateur de réponse et checkpoint à chaque pas.
namespace lpl::research {

struct ResearchOptions {
    std::string   outputRoot = "research_runs";
    SearchOptions search;
    long tokenBudget           = 60000; // tokens approx. (prompt + sortie)
    int  maximumSteps              = 30;
    int  maximumGaps               = 12;
    int  maximumSources            = 40;
    int  skimCharacters             = 6000;  // vue CCR injectée par lecture
    int  maximumLearningsPerRead = 4;
    int  maximumAnswerAttempts    = 3;     // au-delà : la réponse est acceptée
    bool verbose                = true;
    // Email requis par l'API Unpaywall (LAPLACE_UNPAYWALL_EMAIL). Vide => la
    // résolution DOI -> Open Access est désactivée.
    std::string unpaywall_email;

    // LAPLACE_RESEARCH_* : BUDGET, MAX_STEPS, DIR, PROVIDERS (csv),
    // LAPLACE_SEARXNG_URL, GITHUB_TOKEN.
    static ResearchOptions fromEnvironment();
};

class Engine {
public:
    Engine(LanguageModelClient& llm, ResearchOptions opt);

    // Lance un nouveau run ; renvoie le chemin du rapport markdown.
    std::string run(const std::string& topic, const std::string& guidance = "");

    // Reprend un run interrompu là où le checkpoint s'était arrêté.
    std::string resume(const std::string& run_dir);

private:
    std::string execute(RunState& st, CompressCacheRetrieve& ccr);
    void        loop(RunState& st, CompressCacheRetrieve& ccr);
    void        do_search(RunState& st, const std::string& query);
    void        do_read(RunState& st, CompressCacheRetrieve& ccr, const std::string& arg);
    void        do_reflect(RunState& st);
    bool        do_answer(RunState& st); // true = réponse acceptée
    void        log(const std::string& msg) const;

    LanguageModelClient&      llm_;
    ResearchOptions opt_;
};

} // namespace lpl::research

#endif // LPL_RESEARCH_ENGINE_HPP
