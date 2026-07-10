#pragma once
#include <memory>
#include <string>

#include "ccr.h"
#include "llm_client.h"
#include "search.h"
#include "state.h"

// Moteur de deep research : boucle agent à actions typées
// (search / read / reflect / answer) sous contrainte de grammaire GBNF,
// avec gating dynamique, file de lacunes, budget de tokens, beast mode,
// évaluateur de réponse et checkpoint à chaque pas.
namespace laplace::research {

struct ResearchOptions {
    std::string   out_root = "research_runs";
    SearchOptions search;
    long token_budget           = 60000; // tokens approx. (prompt + sortie)
    int  max_steps              = 30;
    int  max_gaps               = 12;
    int  max_sources            = 40;
    int  skim_chars             = 6000;  // vue CCR injectée par lecture
    int  max_learnings_per_read = 4;
    int  max_answer_attempts    = 3;     // au-delà : la réponse est acceptée
    bool verbose                = true;

    // LAPLACE_RESEARCH_* : BUDGET, MAX_STEPS, DIR, PROVIDERS (csv),
    // LAPLACE_SEARXNG_URL, GITHUB_TOKEN.
    static ResearchOptions from_env();
};

class Engine {
public:
    Engine(LlmClient& llm, ResearchOptions opt);

    // Lance un nouveau run ; renvoie le chemin du rapport markdown.
    std::string run(const std::string& topic, const std::string& guidance = "");

    // Reprend un run interrompu là où le checkpoint s'était arrêté.
    std::string resume(const std::string& run_dir);

private:
    std::string execute(RunState& st, Ccr& ccr);
    void        loop(RunState& st, Ccr& ccr);
    void        do_search(RunState& st, const std::string& query);
    void        do_read(RunState& st, Ccr& ccr, const std::string& arg);
    void        do_reflect(RunState& st);
    bool        do_answer(RunState& st); // true = réponse acceptée
    void        log(const std::string& msg) const;

    LlmClient&      llm_;
    ResearchOptions opt_;
};

} // namespace laplace::research
