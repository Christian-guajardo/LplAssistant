/**
 * @file State.hpp
 * @brief The resumable state of a research run.
 *
 * A checkpoint written at every step, atomically (temporary file then rename), so a
 * crash or a reboot costs one step rather than a run. Phases are monotonic; the
 * agent loop lives entirely inside the researching phase.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_STATE_HPP
#    define LPL_RESEARCH_STATE_HPP

#    include <deque>
#    include <map>
#    include <string>
#    include <vector>

#    include <lpl/research/Search.hpp> // ProviderDiagnostic

// État persistant d'un run de recherche : checkpoint JSON écrit à chaque pas
// (écriture atomique tmp+rename), reprise après crash/reboot via load().
// Phases monotones (portage du state.py de star-hengxing) ; la boucle agent
// vit entièrement dans la phase `researching`.
namespace lpl::research {

enum class Phase { initialized, planned, researching, answered, reported, complete, error };

const char* phaseName(Phase p);
Phase       phaseFromName(const std::string& name);

struct Source {
    int         id = 0;
    std::string url, title, snippet, provider;
    std::string cacheId;       // id CCR une fois lue
    bool        read    = false;
    bool        fetchOk = true; // false = lecture tentée et échouée
};

struct FailedAttempt {
    std::string answer;
    std::string reason; // critique de l'évaluateur
};

struct RunState {
    int         version = 1;
    std::string topic;
    std::string guidance;   // précisions utilisateur (questions de cadrage)
    std::string run_dir;
    Phase       phase = Phase::initialized;

    // Boucle agent
    std::deque<std::string>  gaps;        // questions ouvertes, front = courante
    std::vector<std::string> knowledge;   // learnings taggés [S<id>]
    std::vector<Source>      sources;
    std::vector<std::string> queries_done;
    std::vector<FailedAttempt> attempts;
    std::map<std::string, ProviderDiagnostic> diags;

    int  step         = 0;
    long tokens_used  = 0;
    long tokenBudget = 60000;
    bool beast_mode   = false;

    std::string final_answer;
    std::string error;

    // Transition monotone (retour en arrière refusé, error accessible de partout).
    void to_phase(Phase target);

    std::string to_json() const;
    static RunState from_json(const std::string& json_text);
};

// Checkpoint sur disque : <run_dir>/state.json
void     save_state(const RunState& st);
RunState load_state(const std::string& run_dir);
bool     state_exists(const std::string& run_dir);

} // namespace lpl::research

#endif // LPL_RESEARCH_STATE_HPP
