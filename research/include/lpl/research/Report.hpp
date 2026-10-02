/**
 * @file Report.hpp
 * @brief The final report, written section by section.
 *
 * Never one long generation: a plan of sections first, then each section written
 * separately. Plus quality gates on every cited URL and an explicit statement of the
 * run's limits — a report that hides what it could not verify is worse than a short
 * one.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_REPORT_HPP
#    define LPL_RESEARCH_REPORT_HPP

#    include <string>
#    include <vector>

#    include <lpl/research/LanguageModelClient.hpp>
#    include <lpl/research/State.hpp>

namespace lpl::research {
struct ResearchOptions;

/**
 * @brief The heading under which a report lists its findings, one per line.
 *
 * A CONTRACT with a reader in another repository, which is why it is a named constant
 * rather than a literal buried in the assembler. `harvest::ResearchReport` in LplKnowledge
 * anchors on this word to tell a findings list apart from the source list below it, and the
 * two lists have the same bullet shape — `- [S<id>] …` — so nothing else distinguishes them.
 *
 * @warning Renaming it renames it in both repositories or it renames it in neither. A reader that
 * silently finds no findings is indistinguishable from a run that found none, so the reader
 * REPORTS the count rather than defaulting it away.
 */
inline constexpr const char* kFindingsHeading = "Constats";

/**
 * @struct UrlGate
 * @brief The outcome of re-checking one cited URL at write time.
 */
struct UrlGate {
    int  sourceId = 0; ///< Which source, by its run-local id.
    bool ok       = false; ///< Whether it still answered.
    int  status   = 0; ///< The HTTP status that decided it.
};

/**
 * @brief Renders the findings section of a report.
 *
 * The learnings are already `[S<id>] text` in @c RunState::knowledge — the engine tagged
 * them with their source when it extracted them. Rendering them verbatim is therefore not
 * an extra claim about anything; it is the report keeping structure it already had, instead
 * of dissolving it into prose and leaving a second reader to guess it back out.
 *
 * @param st The run.
 * @return The markdown section, empty when the run learned nothing.
 */
[[nodiscard]] std::string renderFindings(const RunState& st);

/**
 * @brief Assembles the final report from parts that have already been decided.
 *
 * Split out of @ref writeReport into its own translation unit, and NOT for tidiness. Two
 * consequences are the point:
 *
 *   - **it can be tested at all.** Everything else in `writeReport` needs a language model
 *     to plan and write sections, and a network to re-check URLs. Those live in
 *     `Report.cpp`, which drags in llama.cpp and cpp-httplib; this file drags in nothing.
 *     A test target can compile it alone, so the exact bytes another repository parses are
 *     under test without an inference run — and `validate.sh` can feed the writer's REAL
 *     output to the reader instead of to a hand-typed imitation of it.
 *   - **it is deterministic.** The timestamp is a PARAMETER rather than a call to
 *     `std::time`, so the same run assembles the same bytes twice. A wall clock inside a
 *     function whose output is compared is the same mistake `InferenceBudget` avoided by
 *     counting turns rather than milliseconds.
 *
 * @param st      The run.
 * @param body    The prose sections, already written, in order.
 * @param gates   The URL re-checks, one per source that was read successfully.
 * @param offline True when every check failed, so no mark means anything.
 * @param stamp   The `YYYY-MM-DD HH:MM` the byline should carry.
 * @return The whole report.
 */
[[nodiscard]] std::string assembleReport(const RunState& st, const std::string& body,
                                         const std::vector<UrlGate>& gates, bool offline,
                                         const std::string& stamp);

// Rédige le rapport final dans <run_dir>/report.md :
// plan de sections généré, rédaction progressive section par section
// (star-hengxing : jamais un long rapport en un seul appel), constats
// structurés, références, quality gates URL (HEAD -> GET, offline-aware),
// diagnostics providers et limites. Renvoie le chemin du fichier.
std::string writeReport(RunState& st, LanguageModelClient& llm, const ResearchOptions& opt);

} // namespace lpl::research

#endif // LPL_RESEARCH_REPORT_HPP
