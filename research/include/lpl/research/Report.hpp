/**
 * @file Report.hpp
 * @brief The final report, written section by section.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_REPORT_HPP
#    define LPL_RESEARCH_REPORT_HPP

#    include <string>

#    include <lpl/research/LanguageModelClient.hpp>
#    include <lpl/research/State.hpp>

namespace lpl::research {
struct ResearchOptions;

/**
 * @brief Writes the final report of a run to `<run_dir>/report.md`.
 *
 * Never one long generation (the star-hengxing rule): the model plans at most five section
 * titles, then writes each section in a call of its own, and the run is checkpointed after each
 * one. An unreadable plan gives a report without detailed sections. Every source that was read
 * successfully is re-checked over HTTP, HEAD first and GET as a fallback. The layout, from the
 * findings list to the provider diagnostics and the run's limits, is @ref assembleReport: a
 * report that hides what it could not verify is worse than a short one.
 *
 * @param st  The run. Its token count grows by what planning and writing cost.
 * @param llm The model that plans and writes the sections.
 * @param opt Not read.
 * @return The path of the written report.
 * @throws std::runtime_error Naming the path, when the report could not be written.
 */
[[nodiscard]] std::string writeReport(RunState& st, LanguageModelClient& llm, const ResearchOptions& opt);

} // namespace lpl::research

#endif // LPL_RESEARCH_REPORT_HPP
