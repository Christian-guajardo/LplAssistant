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

#    include <lpl/research/LanguageModelClient.hpp>
#    include <lpl/research/State.hpp>

namespace lpl::research {
struct ResearchOptions;

// Rédige le rapport final dans <run_dir>/report.md :
// plan de sections généré, rédaction progressive section par section
// (star-hengxing : jamais un long rapport en un seul appel), références,
// quality gates URL (HEAD -> GET, offline-aware), diagnostics providers
// et limites. Renvoie le chemin du fichier.
std::string writeReport(RunState& st, LanguageModelClient& llm, const ResearchOptions& opt);

} // namespace lpl::research

#endif // LPL_RESEARCH_REPORT_HPP
