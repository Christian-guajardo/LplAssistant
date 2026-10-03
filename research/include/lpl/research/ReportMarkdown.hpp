/**
 * @file ReportMarkdown.hpp
 * @brief The markdown of a research report, the half that another repository reads back.
 *
 * `harvest::ResearchReport` in LplKnowledge parses what this writes, and finds its way by the
 * headings, the byline and the reachable mark declared here. Those strings are a CONTRACT,
 * stated in both repositories.
 *
 * Kept apart from `Report.cpp`, and not for tidiness: planning and writing sections needs a
 * language model and re-checking URLs needs a network, so that file pulls llama.cpp and
 * cpp-httplib, and nothing declared here does. A test target compiles `ReportMarkdown.cpp`
 * alone and checks the lines the reader parses without an inference run.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_REPORT_MARKDOWN_HPP
#    define LPL_RESEARCH_REPORT_MARKDOWN_HPP

#    include <string>
#    include <string_view>
#    include <vector>

#    include <lpl/research/State.hpp>

namespace lpl::research {

/**
 * @brief The heading under which a report lists its findings, one per line.
 *
 * The reader tells the findings list apart from the source list below it by this word alone:
 * both lists have the same bullet shape, `- [S<id>] ...`.
 *
 * @warning Renaming it renames it in both repositories or in neither. A reader that no longer
 * finds it reports zero findings, which reads exactly like a run that found none.
 */
inline constexpr std::string_view kFindingsHeading = "Constats";

/// The heading under which a report lists the sources it read.
inline constexpr std::string_view kSourcesHeading = "Sources";

/// The phrase that opens the byline: the reader recognises a report, and finds its date, by it.
inline constexpr std::string_view kReportByline = "Rapport Laplace deep research";

/// The mark of a source whose URL still answered, the only mark the reader takes as evidence.
inline constexpr std::string_view kReachableMark = "✓";

/**
 * @struct UrlGate
 * @brief The outcome of re-checking one cited URL at write time.
 */
struct UrlGate {
    int  sourceId  = 0;     ///< Which source, by its run-local id.
    bool reachable = false; ///< Whether HEAD or GET returned 2xx or 3xx.
};

/**
 * @brief Renders the findings section of a report.
 *
 * Each learning of @c RunState::knowledge becomes one bullet, verbatim, with the `[S<id>]` tag
 * the engine put there when it extracted it: the tag is all that ties a finding to the source
 * that backs it, and a reworded learning would be a claim this function invented. A learning
 * that spans several lines is joined onto one, because the reader addresses a finding by its
 * line.
 *
 * This list is the only structured record of what a run learned that outlives the run:
 * `state.json` is a resume checkpoint, private to the engine, and may be deleted once a run
 * completes.
 *
 * @param st The run.
 * @return The markdown section, empty when the run learned nothing.
 */
[[nodiscard]] std::string renderFindings(const RunState& st);

/**
 * @brief Renders one section the model planned and wrote, so that it can neither hide nor fake
 * the findings and the sources the reader parses.
 *
 * The heading is the planned title on one line. A title, or a heading inside the prose, that
 * the reader would take for one of its own headings, @ref kFindingsHeading or
 * @ref kSourcesHeading, gets a prefix, so that the model's words are never read as the findings
 * or as the sources. A heading inside a code block is code, and is left as written. A code
 * fence that the prose leaves open, typically because the section hit its token limit inside a
 * code block, is closed: the reader skips every line inside a fence, so an open one would hide
 * every finding and every source below it.
 *
 * @param plannedTitle The title the model planned.
 * @param prose        What the model wrote under it.
 * @return The markdown section, heading included.
 */
[[nodiscard]] std::string renderSection(std::string_view plannedTitle, std::string_view prose);

/**
 * @brief Assembles the whole report from parts that have already been decided.
 *
 * The same arguments give the same bytes: the timestamp is a parameter rather than a call to
 * `std::time`, the same choice `InferenceBudget` made by counting turns rather than
 * milliseconds. The synthesis and the body get the same treatment as the prose in
 * @ref renderSection before the findings are written below them.
 *
 * Each source with a gate is marked @ref kReachableMark or `✗`, except when gates were checked
 * and none answered: a network that was down says nothing about any URL, so every mark is then
 * `~` and the report says why.
 *
 * @param st    The run.
 * @param body  The sections, each rendered by @ref renderSection, in order.
 * @param gates The URL re-checks, one per source that was read successfully.
 * @param stamp The `YYYY-MM-DD HH:MM` the byline carries.
 * @return The whole report.
 */
[[nodiscard]] std::string assembleReport(const RunState& st, const std::string& body,
                                         const std::vector<UrlGate>& gates, const std::string& stamp);

} // namespace lpl::research

#endif // LPL_RESEARCH_REPORT_MARKDOWN_HPP
