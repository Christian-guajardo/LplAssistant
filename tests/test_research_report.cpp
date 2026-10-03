/**
 * @file test_research_report.cpp
 * @brief The lines of a report that another repository reads, held to the contract it reads them by.
 *
 * @note **Why this test exists at all, and it is a measurement rather than a worry.** A real run
 * was taken off disk and pushed through `lpl-ingest`: eleven kilobytes, seven cited sources,
 * twenty extracted findings, and the image it baked asserted **zero facts**. The report
 * dissolved its findings into paragraphs and dropped the structure it already had.
 *
 * What this checks is therefore not that the prose is good — no test can — but that the
 * machine-readable half is present and shaped as agreed, whatever the model writes around it.
 *
 * It also WRITES a report, so the maintainer's local `validate.sh` script can feed the writer's
 * real output to the reader in LplKnowledge rather than a hand-typed imitation of it. A fixture
 * typed into the reader's own test proves only that the reader agrees with itself.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/research/ReportMarkdown.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

int gFailures = 0;
int gChecks = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++gFailures;
}

// The two helpers below read a report the way `harvest::ResearchReport` in LplKnowledge does: a
// line whose first three bytes are backticks opens or closes a fence, and a heading is one to six
// '#' then a space, matched by the prefix of its trimmed name.
[[nodiscard]] int countFenceLines(const std::string &text)
{
    int fences = 0;
    for (std::size_t lineStart = 0; lineStart < text.size();)
    {
        if (text.compare(lineStart, 3, "```") == 0)
            ++fences;
        const std::size_t lineEnd = text.find('\n', lineStart);
        if (lineEnd == std::string::npos)
            break;
        lineStart = lineEnd + 1;
    }
    return fences;
}

[[nodiscard]] int headingsStartingWith(const std::string &text, const std::string &name)
{
    int headings = 0;
    bool inFence = false;
    for (std::size_t lineStart = 0; lineStart < text.size();)
    {
        std::size_t lineEnd = text.find('\n', lineStart);
        if (lineEnd == std::string::npos)
            lineEnd = text.size();
        const std::string line = text.substr(lineStart, lineEnd - lineStart);
        lineStart = lineEnd + 1;
        if (line.starts_with("```"))
        {
            inFence = !inFence;
            continue;
        }
        const std::size_t depth = line.find_first_not_of('#');
        if (inFence || depth == 0 || depth > 6 || depth == std::string::npos || line[depth] != ' ')
            continue;
        const std::size_t nameStart = line.find_first_not_of(" \t\r", depth);
        if (nameStart != std::string::npos && line.compare(nameStart, name.size(), name) == 0)
            ++headings;
    }
    return headings;
}

/**
 * @brief Builds a run whose shape exercises every branch a reader has to handle.
 *
 * Three sources on purpose, and each is a different case rather than a repetition: one read
 * and reachable, one read and confirmed dead, one the run could not fetch at all. A fixture
 * where every source is healthy would let a reader that ignores the marks pass.
 *
 * @return The run.
 */
[[nodiscard]] lpl::research::RunState fixtureRun()
{
    lpl::research::RunState st;
    st.topic = "Reproducibility of deterministic simulation";
    st.run_dir = "research_runs/fixture";
    st.step = 6;
    st.tokens_used = 12345;
    st.beast_mode = false;
    st.final_answer = "Fixed-point state is what makes a replay bit-exact [S1].";

    st.sources.push_back({.id = 1,
                          .url = "https://example.org/paper",
                          .title = "A deterministic replay study",
                          .snippet = "snippet",
                          .provider = "openalex",
                          .cacheId = "cache1",
                          .read = true,
                          .fetchOk = true});
    st.sources.push_back({.id = 2,
                          .url = "https://example.org/wiki",
                          .title = "Determinism",
                          .snippet = "snippet",
                          .provider = "wikipedia",
                          .cacheId = "cache2",
                          .read = true,
                          .fetchOk = true});
    st.sources.push_back({.id = 3,
                          .url = "https://example.org/blocked",
                          .title = "",
                          .snippet = "snippet",
                          .provider = "arxiv",
                          .cacheId = "",
                          .read = true,
                          .fetchOk = false});

    st.knowledge.push_back("[S1] Fixed-point arithmetic replays bit-exactly across targets.");
    st.knowledge.push_back("[S1] Floating point contraction is the usual source of divergence.");
    st.knowledge.push_back("[S2] Determinism is a property of the whole pipeline, not one stage.");

    st.diags["openalex"] = {.results = 5, .status = "ok"};
    return st;
}

} // namespace

int main(int argc, char **argv)
{
    using lpl::research::assembleReport;
    using lpl::research::renderFindings;
    using lpl::research::renderSection;
    using lpl::research::UrlGate;

    std::printf("test-research-report — the half of a report a machine reads\n");

    const lpl::research::RunState st = fixtureRun();
    const std::string stamp = "2026-08-06 09:30";

    // ── The findings list ─────────────────────────────────────────────────────
    std::printf("── findings\n");
    const std::string findings = renderFindings(st);
    check(findings.starts_with("## Constats\n\n"), "the section opens with the agreed heading");
    bool everyLearningIsABullet = !st.knowledge.empty();
    for (const std::string &learning : st.knowledge)
        everyLearningIsABullet =
            everyLearningIsABullet && findings.find("- " + learning + "\n") != std::string::npos;
    check(everyLearningIsABullet, "every learning is a bullet, verbatim with its source tag");

    // An empty section and an absent one both mean "no findings" to the reader, but a heading
    // with no bullets would make it report a section it could not use.
    lpl::research::RunState barren = st;
    barren.knowledge.clear();
    check(renderFindings(barren).empty(), "a run that learned nothing renders no section");

    lpl::research::RunState folded = st;
    folded.knowledge.clear();
    folded.knowledge.push_back("[S1] First half\nsecond half");
    check(renderFindings(folded).find("- [S1] First half second half\n") != std::string::npos,
          "a multi-line learning stays one line");

    // ── A section the model wrote ─────────────────────────────────────────────
    std::printf("── sections\n");
    check(renderSection("Background", "Some prose.") == "## Background\n\nSome prose.\n\n",
          "a planned section renders as its heading and its prose");

    const std::string findingsTitled =
        renderSection("Constats principaux", "- [S2] A sentence of the writer, not a finding.");
    check(headingsStartingWith(findingsTitled, "Constats") == 0 &&
              findingsTitled.find("Constats principaux") != std::string::npos,
          "a section titled like the findings list keeps its words and is not read as it");

    const std::string sourcesTitled = renderSection(" Sources consultées", "Some prose.");
    check(headingsStartingWith(sourcesTitled, "Sources") == 0 &&
              sourcesTitled.find("Sources consultées") != std::string::npos,
          "a section titled like the source list keeps its words and is not read as it");

    const std::string twoLineTitle = renderSection("Background\n## Constats", "Some prose.");
    check(headingsStartingWith(twoLineTitle, "Constats") == 0 &&
              twoLineTitle.find("\n\nSome prose.\n\n") != std::string::npos,
          "a planned title stays one heading line");

    check(countFenceLines(renderSection("Example", "```cpp\nint x = 0;")) == 2,
          "a section cut inside a code block closes it");

    const std::string findingsInProse = renderSection("Background", "### Constats\n\n- [S1] x");
    check(headingsStartingWith(findingsInProse, "Constats") == 0 &&
              findingsInProse.find("### Section : Constats\n") != std::string::npos,
          "a heading in the prose that reads like the findings list is disarmed like a title");

    check(renderSection("Background", "### Details\n\nx").find("\n### Details\n") != std::string::npos,
          "an ordinary heading in the prose is left as written");

    check(renderSection("Example", "```\n## Constats\n```").find("\n## Constats\n") != std::string::npos,
          "a reserved heading inside a code block is code, and is left as written");

    // ── The whole report ──────────────────────────────────────────────────────
    std::printf("── assembly\n");
    const std::vector<UrlGate> gates{{.sourceId = 1, .reachable = true}, {.sourceId = 2, .reachable = false}};
    const std::string background = "## Background\n\nSome prose citing [S1].\n\n";
    const std::string report = assembleReport(st, background, gates, stamp);

    check(report.starts_with("# Reproducibility of deterministic simulation\n"), "the report opens with the topic");
    check(report.find("*Rapport Laplace deep research — 2026-08-06 09:30 — ") != std::string::npos,
          "the byline names the writer and carries the stamp");
    check(report.find("- [S1] Fixed-point arithmetic replays bit-exactly across targets.\n") != std::string::npos,
          "the findings survive into the report");
    const std::size_t findingsAt = report.find("## Constats\n");
    const std::size_t sourcesAt = report.find("## Sources\n");
    check(findingsAt != std::string::npos && sourcesAt != std::string::npos && findingsAt < sourcesAt,
          "the findings precede the source list");
    check(report.find("<https://example.org/paper> (openalex) \xE2\x9C\x93\n") != std::string::npos,
          "a reachable source is marked reachable");
    check(report.find("<https://example.org/wiki> (wikipedia) \xE2\x9C\x97\n") != std::string::npos,
          "an unreachable source is marked unreachable");
    check(report.find("- [S3] ~~https://example.org/blocked~~ (lecture échouée)\n") != std::string::npos,
          "an unfetchable source is struck through");
    check(report == assembleReport(st, background, gates, stamp), "assembly is deterministic");

    // When no gate answered, no mark means anything: crosses would read as "these URLs are dead".
    const std::vector<UrlGate> noGateAnswered{{.sourceId = 1, .reachable = false},
                                              {.sourceId = 2, .reachable = false}};
    const std::string dark = assembleReport(st, "", noGateAnswered, stamp);
    check(dark.find("(wikipedia) \xE2\x9C\x97") == std::string::npos &&
              dark.find("(wikipedia) ~\n") != std::string::npos,
          "an offline pass marks the dead source unchecked rather than broken");
    check(dark.find("Réseau indisponible") != std::string::npos, "an offline pass says why");
    check(report.find("Réseau indisponible") == std::string::npos,
          "one gate that answered is enough to trust the others");
    check(assembleReport(st, "", {}, stamp).find("Réseau indisponible") == std::string::npos,
          "a run with nothing to re-check does not claim the network was down");

    lpl::research::RunState multiLineTitle = st;
    multiLineTitle.sources[0].title = "A deterministic\nreplay study";
    const std::string foldedTitle = assembleReport(multiLineTitle, "", gates, stamp);
    check(foldedTitle.find("- [S1] A deterministic replay study \xE2\x80\x94 <https://example.org/paper>") !=
              std::string::npos,
          "a multi-line source title stays on its source line");

    const std::string cutInsideCode =
        assembleReport(st, "## Example\n\nThe code:\n\n```cpp\nint x = 0;\n\n", gates, stamp);
    const std::size_t cutFindingsAt = cutInsideCode.find("## Constats\n");
    check(cutFindingsAt != std::string::npos && countFenceLines(cutInsideCode.substr(0, cutFindingsAt)) % 2 == 0,
          "a body cut inside a code block leaves no fence open above the findings");

    check(countFenceLines(assembleReport(st, "## Example\n\n```cpp\nint x = 0;\n```\n\n", gates, stamp)) == 2,
          "a closed code block gets no extra fence");

    lpl::research::RunState openSynthesis = st;
    openSynthesis.final_answer = "The replay:\n\n```\nstep(state);";
    const std::string synthesisCut = assembleReport(openSynthesis, background, gates, stamp);
    const std::size_t cutSectionsAt = synthesisCut.find("## Background\n");
    check(cutSectionsAt != std::string::npos && countFenceLines(synthesisCut.substr(0, cutSectionsAt)) % 2 == 0,
          "a synthesis cut inside a code block leaves no fence open above the sections");

    const std::string rawBody =
        assembleReport(st, "## Notes\n\n### Sources consultées\n\n- [S1] x\n\n", gates, stamp);
    check(headingsStartingWith(rawBody, "Sources") == 1,
          "a heading in the body that reads like the source list is disarmed");

    lpl::research::RunState headedSynthesis = st;
    headedSynthesis.final_answer = "### Constats\n\n- [S1] A sentence of the synthesis.";
    check(headingsStartingWith(assembleReport(headedSynthesis, background, gates, stamp), "Constats") == 1,
          "a heading in the synthesis that reads like the findings list is disarmed");

    // ── The fixture the local validate.sh script feeds to the reader ──────────
    if (argc >= 2)
    {
        std::ofstream out{argv[1], std::ios::binary};
        out << report;
        out.close();
        check(out.good(), "the fixture was written");
        std::printf("fixture: %s\n", argv[1]);
    }

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
