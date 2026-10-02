/**
 * @file test_research_report.cpp
 * @brief The bytes of a report, held to the contract another repository reads them by.
 *
 * A research run's report is the only durable record of what the run learned. `state.json`
 * beside it is a resume checkpoint — private to this engine, versioned for its own
 * convenience, and legitimately deleted once a run completes — so anything a librarian is
 * going to keep has to survive in the markdown.
 *
 * @warning **Why this test exists at all, and it is a measurement rather than a worry.** A real run
 * was taken off disk and pushed through `lpl-ingest`: eleven kilobytes, seven cited sources,
 * twenty extracted findings, and the image it baked asserted **zero facts**. The report
 * dissolved its findings into paragraphs and dropped the structure it already had.
 *
 * What this checks is therefore not that the prose is good — no test can — but that the
 * machine-readable half is present and shaped as agreed. It compiles `Findings.cpp` alone,
 * with neither llama.cpp nor cpp-httplib, which is the whole reason that file is separate:
 * a report's format can be checked without an inference run.
 *
 * It also WRITES a report, so `validate.sh` can feed the writer's real output to the reader
 * in LplKnowledge rather than to a hand-typed imitation of it. A fixture typed into the
 * reader's own test proves only that the reader agrees with itself.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/research/Report.hpp>

#include <cstdio>
#include <fstream>
#include <string>

namespace {

int gChecks = 0;
int gFailures = 0;

/**
 * @brief Records one check.
 *
 * @param label What was checked.
 * @param ok    Whether it held.
 */
void check(const char *label, bool ok)
{
    ++gChecks;
    if (!ok)
    {
        ++gFailures;
        std::printf("  (fail) %s\n", label);
    }
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

    st.sources.push_back({1, "https://example.org/paper", "A deterministic replay study", "snippet",
                          "openalex", "cache1", true, true});
    st.sources.push_back({2, "https://example.org/wiki", "Determinism", "snippet", "wikipedia", "cache2",
                          true, true});
    st.sources.push_back({3, "https://example.org/blocked", "", "snippet", "arxiv", "", true, false});

    st.knowledge.push_back("[S1] Fixed-point arithmetic replays bit-exactly across targets.");
    st.knowledge.push_back("[S1] Floating point contraction is the usual source of divergence.");
    st.knowledge.push_back("[S2] Determinism is a property of the whole pipeline, not one stage.");

    st.diags["openalex"] = {5, "ok"};
    return st;
}

} // namespace

int main(int argc, char **argv)
{
    std::printf("test-research-report — the half of a report a machine reads\n");

    const lpl::research::RunState st = fixtureRun();

    // ── The findings list ─────────────────────────────────────────────────────
    std::printf("── findings\n");
    const std::string findings = lpl::research::renderFindings(st);
    check("a run with learnings renders a findings section", !findings.empty());
    check("the section carries the agreed heading",
          findings.find(std::string{"## "} + lpl::research::kFindingsHeading) == 0u);
    check("every learning is a bullet", findings.find("\n- [S1] Fixed-point arithmetic") != std::string::npos);
    check("the source tag survives verbatim", findings.find("- [S2] Determinism is a property") != std::string::npos);

    // A run that learned nothing renders nothing. The distinction matters to the reader:
    // an empty section and an absent one both mean "no findings", but emitting a heading
    // with no bullets would make a reader report a section it could not use.
    lpl::research::RunState barren = st;
    barren.knowledge.clear();
    check("a run that learned nothing renders no section", lpl::research::renderFindings(barren).empty());

    // A multi-line learning must not become two findings: a locus addresses ONE line, so a
    // finding spanning two would have a citation pointing at half of itself.
    lpl::research::RunState folded = st;
    folded.knowledge.clear();
    folded.knowledge.push_back("[S1] First half\nsecond half");
    const std::string one = lpl::research::renderFindings(folded);
    check("a multi-line learning stays one line", one.find("- [S1] First half second half\n") != std::string::npos);

    // ── The whole report ──────────────────────────────────────────────────────
    std::printf("── assembly\n");
    std::vector<lpl::research::UrlGate> gates;
    gates.push_back({1, true, 200});
    gates.push_back({2, false, 404});
    const std::string report =
        lpl::research::assembleReport(st, "## Background\n\nSome prose citing [S1].\n\n", gates, false,
                                      "2026-08-06 09:30");

    check("the report opens with the topic", report.find("# Reproducibility of deterministic simulation") == 0u);
    check("the byline names the writer", report.find("*Rapport Laplace deep research — 2026-08-06 09:30") != std::string::npos);
    check("the byline carries an ISO date the reader can find", report.find("2026-08-06") != std::string::npos);
    check("the findings survive into the report", report.find("- [S1] Fixed-point arithmetic") != std::string::npos);
    check("the findings precede the source list",
          report.find("## Constats") < report.find("## Sources"));
    check("a reachable source is marked", report.find("<https://example.org/paper> (openalex) \xE2\x9C\x93") != std::string::npos);
    check("an unreachable source is marked apart", report.find("<https://example.org/wiki> (wikipedia) \xE2\x9C\x97") != std::string::npos);
    check("an unfetchable source is struck through", report.find("~~https://example.org/blocked~~") != std::string::npos);

    // Determinism: the same run assembles the same bytes twice. The timestamp is a
    // parameter for exactly this reason — a wall clock inside a function whose output is
    // compared makes the comparison meaningless.
    const std::string again =
        lpl::research::assembleReport(st, "## Background\n\nSome prose citing [S1].\n\n", gates, false,
                                      "2026-08-06 09:30");
    check("assembly is deterministic", report == again);

    // Offline: when no check succeeded, no mark means anything and the report says so
    // rather than printing crosses that would read as "these URLs are dead".
    const std::string dark =
        lpl::research::assembleReport(st, "", gates, true, "2026-08-06 09:30");
    check("an offline pass marks nothing as broken", dark.find("(openalex) \xE2\x9C\x97") == std::string::npos);
    check("an offline pass says why", dark.find("Réseau indisponible") != std::string::npos);

    // ── The fixture validate.sh feeds to the reader ───────────────────────────
    if (argc >= 2)
    {
        std::ofstream out{argv[1], std::ios::binary};
        out << report;
        check("the fixture was written", out.good());
        std::printf("fixture: %s\n", argv[1]);
    }

    std::printf("%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
