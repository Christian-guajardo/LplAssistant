/**
 * @file Findings.cpp
 * @brief The machine-readable half of a report: the findings list, and the assembly.
 *
 * Deliberately dependency-free. `Report.cpp` next door plans sections with a language model
 * and re-checks URLs over HTTP, so it pulls llama.cpp and cpp-httplib; nothing in this file
 * does either. That is what lets a test compile it on its own and check the exact bytes a
 * reader in another repository parses, without an inference run and without a network.
 *
 * The reader is `lpl::harvest::ResearchReport` in LplKnowledge. The two halves agree on one
 * thing — the @ref lpl::research::kFindingsHeading heading and the `- [S<id>] text` line
 * shape — and that agreement is a CONTRACT, stated in both places.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/research/Report.hpp>

#include <sstream>

namespace lpl::research {

std::string renderFindings(const RunState& st) {
    if (st.knowledge.empty()) return {};
    std::ostringstream f;
    f << "## " << kFindingsHeading << "\n\n";
    for (const auto& l : st.knowledge) {
        // Verbatim, including the [S<id>] tag the extraction step put there. A learning
        // reworded on the way out would be a claim this function invented, and the tag is
        // the only thing tying it to the source that backs it.
        std::string one = l;
        // One finding, one line: a locus addresses a line, so a finding that spanned two
        // would be a claim whose citation points at half of it.
        for (char& c : one)
            if (c == '\n' || c == '\r') c = ' ';
        f << "- " << one << "\n";
    }
    f << "\n";
    return f.str();
}

std::string assembleReport(const RunState& st, const std::string& body,
                           const std::vector<UrlGate>& gates, bool offline,
                           const std::string& stamp) {
    std::ostringstream md;
    md << "# " << st.topic << "\n\n";
    md << "*Rapport Laplace deep research — " << stamp << " — "
       << st.step << " pas, ~" << st.tokens_used << " tokens"
       << (st.beast_mode ? ", beast mode" : "") << "*\n\n";
    md << "## Synthèse\n\n" << st.final_answer << "\n\n";
    md << body;

    // The findings, one per line, each still carrying the source that backs it. The prose
    // sections above are for a person; this list is what survives being read by something
    // that is not one. Without it the only structured record of what the run learned is
    // `state.json`, which is a resume checkpoint — private to this engine, versioned for its
    // own convenience, and legitimately deleted once a run completes.
    md << renderFindings(st);

    md << "## Sources\n\n";
    for (const auto& s : st.sources) {
        if (!s.read) continue;
        md << "- [S" << s.id << "] ";
        if (!s.fetchOk) { md << "~~" << s.url << "~~ (lecture échouée)\n"; continue; }
        std::string mark = "";
        for (const auto& g : gates)
            if (g.sourceId == s.id) mark = offline ? " ~" : (g.ok ? " ✓" : " ✗");
        md << (s.title.empty() ? s.url : s.title) << " — <" << s.url << "> ("
           << s.provider << ")" << mark << "\n";
    }
    if (offline)
        md << "\n*Réseau indisponible au moment de la vérification : "
              "gates URL ignorées (~).*\n";

    md << "\n## Diagnostics providers\n\n| Provider | Résultats | Statut |\n|---|---|---|\n";
    for (const auto& [name, d] : st.diags)
        md << "| " << name << " | " << d.results << " | " << d.status << " |\n";

    md << "\n## Limites\n\n";
    if (st.beast_mode)
        md << "- Réponse produite sous contrainte de budget (beast mode) : "
              "couverture possiblement partielle.\n";
    for (const auto& a : st.attempts)
        md << "- Tentative rejetée en cours de route : " << a.reason << "\n";
    int failed_reads = 0;
    for (const auto& s : st.sources)
        if (s.read && !s.fetchOk) ++failed_reads;
    if (failed_reads)
        md << "- " << failed_reads << " source(s) non lisible(s) "
              "(anti-bot/JS lourd) : contenu limité aux extraits de recherche.\n";
    md << "- Texte intégral des pages en cache : `cache/` (ids [S] ci-dessus).\n";

    return md.str();
}

} // namespace lpl::research
