#include "report.h"
#include "engine.h"
#include "grammar.h"
#include "http.h"

#include <nlohmann/json.hpp>

#include <ctime>
#include <fstream>
#include <sstream>

using nlohmann::json;

namespace laplace::research {

namespace {

constexpr const char* kWriterSystem =
    "You are a precise technical writer. You write dense, well-structured "
    "markdown grounded ONLY in the provided knowledge. You cite sources "
    "inline with their [S<id>] tags. No filler, no invented facts.";

std::string knowledge_block(const RunState& st) {
    std::ostringstream k;
    for (const auto& l : st.knowledge) k << "- " << l << "\n";
    return k.str();
}

} // namespace

std::string write_report(RunState& st, LlmClient& llm, const ResearchOptions& opt) {
    (void)opt;
    // 1. Plan des sections.
    std::vector<std::string> sections;
    try {
        std::ostringstream user;
        user << "RESEARCH TOPIC: " << st.topic << "\n\nKNOWLEDGE:\n"
             << knowledge_block(st)
             << "\nPropose 3 to 5 section titles for the final report, in the "
                "language of the topic. Output JSON: {\"sections\": [...]}";
        auto reply = llm.complete(kWriterSystem, user.str(),
                                  string_lists_grammar({"sections"}), 250, 0.3f);
        st.tokens_used += reply.approx_tokens;
        for (const auto& s : json::parse(reply.text).value("sections", json::array())) {
            if (sections.size() >= 5) break;
            sections.push_back(s.get<std::string>());
        }
    } catch (...) { /* plan illisible : rapport sans sections détaillées */ }

    // 2. Rédaction progressive, section par section.
    std::ostringstream body;
    for (const auto& title : sections) {
        try {
            std::ostringstream user;
            user << "RESEARCH TOPIC: " << st.topic << "\n\nKNOWLEDGE:\n"
                 << knowledge_block(st) << "\nWrite ONLY the section titled « "
                 << title << " » of the report (2-4 paragraphs, markdown, "
                    "cite [S<id>]). Do not repeat the title.";
            auto reply = llm.complete(kWriterSystem, user.str(), "", 700, 0.4f);
            st.tokens_used += reply.approx_tokens;
            body << "## " << title << "\n\n" << reply.text << "\n\n";
            save_state(st); // checkpoint entre les sections
        } catch (const std::exception&) {
            body << "## " << title << "\n\n*(section non générée)*\n\n";
        }
    }

    // 3. Quality gates : vérification HTTP des sources lues (star-hengxing).
    //    HEAD d'abord, GET en secours ; si tout échoue, réseau supposé absent.
    struct Gate { const Source* src; bool ok; int status; };
    std::vector<Gate> gates;
    int reachable = 0, checked = 0;
    for (const auto& s : st.sources) {
        if (!s.read || !s.fetch_ok) continue;
        auto h = http_head(s.url, 8);
        bool ok = h.status >= 200 && h.status < 400;
        if (!ok) {
            auto g = http_get(s.url, 8);
            ok = g.status >= 200 && g.status < 400;
            if (ok) h.status = g.status;
        }
        gates.push_back({&s, ok, h.status});
        ++checked;
        if (ok) ++reachable;
    }
    const bool offline = checked > 0 && reachable == 0;

    // 4. Assemblage.
    char date[32];
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M", &tm);

    std::ostringstream md;
    md << "# " << st.topic << "\n\n";
    md << "*Rapport Laplace deep research — " << date << " — "
       << st.step << " pas, ~" << st.tokens_used << " tokens"
       << (st.beast_mode ? ", beast mode" : "") << "*\n\n";
    md << "## Synthèse\n\n" << st.final_answer << "\n\n";
    md << body.str();

    md << "## Sources\n\n";
    for (const auto& s : st.sources) {
        if (!s.read) continue;
        md << "- [S" << s.id << "] ";
        if (!s.fetch_ok) { md << "~~" << s.url << "~~ (lecture échouée)\n"; continue; }
        std::string mark = "";
        for (const auto& g : gates)
            if (g.src == &s) mark = offline ? " ~" : (g.ok ? " ✓" : " ✗");
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
        if (s.read && !s.fetch_ok) ++failed_reads;
    if (failed_reads)
        md << "- " << failed_reads << " source(s) non lisible(s) "
              "(anti-bot/JS lourd) : contenu limité aux extraits de recherche.\n";
    md << "- Texte intégral des pages en cache : `cache/` (ids [S] ci-dessus).\n";

    const std::string path = st.run_dir + "/report.md";
    std::ofstream f(path, std::ios::binary);
    f << md.str();
    return path;
}

} // namespace laplace::research
