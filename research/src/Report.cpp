/**
 * @file Report.cpp
 * @brief Implementation of the final report, written section by section.
 *
 *  
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/research/Report.hpp>
#include <lpl/research/Engine.hpp>
#include <lpl/research/Grammar.hpp>
#include <lpl/research/WebFetch.hpp>

#include <nlohmann/json.hpp>

#include <ctime>
#include <fstream>
#include <sstream>

using nlohmann::json;

namespace lpl::research {

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

std::string writeReport(RunState& st, LanguageModelClient& llm, const ResearchOptions& opt) {
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
                                  stringListsGrammar({"sections"}), 250, 0.3f);
        st.tokens_used += reply.approximateTokens;
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
            st.tokens_used += reply.approximateTokens;
            body << "## " << title << "\n\n" << reply.text << "\n\n";
            save_state(st); // checkpoint entre les sections
        } catch (const std::exception&) {
            body << "## " << title << "\n\n*(section non générée)*\n\n";
        }
    }

    // 3. Quality gates : vérification HTTP des sources lues (star-hengxing).
    //    HEAD d'abord, GET en secours ; si tout échoue, réseau supposé absent.
    std::vector<UrlGate> gates;
    int reachable = 0, checked = 0;
    for (const auto& s : st.sources) {
        if (!s.read || !s.fetchOk) continue;
        auto h = headUrl(s.url, 8);
        bool ok = h.status >= 200 && h.status < 400;
        if (!ok) {
            auto g = fetchUrl(s.url, 8);
            ok = g.status >= 200 && g.status < 400;
            if (ok) h.status = g.status;
        }
        gates.push_back({s.id, ok, h.status});
        ++checked;
        if (ok) ++reachable;
    }
    const bool offline = checked > 0 && reachable == 0;

    // 4. Assemblage. La mise en page vit dans Findings.cpp, qui ne dépend ni d'un
    //    modèle ni du réseau : c'est ce qui permet de la tester octet pour octet, et
    //    c'est la moitié du rapport qu'un lecteur d'un autre dépôt analyse.
    char date[32];
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M", &tm);

    const std::string markdown = assembleReport(st, body.str(), gates, offline, date);

    const std::string path = st.run_dir + "/report.md";
    std::ofstream f(path, std::ios::binary);
    f << markdown;
    return path;
}

} // namespace lpl::research
