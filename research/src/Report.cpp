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
#include <lpl/research/ReportMarkdown.hpp>
#include <lpl/research/WebFetch.hpp>

#include <nlohmann/json.hpp>

#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

using nlohmann::json;

namespace lpl::research {

namespace {

constexpr const char* kWriterSystem =
    "You are a precise technical writer. You write dense, well-structured "
    "markdown grounded ONLY in the provided knowledge. You cite sources "
    "inline with their [S<id>] tags. No filler, no invented facts.";

constexpr int kUrlRecheckTimeoutSeconds = 8;

std::string knowledge_block(const RunState& st) {
    std::ostringstream k;
    for (const auto& l : st.knowledge) k << "- " << l << "\n";
    return k.str();
}

std::vector<std::string> planSections(RunState& st, LanguageModelClient& llm) {
    std::vector<std::string> titles;
    try {
        std::ostringstream user;
        user << "RESEARCH TOPIC: " << st.topic << "\n\nKNOWLEDGE:\n"
             << knowledge_block(st)
             << "\nPropose 3 to 5 section titles for the final report, in the "
                "language of the topic. Output JSON: {\"sections\": [...]}";
        auto reply = llm.complete(kWriterSystem, user.str(),
                                  stringListsGrammar({"sections"}), 250, 0.3f);
        st.tokens_used += reply.approximateTokens;
        for (const auto& title : json::parse(reply.text).value("sections", json::array())) {
            if (titles.size() >= 5) break;
            titles.push_back(title.get<std::string>());
        }
    } catch (...) {
        // An unreadable plan still gives a report, without detailed sections.
    }
    return titles;
}

std::string writeSections(RunState& st, LanguageModelClient& llm, const std::vector<std::string>& titles) {
    std::ostringstream body;
    for (const std::string& title : titles) {
        try {
            std::ostringstream user;
            user << "RESEARCH TOPIC: " << st.topic << "\n\nKNOWLEDGE:\n"
                 << knowledge_block(st) << "\nWrite ONLY the section titled « "
                 << title << " » of the report (2-4 paragraphs, markdown, "
                    "cite [S<id>]). Do not repeat the title.";
            auto reply = llm.complete(kWriterSystem, user.str(), "", 700, 0.4f);
            st.tokens_used += reply.approximateTokens;
            body << renderSection(title, reply.text);
            save_state(st);
        } catch (const std::exception&) {
            body << renderSection(title, "*(section non générée)*");
        }
    }
    return body.str();
}

bool isSuccessOrRedirect(int httpStatus) {
    return httpStatus >= 200 && httpStatus < 400;
}

std::vector<UrlGate> recheckCitedUrls(const RunState& st) {
    std::vector<UrlGate> gates;
    for (const Source& source : st.sources) {
        if (!source.read || !source.fetchOk) continue;
        const bool reachable = isSuccessOrRedirect(headUrl(source.url, kUrlRecheckTimeoutSeconds).status) ||
                               isSuccessOrRedirect(fetchUrl(source.url, kUrlRecheckTimeoutSeconds).status);
        gates.push_back({.sourceId = source.id, .reachable = reachable});
    }
    return gates;
}

std::string localStamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M", &local);
    return stamp;
}

} // namespace

std::string writeReport(RunState& st, LanguageModelClient& llm, const ResearchOptions& opt) {
    (void)opt;
    const std::vector<std::string> titles = planSections(st, llm);
    const std::string body = writeSections(st, llm, titles);
    const std::vector<UrlGate> gates = recheckCitedUrls(st);
    const std::string report = assembleReport(st, body, gates, localStamp());

    const std::string path = st.run_dir + "/report.md";
    std::ofstream file(path, std::ios::binary);
    file << report;
    file.close();
    if (!file) throw std::runtime_error("research report: could not write " + path);
    return path;
}

} // namespace lpl::research
