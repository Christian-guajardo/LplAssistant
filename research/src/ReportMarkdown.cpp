/**
 * @file ReportMarkdown.cpp
 * @brief Implementation of the markdown of a research report.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/research/ReportMarkdown.hpp>

#include <algorithm>
#include <sstream>

namespace lpl::research {

namespace {

constexpr std::string_view kFence = "```";
constexpr std::size_t kDeepestHeading = 6;
constexpr std::string_view kUnreachableMark = "✗";
constexpr std::string_view kUncheckedMark = "~";
constexpr std::string_view kPrefixBeforeReservedHeading = "Section : ";

std::string toSingleLine(std::string_view text) {
    std::string line{text};
    std::replace_if(line.begin(), line.end(), [](char c) { return c == '\n' || c == '\r'; }, ' ');
    return line;
}

std::string_view trimmed(std::string_view text) {
    const std::size_t first = text.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) return {};
    return text.substr(first, text.find_last_not_of(" \t\r") - first + 1);
}

// The reader's rules, byte for byte. A heading is one to six '#' then a space, matched by the
// prefix of its trimmed name. A line whose first three bytes are backticks opens or closes a
// fence; an indented or a tilde fence, which CommonMark accepts, is not one to the reader.
bool readsAsReservedHeading(std::string_view name) {
    return name.starts_with(kFindingsHeading) || name.starts_with(kSourcesHeading);
}

std::string headingTitle(std::string_view plannedTitle) {
    const std::string title = toSingleLine(plannedTitle);
    const std::string_view name = trimmed(title);
    if (!readsAsReservedHeading(name)) return title;
    return std::string{kPrefixBeforeReservedHeading} + std::string{name};
}

std::string disarmedLine(std::string_view line) {
    const std::size_t depth = line.find_first_not_of('#');
    const bool isHeading =
        depth != 0 && depth != std::string_view::npos && depth <= kDeepestHeading && line[depth] == ' ';
    if (!isHeading || !readsAsReservedHeading(trimmed(line.substr(depth)))) return std::string{line};
    return std::string{line.substr(0, depth + 1)} + headingTitle(line.substr(depth));
}

std::string disarmedForTheReader(std::string_view prose) {
    std::string disarmed;
    bool insideFence = false;
    for (std::size_t lineStart = 0; lineStart < prose.size();) {
        const std::size_t lineEnd = std::min(prose.find('\n', lineStart), prose.size());
        const std::string_view line = prose.substr(lineStart, lineEnd - lineStart);
        if (line.starts_with(kFence)) {
            insideFence = !insideFence;
            disarmed += line;
        } else {
            disarmed += insideFence ? std::string{line} : disarmedLine(line);
        }
        if (lineEnd < prose.size()) disarmed += '\n';
        lineStart = lineEnd + 1;
    }
    if (!insideFence) return disarmed;
    if (!disarmed.empty() && disarmed.back() != '\n') disarmed += '\n';
    disarmed += kFence;
    disarmed += '\n';
    return disarmed;
}

bool networkLooksDown(const std::vector<UrlGate>& gates) {
    return !gates.empty() &&
           std::none_of(gates.begin(), gates.end(), [](const UrlGate& gate) { return gate.reachable; });
}

std::string_view reachabilityMark(int sourceId, const std::vector<UrlGate>& gates) {
    const auto gate = std::find_if(gates.begin(), gates.end(), [sourceId](const UrlGate& candidate) {
        return candidate.sourceId == sourceId;
    });
    if (gate == gates.end()) return {};
    if (networkLooksDown(gates)) return kUncheckedMark;
    return gate->reachable ? kReachableMark : kUnreachableMark;
}

std::string renderSources(const RunState& st, const std::vector<UrlGate>& gates) {
    std::ostringstream section;
    section << "## " << kSourcesHeading << "\n\n";
    for (const Source& source : st.sources) {
        if (!source.read) continue;
        section << "- [S" << source.id << "] ";
        if (!source.fetchOk) {
            section << "~~" << source.url << "~~ (lecture échouée)\n";
            continue;
        }
        section << (source.title.empty() ? source.url : toSingleLine(source.title)) << " — <" << source.url
                << "> (" << source.provider << ")";
        if (const std::string_view mark = reachabilityMark(source.id, gates); !mark.empty())
            section << ' ' << mark;
        section << "\n";
    }
    if (networkLooksDown(gates))
        section << "\n*Réseau indisponible au moment de la vérification : "
                   "gates URL ignorées (~).*\n";
    return section.str();
}

std::string renderDiagnostics(const RunState& st) {
    std::ostringstream section;
    section << "## Diagnostics providers\n\n| Provider | Résultats | Statut |\n|---|---|---|\n";
    for (const auto& [provider, diagnostic] : st.diags)
        section << "| " << provider << " | " << diagnostic.results << " | " << diagnostic.status << " |\n";
    return section.str();
}

std::string renderLimits(const RunState& st) {
    std::ostringstream section;
    section << "## Limites\n\n";
    if (st.beast_mode)
        section << "- Réponse produite sous contrainte de budget (beast mode) : "
                   "couverture possiblement partielle.\n";
    for (const FailedAttempt& attempt : st.attempts)
        section << "- Tentative rejetée en cours de route : " << attempt.reason << "\n";
    const auto unreadable = std::count_if(st.sources.begin(), st.sources.end(), [](const Source& source) {
        return source.read && !source.fetchOk;
    });
    if (unreadable > 0)
        section << "- " << unreadable << " source(s) non lisible(s) "
                   "(anti-bot/JS lourd) : contenu limité aux extraits de recherche.\n";
    section << "- Texte intégral des pages en cache : `cache/` (ids [S] ci-dessus).\n";
    return section.str();
}

} // namespace

std::string renderFindings(const RunState& st) {
    if (st.knowledge.empty()) return {};
    std::ostringstream section;
    section << "## " << kFindingsHeading << "\n\n";
    for (const std::string& learning : st.knowledge) section << "- " << toSingleLine(learning) << "\n";
    section << "\n";
    return section.str();
}

std::string renderSection(std::string_view plannedTitle, std::string_view prose) {
    return "## " + headingTitle(plannedTitle) + "\n\n" + disarmedForTheReader(prose) + "\n\n";
}

std::string assembleReport(const RunState& st, const std::string& body, const std::vector<UrlGate>& gates,
                           const std::string& stamp) {
    std::ostringstream report;
    report << "# " << st.topic << "\n\n";
    report << "*" << kReportByline << " — " << stamp << " — " << st.step << " pas, ~" << st.tokens_used
           << " tokens" << (st.beast_mode ? ", beast mode" : "") << "*\n\n";
    report << "## Synthèse\n\n" << disarmedForTheReader(st.final_answer) << "\n\n";
    report << disarmedForTheReader(body);
    report << renderFindings(st);
    report << renderSources(st, gates) << "\n";
    report << renderDiagnostics(st) << "\n";
    report << renderLimits(st);
    return report.str();
}

} // namespace lpl::research
