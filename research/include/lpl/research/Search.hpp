/**
 * @file Search.hpp
 * @brief Search providers, each degrading on its own.
 *
 * Local-first and API-first: no provider is required, and each is classified ok,
 * empty or error independently. One dead provider must never take the run down with
 * it, which is why the diagnostic is part of the result rather than a log line.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_SEARCH_HPP
#    define LPL_RESEARCH_SEARCH_HPP

#    include <map>
#    include <string>
#    include <vector>

// Providers de recherche 100 % locaux/API-first — la table du POC
// masterlaplace_deep-research portée en C++. Chaque provider est indépendant
// et classé ok/empty/error dans les diagnostics (dégradation propre).
namespace lpl::research {

struct SearchResult {
    std::string url;
    std::string title;
    std::string snippet;
    std::string provider; // "searxng" | "wikipedia" | "openalex" | "arxiv" | ...
};

struct ProviderDiagnostic {
    int         results = 0;
    std::string status; // "ok" | "empty" | "error: ..."
};

struct SearchOptions {
    std::string searxngUrl;                 // vide = provider désactivé
    std::string githubToken;                // optionnel (60 -> 5000 req/h)
    std::vector<std::string> providers;      // liste active (vide = défauts)
    int         perProvider = 5;
    int         timeoutSeconds    = 12;
};

// Interroge tous les providers actifs, déduplique par URL, remplit `diags`.
std::vector<SearchResult> aggregate_search(const std::string& query,
                                           const SearchOptions& opt,
                                           std::map<std::string, ProviderDiagnostic>& diags);

} // namespace lpl::research

#endif // LPL_RESEARCH_SEARCH_HPP
