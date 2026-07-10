#pragma once
#include <map>
#include <string>
#include <vector>

// Providers de recherche 100 % locaux/API-first — la table du POC
// masterlaplace_deep-research portée en C++. Chaque provider est indépendant
// et classé ok/empty/error dans les diagnostics (dégradation propre).
namespace laplace::research {

struct SearchResult {
    std::string url;
    std::string title;
    std::string snippet;
    std::string provider; // "searxng" | "wikipedia" | "openalex" | "arxiv" | ...
};

struct ProviderDiag {
    int         results = 0;
    std::string status; // "ok" | "empty" | "error: ..."
};

struct SearchOptions {
    std::string searxng_url;                 // vide = provider désactivé
    std::string github_token;                // optionnel (60 -> 5000 req/h)
    std::vector<std::string> providers;      // liste active (vide = défauts)
    int         per_provider = 5;
    int         timeout_s    = 12;
};

// Interroge tous les providers actifs, déduplique par URL, remplit `diags`.
std::vector<SearchResult> aggregate_search(const std::string& query,
                                           const SearchOptions& opt,
                                           std::map<std::string, ProviderDiag>& diags);

} // namespace laplace::research
