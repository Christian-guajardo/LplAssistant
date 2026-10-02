/**
 * @file OpenAccess.cpp
 * @brief Implementation of resolving a work to a legally free copy.
 *
 *  
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/research/OpenAccess.hpp>

#include <lpl/research/WebFetch.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>

namespace lpl::research {

namespace {

using nlohmann::json;

// Extrait un DOI d'une URL. Gère https://doi.org/10.x/..., http://dx.doi.org/...
// et les URLs d'éditeur qui embarquent le DOI (dl.acm.org/doi/10.1145/...).
// Un DOI commence par "10." suivi d'un préfixe numérique puis d'un suffixe.
std::string extract_doi(const std::string& url) {
    auto pos = url.find("10.");
    while (pos != std::string::npos) {
        // Valider : "10." + chiffres + "/" + au moins un caractère.
        size_t i = pos + 3;
        size_t digits = 0;
        while (i < url.size() && std::isdigit((unsigned char)url[i])) { ++i; ++digits; }
        if (digits >= 3 && i < url.size() && url[i] == '/' && i + 1 < url.size()) {
            std::string doi = url.substr(pos);
            // Couper sur les caractères qui ne font jamais partie d'un DOI utile.
            auto cut = doi.find_first_of("?#");
            if (cut != std::string::npos) doi = doi.substr(0, cut);
            // Retirer une éventuelle barre finale.
            while (!doi.empty() && doi.back() == '/') doi.pop_back();
            return doi;
        }
        pos = url.find("10.", pos + 1);
    }
    return "";
}

} // namespace

std::string resolve_open_access(const std::string& url, const std::string& email,
                                int timeoutSeconds) {
    if (email.empty()) return url;
    std::string doi = extract_doi(url);
    if (doi.empty()) return url;

    std::string api = "https://api.unpaywall.org/v2/" + doi + "?email=" + email;
    auto r = fetchUrl(api, timeoutSeconds);
    if (r.status < 200 || r.status >= 300 || r.body.empty()) return url;

    try {
        auto j = json::parse(r.body);
        if (!j.value("is_oa", false)) return url;
        // Parcourir toutes les localisations OA, préférer un dépôt (non muré)
        // et un lien PDF direct.
        const json& locs = j.contains("oa_locations") ? j["oa_locations"] : json::array();
        std::string repo_pdf, repo_any;
        for (const auto& loc : locs) {
            if (loc.value("host_type", "") != "repository") continue;
            std::string pdf = loc.value("url_for_pdf", "");
            std::string any = loc.value("url", "");
            if (!pdf.empty() && repo_pdf.empty()) repo_pdf = pdf;
            if (!any.empty() && repo_any.empty()) repo_any = any;
        }
        if (!repo_pdf.empty()) return repo_pdf;
        if (!repo_any.empty()) return repo_any;
    } catch (...) {
        // JSON Unpaywall illisible : on garde l'URL d'origine.
    }
    return url; // seule copie OA chez l'éditeur (murée) : rien de mieux à offrir.
}

} // namespace lpl::research
