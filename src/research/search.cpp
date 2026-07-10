#include "search.h"
#include "http.h"
#include "reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <set>

using nlohmann::json;

namespace laplace::research {

namespace {

// Retire les balises HTML des snippets d'API (StackExchange, Wikipedia en mettent).
std::string strip_tags(const std::string& s) {
    std::string out;
    bool in_tag = false;
    for (char c : s) {
        if (c == '<') in_tag = true;
        else if (c == '>') in_tag = false;
        else if (!in_tag) out += c;
    }
    return out;
}

std::string clip(const std::string& s, size_t n) {
    return s.size() <= n ? s : s.substr(0, n) + "…";
}

using Results = std::vector<SearchResult>;

Results searxng(const std::string& q, const SearchOptions& opt) {
    Results out;
    auto r = http_get(opt.searxng_url + "/search?q=" + url_encode(q) +
                          "&format=json",
                      opt.timeout_s);
    if (r.status != 200) throw std::runtime_error("HTTP " + std::to_string(r.status) + " " + r.error);
    auto j = json::parse(r.body);
    for (const auto& it : j.value("results", json::array())) {
        if ((int)out.size() >= opt.per_provider) break;
        out.push_back({it.value("url", ""), it.value("title", ""),
                       clip(it.value("content", ""), 300), "searxng"});
    }
    return out;
}

Results wikipedia(const std::string& q, const SearchOptions& opt) {
    Results out;
    auto r = http_get("https://en.wikipedia.org/w/api.php?action=query&list=search"
                      "&srsearch=" + url_encode(q) + "&format=json&srlimit=" +
                          std::to_string(opt.per_provider),
                      opt.timeout_s);
    if (r.status != 200) throw std::runtime_error("HTTP " + std::to_string(r.status) + " " + r.error);
    auto j = json::parse(r.body);
    for (const auto& it : j["query"]["search"]) {
        std::string title = it.value("title", "");
        std::string slug  = title;
        std::replace(slug.begin(), slug.end(), ' ', '_');
        out.push_back({"https://en.wikipedia.org/wiki/" + slug, title,
                       clip(strip_tags(it.value("snippet", "")), 300), "wikipedia"});
    }
    return out;
}

// Reconstruit l'abstract OpenAlex depuis son index inversé {mot: [positions]}.
std::string openalex_abstract(const json& inv) {
    if (!inv.is_object()) return {};
    std::vector<std::string> words;
    for (auto it = inv.begin(); it != inv.end(); ++it)
        for (const auto& pos : it.value()) {
            size_t p = pos.get<size_t>();
            if (p >= words.size()) words.resize(p + 1);
            words[p] = it.key();
        }
    std::string s;
    for (const auto& w : words) {
        if (!s.empty()) s += ' ';
        s += w;
    }
    return s;
}

Results openalex(const std::string& q, const SearchOptions& opt) {
    Results out;
    auto r = http_get("https://api.openalex.org/works?search=" + url_encode(q) +
                          "&filter=is_oa:true&per-page=" + std::to_string(opt.per_provider),
                      opt.timeout_s);
    if (r.status != 200) throw std::runtime_error("HTTP " + std::to_string(r.status) + " " + r.error);
    auto j = json::parse(r.body);
    for (const auto& it : j.value("results", json::array())) {
        std::string url;
        if (it.contains("best_oa_location") && it["best_oa_location"].is_object()) {
            url = it["best_oa_location"].value("landing_page_url", "");
            if (url.empty()) url = it["best_oa_location"].value("pdf_url", "");
        }
        if (url.empty()) url = it.value("id", "");
        std::string snippet = openalex_abstract(it.value("abstract_inverted_index", json()));
        if (it.contains("publication_year") && it["publication_year"].is_number())
            snippet = "(" + std::to_string(it["publication_year"].get<int>()) + ") " + snippet;
        out.push_back({url, it.value("display_name", ""), clip(snippet, 300), "openalex"});
    }
    return out;
}

// L'API arXiv rend de l'Atom XML : extraction minimale <entry><id><title><summary>.
Results arxiv(const std::string& q, const SearchOptions& opt) {
    Results out;
    auto r = http_get("https://export.arxiv.org/api/query?search_query=all:" +
                          url_encode(q) + "&max_results=" + std::to_string(opt.per_provider),
                      opt.timeout_s);
    if (r.status != 200) throw std::runtime_error("HTTP " + std::to_string(r.status) + " " + r.error);
    auto field = [](const std::string& s, size_t from, const char* tag,
                    size_t* end_out) -> std::string {
        std::string open = std::string("<") + tag;
        size_t b = s.find(open, from);
        if (b == std::string::npos) return {};
        b = s.find('>', b);
        size_t e = s.find(std::string("</") + tag + ">", b);
        if (b == std::string::npos || e == std::string::npos) return {};
        if (end_out) *end_out = e;
        return s.substr(b + 1, e - b - 1);
    };
    size_t pos = 0;
    while (out.size() < (size_t)opt.per_provider) {
        size_t entry = r.body.find("<entry>", pos);
        if (entry == std::string::npos) break;
        size_t dummy = 0;
        std::string id      = field(r.body, entry, "id", &dummy);
        std::string title   = field(r.body, entry, "title", &dummy);
        std::string summary = field(r.body, entry, "summary", &dummy);
        pos = entry + 7;
        if (id.empty()) continue;
        auto compact = [](std::string s) {
            std::string t;
            bool sp = false;
            for (char c : s) {
                if (c == '\n' || c == ' ' || c == '\t') { sp = true; continue; }
                if (sp && !t.empty()) t += ' ';
                sp = false;
                t += c;
            }
            return t;
        };
        out.push_back({id, compact(title), clip(compact(summary), 300), "arxiv"});
    }
    return out;
}

Results stackexchange(const std::string& q, const SearchOptions& opt) {
    Results out;
    // L'API SE gzippe toujours : passe par le chemin curl --compressed.
    auto r = http_get("https://api.stackexchange.com/2.3/search/advanced?order=desc"
                      "&sort=relevance&q=" + url_encode(q) +
                          "&site=stackoverflow&pagesize=" + std::to_string(opt.per_provider),
                      opt.timeout_s, {}, /*compressed=*/true);
    if (r.status != 200) throw std::runtime_error("HTTP " + std::to_string(r.status) + " " + r.error);
    auto j = json::parse(r.body);
    for (const auto& it : j.value("items", json::array())) {
        out.push_back({it.value("link", ""), strip_tags(it.value("title", "")),
                       "score " + std::to_string(it.value("score", 0)) + ", " +
                           std::to_string(it.value("answer_count", 0)) + " réponses",
                       "stackexchange"});
    }
    return out;
}

Results github(const std::string& q, const SearchOptions& opt) {
    Results out;
    std::map<std::string, std::string> headers{{"Accept", "application/vnd.github+json"}};
    if (!opt.github_token.empty())
        headers["Authorization"] = "Bearer " + opt.github_token;
    auto r = http_get("https://api.github.com/search/repositories?q=" + url_encode(q) +
                          "&per_page=" + std::to_string(opt.per_provider),
                      opt.timeout_s, headers);
    if (r.status != 200) throw std::runtime_error("HTTP " + std::to_string(r.status) + " " + r.error);
    auto j = json::parse(r.body);
    for (const auto& it : j.value("items", json::array())) {
        out.push_back({it.value("html_url", ""), it.value("full_name", ""),
                       clip(it.value("description", "") + " (" +
                                std::to_string(it.value("stargazers_count", 0)) + "★)",
                            300),
                       "github"});
    }
    return out;
}

} // namespace

std::vector<SearchResult> aggregate_search(const std::string& query,
                                           const SearchOptions& opt,
                                           std::map<std::string, ProviderDiag>& diags) {
    using Fn = Results (*)(const std::string&, const SearchOptions&);
    std::vector<std::pair<std::string, Fn>> table;
    auto enabled = [&](const std::string& name) {
        if (opt.providers.empty()) return true;
        return std::find(opt.providers.begin(), opt.providers.end(), name) !=
               opt.providers.end();
    };
    if (!opt.searxng_url.empty() && enabled("searxng")) table.push_back({"searxng", searxng});
    if (enabled("wikipedia"))     table.push_back({"wikipedia", wikipedia});
    if (enabled("openalex"))      table.push_back({"openalex", openalex});
    if (enabled("arxiv"))         table.push_back({"arxiv", arxiv});
    if (enabled("stackexchange")) table.push_back({"stackexchange", stackexchange});
    if (enabled("github"))        table.push_back({"github", github});

    std::vector<SearchResult> all;
    std::set<std::string> seen;
    for (auto& [name, fn] : table) {
        auto& d = diags[name];
        try {
            auto res = fn(query, opt);
            d.results += (int)res.size();
            d.status = res.empty() ? (d.status == "ok" ? "ok" : "empty") : "ok";
            for (auto& s : res) {
                if (s.url.empty() || !seen.insert(s.url).second) continue;
                all.push_back(std::move(s));
            }
        } catch (const std::exception& e) {
            if (d.status != "ok") d.status = std::string("error: ") + e.what();
        }
    }
    return all;
}

} // namespace laplace::research
