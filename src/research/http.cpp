#include "http.h"

#include <httplib.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace laplace::research {

namespace {

constexpr const char* kUserAgent =
    "Mozilla/5.0 (X11; Linux x86_64) LaplaceResearch/0.1";

struct ParsedUrl {
    std::string scheme, host_port, path; // path inclut la query
    bool        ok = false;
};

ParsedUrl parse_url(const std::string& url) {
    ParsedUrl p;
    auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) return p;
    p.scheme = url.substr(0, scheme_end);
    auto rest = url.substr(scheme_end + 3);
    auto slash = rest.find('/');
    p.host_port = slash == std::string::npos ? rest : rest.substr(0, slash);
    p.path      = slash == std::string::npos ? "/" : rest.substr(slash);
    p.ok        = !p.host_port.empty();
    return p;
}

// Neutralise une URL avant interpolation dans une ligne de commande shell :
// entourée de quotes simples, avec échappement des quotes internes. Les
// caractères de contrôle sont retirés (une URL n'en contient jamais de licite).
std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if ((unsigned char)c < 0x20 || c == 0x7F) continue;
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

// Exécute curl et sépare le corps du code HTTP final (écrit en dernière ligne
// via -w). Utilisé pour tout le HTTPS : redirections, gzip et TLS gérés par
// curl, aucune dépendance de compilation.
HttpResponse curl_request(const std::string& args, const std::string& url,
                          int timeout_s) {
    HttpResponse r;
    std::ostringstream cmd;
    cmd << "curl -sL --max-time " << timeout_s
        << " -A " << shell_quote(kUserAgent)
        << " " << args
        << " -w '\\n%{http_code}'"
        << " " << shell_quote(url) << " 2>/dev/null";
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) { r.error = "popen curl a échoué"; return r; }
    std::string out;
    std::array<char, 8192> buf;
    size_t n;
    while ((n = fread(buf.data(), 1, buf.size(), pipe)) > 0)
        out.append(buf.data(), n);
    int rc = pclose(pipe);
    auto last_nl = out.rfind('\n');
    if (last_nl == std::string::npos) {
        r.error = "sortie curl vide (rc=" + std::to_string(rc) + ")";
        return r;
    }
    r.status = std::atoi(out.c_str() + last_nl + 1);
    r.body   = out.substr(0, last_nl);
    if (r.status == 0)
        r.error = "échec transport curl (rc=" + std::to_string(rc) + ")";
    return r;
}

HttpResponse httplib_get(const ParsedUrl& p, int timeout_s,
                         const httplib::Headers& headers) {
    HttpResponse r;
    httplib::Client cli(("http://" + p.host_port).c_str());
    cli.set_connection_timeout(timeout_s);
    cli.set_read_timeout(timeout_s);
    cli.set_follow_location(true);
    auto res = cli.Get(p.path, headers);
    if (!res) { r.error = "httplib: " + httplib::to_string(res.error()); return r; }
    r.status = res->status;
    r.body   = res->body;
    return r;
}

} // namespace

// httplib (sans TLS) ne convient qu'aux services locaux (llama-server, SearXNG).
// Tout hôte distant en http:// peut rediriger vers https:// — que httplib ne
// sait pas suivre — donc on le confie à curl. arXiv en est le cas typique
// (http://arxiv.org/abs/... -> 301 https://).
static bool is_local_host(const std::string& host_port) {
    return host_port.rfind("127.0.0.1", 0) == 0 ||
           host_port.rfind("localhost", 0) == 0 ||
           host_port.rfind("0.0.0.0", 0) == 0 ||
           host_port.rfind("[::1]", 0) == 0;
}

std::string url_encode(const std::string& s) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
            out += (char)c;
        else if (c == ' ')
            out += '+';
        else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

HttpResponse http_get(const std::string& url, int timeout_s,
                      const std::map<std::string, std::string>& headers,
                      bool compressed) {
    ParsedUrl p = parse_url(url);
    if (!p.ok) return {0, "", "URL invalide : " + url};

    if (p.scheme == "http" && !compressed && is_local_host(p.host_port)) {
        httplib::Headers h{{"User-Agent", kUserAgent}};
        for (auto& [k, v] : headers) h.emplace(k, v);
        return httplib_get(p, timeout_s, h);
    }
    // HTTPS, http distant (redirections possibles) ou gzip requis : curl.
    std::string args = compressed ? "--compressed" : "";
    for (auto& [k, v] : headers)
        args += " -H " + shell_quote(k + ": " + v);
    return curl_request(args, url, timeout_s);
}

HttpResponse http_head(const std::string& url, int timeout_s) {
    ParsedUrl p = parse_url(url);
    if (!p.ok) return {0, "", "URL invalide : " + url};
    // -I peut être refusé (405) : les gates re-tentent alors un GET.
    return curl_request("-I -o /dev/null", url, timeout_s);
}

HttpResponse http_post_json(const std::string& url, const std::string& json_body,
                            int timeout_s) {
    ParsedUrl p = parse_url(url);
    if (!p.ok) return {0, "", "URL invalide : " + url};

    if (p.scheme == "http" && is_local_host(p.host_port)) {
        HttpResponse r;
        httplib::Client cli(("http://" + p.host_port).c_str());
        cli.set_connection_timeout(timeout_s);
        cli.set_read_timeout(timeout_s); // la génération LLM peut être longue
        auto res = cli.Post(p.path, json_body, "application/json");
        if (!res) { r.error = "httplib: " + httplib::to_string(res.error()); return r; }
        r.status = res->status;
        r.body   = res->body;
        return r;
    }
    // POST HTTPS : le corps passe par un heredoc pour éviter les limites argv.
    std::string args = "-X POST -H 'Content-Type: application/json' --data-binary @-";
    HttpResponse r;
    std::ostringstream cmd;
    cmd << "curl -sL --max-time " << timeout_s << " " << args
        << " -w '\\n%{http_code}' " << shell_quote(url) << " 2>/dev/null";
    FILE* pipe = popen(("printf '%s' " + shell_quote(json_body) + " | " + cmd.str()).c_str(), "r");
    if (!pipe) { r.error = "popen curl a échoué"; return r; }
    std::string out;
    std::array<char, 8192> buf;
    size_t n;
    while ((n = fread(buf.data(), 1, buf.size(), pipe)) > 0)
        out.append(buf.data(), n);
    pclose(pipe);
    auto last_nl = out.rfind('\n');
    if (last_nl == std::string::npos) { r.error = "sortie curl vide"; return r; }
    r.status = std::atoi(out.c_str() + last_nl + 1);
    r.body   = out.substr(0, last_nl);
    return r;
}

} // namespace laplace::research
