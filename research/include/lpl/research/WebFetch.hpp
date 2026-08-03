/**
 * @file WebFetch.hpp
 * @brief Fetching a URL, with the timeouts that matter.
 *
 * Named for what it does rather than for the protocol's acronym. Bounded
 * everywhere: a research run that hangs on one unresponsive host has failed even if
 * it eventually recovers.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_WEBFETCH_HPP
#    define LPL_RESEARCH_WEBFETCH_HPP

#    include <map>
#    include <string>

// Client HTTP minimal du module recherche : httplib pour le HTTP simple
// (localhost : llama-server, SearXNG) et repli sur le binaire `curl` pour le
// HTTPS quand le support SSL n'est pas compilé (cas WSL sans openssl-dev).
namespace lpl::research {

struct HttpResponse {
    int         status = 0;   // 0 = échec transport (voir error)
    std::string body;
    std::string error;
};

// GET avec redirections suivies. `compressed` active la décompression gzip
// (obligatoire pour l'API StackExchange).
HttpResponse fetchUrl(const std::string& url, int timeoutSeconds = 15,
                      const std::map<std::string, std::string>& headers = {},
                      bool compressed = false);

// HEAD léger pour les quality gates (statut seul, pas de corps).
HttpResponse headUrl(const std::string& url, int timeoutSeconds = 8);

// POST JSON (Content-Type: application/json), utilisé pour llama-server.
HttpResponse http_post_json(const std::string& url, const std::string& json_body,
                            int timeoutSeconds = 300);

// Encodage percent d'un paramètre de query string.
std::string url_encode(const std::string& s);

} // namespace lpl::research

#endif // LPL_RESEARCH_WEBFETCH_HPP
