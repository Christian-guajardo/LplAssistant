#pragma once
#include <map>
#include <string>

// Client HTTP minimal du module recherche : httplib pour le HTTP simple
// (localhost : llama-server, SearXNG) et repli sur le binaire `curl` pour le
// HTTPS quand le support SSL n'est pas compilé (cas WSL sans openssl-dev).
namespace laplace::research {

struct HttpResponse {
    int         status = 0;   // 0 = échec transport (voir error)
    std::string body;
    std::string error;
};

// GET avec redirections suivies. `compressed` active la décompression gzip
// (obligatoire pour l'API StackExchange).
HttpResponse http_get(const std::string& url, int timeout_s = 15,
                      const std::map<std::string, std::string>& headers = {},
                      bool compressed = false);

// HEAD léger pour les quality gates (statut seul, pas de corps).
HttpResponse http_head(const std::string& url, int timeout_s = 8);

// POST JSON (Content-Type: application/json), utilisé pour llama-server.
HttpResponse http_post_json(const std::string& url, const std::string& json_body,
                            int timeout_s = 300);

// Encodage percent d'un paramètre de query string.
std::string url_encode(const std::string& s);

} // namespace laplace::research
