#include "ccr.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace laplace::research {

namespace {

std::string fnv1a_hex(const std::string& data) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : data) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return buf;
}

} // namespace

Ccr::Ccr(std::string cache_dir) : dir_(std::move(cache_dir)) {
    fs::create_directories(dir_);
}

Ccr::Stored Ccr::store(const std::string& url, const std::string& title,
                       const std::string& full_text, size_t skim_chars) {
    Stored s;
    s.id          = fnv1a_hex(url + "\n" + full_text);
    s.total_chars = full_text.size();

    std::ofstream f(dir_ + "/" + s.id + ".txt", std::ios::binary);
    f << "URL: " << url << "\nTITLE: " << title << "\n\n" << full_text;

    // Vue écrémée : début du texte + plan des titres (lignes # markdown que le
    // lecteur HTML a préservées) pour que l'agent sache quoi demander en retrieve.
    std::ostringstream skim;
    if (!title.empty()) skim << "# " << title << "\n";
    skim << full_text.substr(0, std::min(skim_chars, full_text.size()));
    if (full_text.size() > skim_chars) {
        skim << "\n\n[--- texte écrémé : " << full_text.size()
             << " caractères au total, cache id=" << s.id
             << " — plan des sections restantes :";
        std::istringstream lines(full_text.substr(skim_chars));
        std::string line;
        int shown = 0;
        while (std::getline(lines, line) && shown < 20) {
            if (!line.empty() && line[0] == '#') {
                skim << "\n  " << line;
                ++shown;
            }
        }
        skim << " ---]";
    }
    s.skim = skim.str();
    return s;
}

std::string Ccr::retrieve(const std::string& id, size_t offset, size_t len) const {
    // Refuse tout id qui ne ressemble pas à un hash (pas de traversée de chemin).
    for (char c : id)
        if (!std::isxdigit((unsigned char)c)) return {};
    std::ifstream f(dir_ + "/" + id + ".txt", std::ios::binary);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    std::string all = ss.str();
    if (offset >= all.size()) return {};
    return all.substr(offset, std::min(len, all.size() - offset));
}

} // namespace laplace::research
