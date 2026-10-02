/**
 * @file CompressCacheRetrieve.cpp
 * @brief Implementation of full text to disk, a bounded view to the prompt.
 *
 *  
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/research/CompressCacheRetrieve.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace lpl::research {

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

CompressCacheRetrieve::CompressCacheRetrieve(std::string cache_dir) : dir_(std::move(cache_dir)) {
    fs::create_directories(dir_);
}

CompressCacheRetrieve::Stored CompressCacheRetrieve::store(const std::string& url, const std::string& title,
                       const std::string& full_text, size_t skimCharacters) {
    Stored s;
    s.id          = fnv1a_hex(url + "\n" + full_text);
    s.totalCharacters = full_text.size();

    std::ofstream f(dir_ + "/" + s.id + ".txt", std::ios::binary);
    f << "URL: " << url << "\nTITLE: " << title << "\n\n" << full_text;

    // Vue écrémée : début du texte + plan des titres (lignes # markdown que le
    // lecteur HTML a préservées) pour que l'agent sache quoi demander en retrieve.
    std::ostringstream skim;
    if (!title.empty()) skim << "# " << title << "\n";
    skim << full_text.substr(0, std::min(skimCharacters, full_text.size()));
    if (full_text.size() > skimCharacters) {
        skim << "\n\n[--- texte écrémé : " << full_text.size()
             << " caractères au total, cache id=" << s.id
             << " — plan des sections restantes :";
        std::istringstream lines(full_text.substr(skimCharacters));
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

std::string CompressCacheRetrieve::retrieve(const std::string& id, size_t offset, size_t len) const {
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

} // namespace lpl::research
