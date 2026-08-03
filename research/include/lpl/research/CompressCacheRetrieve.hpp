/**
 * @file CompressCacheRetrieve.hpp
 * @brief Full text to disk, a bounded view to the prompt.
 *
 * The rule that keeps long sources usable: nothing is ever truncated and thrown
 * away. The whole page goes to a disk cache under a content hash, and only a bounded
 * skim enters the prompt — with an identifier that lets any slice of the original be
 * fetched back later. Truncation destroys evidence; this defers it instead.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_COMPRESSCACHERETRIEVE_HPP
#    define LPL_RESEARCH_COMPRESSCACHERETRIEVE_HPP

#    include <string>

// CCR — Compress-Cache-Retrieve (protocole honey-for-devs).
// Le texte intégral d'une page scrapée n'est JAMAIS tronqué-détruit :
// il part en cache disque, et seul un « écrémage » borné entre dans le
// prompt, avec un identifiant permettant de récupérer n'importe quelle
// tranche du texte original.
namespace lpl::research {

class CompressCacheRetrieve {
public:
    explicit CompressCacheRetrieve(std::string cache_dir);

    struct Stored {
        std::string id;    // hash FNV-1a du contenu (hex)
        std::string skim;  // vue écrémée bornée pour le prompt
        size_t      totalCharacters = 0;
    };

    // Enregistre le texte intégral et fabrique la vue écrémée :
    // titre + premières lignes + plan des titres markdown détectés.
    Stored store(const std::string& url, const std::string& title,
                 const std::string& full_text, size_t skimCharacters = 6000);

    // Tranche du texte original (pour l'action retrieve de l'agent).
    std::string retrieve(const std::string& id, size_t offset, size_t len) const;

    const std::string& dir() const { return dir_; }

private:
    std::string dir_;
};

} // namespace lpl::research

#endif // LPL_RESEARCH_COMPRESSCACHERETRIEVE_HPP
