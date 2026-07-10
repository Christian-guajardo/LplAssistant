#pragma once
#include <string>

// CCR — Compress-Cache-Retrieve (protocole honey-for-devs).
// Le texte intégral d'une page scrapée n'est JAMAIS tronqué-détruit :
// il part en cache disque, et seul un « écrémage » borné entre dans le
// prompt, avec un identifiant permettant de récupérer n'importe quelle
// tranche du texte original.
namespace laplace::research {

class Ccr {
public:
    explicit Ccr(std::string cache_dir);

    struct Stored {
        std::string id;    // hash FNV-1a du contenu (hex)
        std::string skim;  // vue écrémée bornée pour le prompt
        size_t      total_chars = 0;
    };

    // Enregistre le texte intégral et fabrique la vue écrémée :
    // titre + premières lignes + plan des titres markdown détectés.
    Stored store(const std::string& url, const std::string& title,
                 const std::string& full_text, size_t skim_chars = 6000);

    // Tranche du texte original (pour l'action retrieve de l'agent).
    std::string retrieve(const std::string& id, size_t offset, size_t len) const;

    const std::string& dir() const { return dir_; }

private:
    std::string dir_;
};

} // namespace laplace::research
