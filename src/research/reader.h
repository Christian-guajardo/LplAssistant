#pragma once
#include <string>

// Lecteur local HTML -> texte : l'équivalent maison de Jina Reader.
// Supprime scripts/styles/nav/footer, convertit les blocs en lignes,
// décode les entités. Aucun JavaScript exécuté : les sites full-JS
// sortent vides et sont classés en échec par l'appelant (dégradation propre).
namespace laplace::research {

struct PageText {
    std::string title;
    std::string text; // texte lisible, lignes séparées par \n
};

PageText html_to_text(const std::string& html);

// Heuristique : le contenu ressemble-t-il à du HTML ? (sinon on le garde brut :
// texte, markdown, JSON d'API...)
bool looks_like_html(const std::string& body);

} // namespace laplace::research
