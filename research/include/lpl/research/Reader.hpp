/**
 * @file Reader.hpp
 * @brief HTML to readable text, locally.
 *
 * Strips scripts, styles, navigation and footers, then decodes entities. No
 * JavaScript is executed, so pages that render entirely client-side come out empty
 * and are classified as failures by the caller — degrading cleanly rather than
 * injecting garbage into the prompt.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_READER_HPP
#    define LPL_RESEARCH_READER_HPP

#    include <string>

// Lecteur local HTML -> texte : l'équivalent maison de Jina Reader.
// Supprime scripts/styles/nav/footer, convertit les blocs en lignes,
// décode les entités. Aucun JavaScript exécuté : les sites full-JS
// sortent vides et sont classés en échec par l'appelant (dégradation propre).
namespace lpl::research {

struct PageText {
    std::string title;
    std::string text; // texte lisible, lignes séparées par \n
};

PageText htmlToText(const std::string& html);

// Heuristique : le contenu ressemble-t-il à du HTML ? (sinon on le garde brut :
// texte, markdown, JSON d'API...)
bool looksLikeHtml(const std::string& body);

// Le corps est-il un PDF ? (signature « %PDF- » en tête, tolérant à un BOM/espaces).
bool looksLikePortableDocument(const std::string& body);

// Extraction texte d'un PDF via `pdftotext` (poppler-utils), en shell-out comme
// curl. text vide si pdftotext est absent ou échoue -> l'appelant classe la
// source en échec (dégradation propre, jamais de charabia binaire injecté).
PageText portableDocumentToText(const std::string& pdf_bytes);

} // namespace lpl::research

#endif // LPL_RESEARCH_READER_HPP
