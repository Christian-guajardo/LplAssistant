/**
 * @file OpenAccess.hpp
 * @brief Resolving a work to a legally free copy.
 *
 * The project's rule is that the open path is the only path. This looks for a
 * legitimate free version and records that it looked; it never circumvents.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_OPENACCESS_HPP
#    define LPL_RESEARCH_OPENACCESS_HPP

#    include <string>

// Résolution Open Access : un lien doi.org/ (ou une URL d'éditeur payant) est
// souvent muré (paywall, challenge anti-bot Cloudflare) et illisible par un
// simple GET. Unpaywall (API gratuite) mappe un DOI vers sa meilleure copie en
// accès libre ; on ne bascule que si cette copie est dans un DÉPÔT (arXiv, PMC,
// institutionnel) — pas chez l'éditeur, qui reste muré.
//
// On ne cherche jamais à contourner une vérification anti-bot : on contourne le
// paywall en préférant une copie légalement ouverte, rien de plus.
namespace lpl::research {

// Si url est un DOI (ou contient un DOI) et qu'Unpaywall connaît une copie OA
// en dépôt, renvoie cette URL (PDF de préférence). Sinon renvoie url inchangée.
// email : requis par l'API Unpaywall (LAPLACE_UNPAYWALL_EMAIL). Vide => no-op.
std::string resolve_open_access(const std::string& url, const std::string& email,
                                int timeoutSeconds = 15);

} // namespace lpl::research

#endif // LPL_RESEARCH_OPENACCESS_HPP
