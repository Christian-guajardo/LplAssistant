/**
 * @file Grammar.hpp
 * @brief Grammars that make an illegal decision unrepresentable.
 *
 * Regenerated at every step so that only the currently permitted actions appear.
 * An action that is not allowed is not merely discouraged: the sampler cannot emit
 * it. That is what makes a small local model trustworthy as a decision maker.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_GRAMMAR_HPP
#    define LPL_RESEARCH_GRAMMAR_HPP

#    include <string>
#    include <vector>

// Générateurs de grammaires GBNF : la sortie JSON de chaque étape de la
// machine à états est contrainte physiquement au niveau du sampler.
// Principe clé (jina + gating) : la grammaire de décision est régénérée à
// chaque étape et n'expose QUE les actions autorisées — une action interdite
// est ingénérable, pas seulement déconseillée.
namespace lpl::research {

// {"action": "<une des actions autorisées>", "arg": "<chaîne>"}
std::string decisionGrammar(const std::vector<std::string>& allowed_actions);

// {"<key1>": ["...", ...], "<key2>": ["...", ...]} — listes de chaînes.
std::string stringListsGrammar(const std::vector<std::string>& keys);

// {"pass": true|false, "reason": "..."} — évaluateur de réponse.
std::string evaluationGrammar();

} // namespace lpl::research

#endif // LPL_RESEARCH_GRAMMAR_HPP
