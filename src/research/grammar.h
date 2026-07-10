#pragma once
#include <string>
#include <vector>

// Générateurs de grammaires GBNF : la sortie JSON de chaque étape de la
// machine à états est contrainte physiquement au niveau du sampler.
// Principe clé (jina + gating) : la grammaire de décision est régénérée à
// chaque étape et n'expose QUE les actions autorisées — une action interdite
// est ingénérable, pas seulement déconseillée.
namespace laplace::research {

// {"action": "<une des actions autorisées>", "arg": "<chaîne>"}
std::string decision_grammar(const std::vector<std::string>& allowed_actions);

// {"<key1>": ["...", ...], "<key2>": ["...", ...]} — listes de chaînes.
std::string string_lists_grammar(const std::vector<std::string>& keys);

// {"pass": true|false, "reason": "..."} — évaluateur de réponse.
std::string evaluation_grammar();

} // namespace laplace::research
