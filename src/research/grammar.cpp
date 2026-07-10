#include "grammar.h"

#include <sstream>

namespace laplace::research {

namespace {

// Primitives partagées : chaîne JSON (échappements standards, sans répétition
// bornée {m,n} pour rester compatible avec les vieux parseurs GBNF) et blancs.
const char* kCommonRules = R"GBNF(
string ::= "\"" chars "\""
chars ::= char chars | ""
char ::= [^"\\] | "\\" esc
esc ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
ws ::= [ \t\n]*
)GBNF";

} // namespace

std::string decision_grammar(const std::vector<std::string>& allowed_actions) {
    std::ostringstream g;
    g << "root ::= \"{\" ws \"\\\"action\\\"\" ws \":\" ws action ws \",\" ws "
         "\"\\\"arg\\\"\" ws \":\" ws string ws \"}\"\n";
    g << "action ::= ";
    for (size_t i = 0; i < allowed_actions.size(); ++i) {
        if (i) g << " | ";
        g << "\"\\\"" << allowed_actions[i] << "\\\"\"";
    }
    g << "\n" << kCommonRules;
    return g.str();
}

std::string string_lists_grammar(const std::vector<std::string>& keys) {
    std::ostringstream g;
    g << "root ::= \"{\" ws ";
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i) g << "\",\" ws ";
        g << "\"\\\"" << keys[i] << "\\\"\" ws \":\" ws strlist ws ";
    }
    g << "\"}\"\n";
    g << "strlist ::= \"[\" ws items \"]\" | \"[\" ws \"]\"\n";
    g << "items ::= string ws \",\" ws items | string ws\n";
    g << kCommonRules;
    return g.str();
}

std::string evaluation_grammar() {
    std::ostringstream g;
    g << "root ::= \"{\" ws \"\\\"pass\\\"\" ws \":\" ws bool ws \",\" ws "
         "\"\\\"reason\\\"\" ws \":\" ws string ws \"}\"\n";
    g << "bool ::= \"true\" | \"false\"\n";
    g << kCommonRules;
    return g.str();
}

} // namespace laplace::research
