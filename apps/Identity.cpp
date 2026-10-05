/**
 * @file Identity.cpp
 * @brief Implementation of what a tool says about itself with --version.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include "Identity.hpp"

#include <lpl/Foundation.hpp>

namespace lpl::apps {

void printIdentity(std::FILE *out, const char *tool)
{
    std::fprintf(out, "%s (%s) %s+%s %s\n", tool, LPLASSISTANT_NAME, LPLASSISTANT_VERSION_STRING, LPLASSISTANT_BUILD,
                 LPLASSISTANT_COMMIT);
#if defined(LPL_HAS_FOUNDATION)
    std::fprintf(out, "  built with %s %s %s\n", LPLPLUGIN_NAME, LPLPLUGIN_VERSION_STRING, LPLPLUGIN_COMMIT);
#else
    std::fprintf(out, "  built without LplPlugin\n");
#endif
    std::fprintf(out, "  for %s on %s, compiled by %s\n", LPLASSISTANT_SYSTEM_STRING, LPLASSISTANT_ARCH_STRING,
                 LPLASSISTANT_COMPILER_STRING);
}

} // namespace lpl::apps
