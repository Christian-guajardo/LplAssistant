/**
 * @file Identity.hpp
 * @brief What a tool of this repository says about itself with --version.
 *
 * Everything it prints is compiled in: the version from include/lplassistant/config.h,
 * the commits and the build stamped by the build into Identity.cpp alone. A copy of the
 * binary far from any repository still knows what it is.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#include <cstdio>

namespace lpl::apps {

/**
 * @brief Prints which tool this is, the version, commit and build of LplAssistant, and the
 *        LplPlugin it was built with.
 *
 * @param out  Where to print.
 * @param tool The tool's name, as typed on the command line.
 */
void printIdentity(std::FILE *out, const char *tool);

} // namespace lpl::apps
