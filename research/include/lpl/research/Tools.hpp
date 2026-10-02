/**
 * @file Tools.hpp
 * @brief Deep research as a TOOL SURFACE, not as a command.
 *
 * The distinction matters and it is the author's: this module is not a program a
 * human runs, it is a library of capabilities an intelligence calls. `search`,
 * `read`, `reflect` and `answer` already exist as typed actions inside the engine's
 * state machine; this header lifts them to the same tool vocabulary the rest of the
 * project uses to let a model act — one declaration, a JSON schema derived from it,
 * and a grammar derived from that.
 *
 * It is the same inversion as everywhere else in Laplace, applied to documentary
 * research: the intelligence decides WHAT to look for, deterministic C++ decides HOW
 * to look, and the tokens spent are a handful of tool calls rather than a corpus
 * poured into a context window.
 *
 * Two properties carry over from the engine and must not be lost at this boundary:
 *
 *   - the grammar is REGENERATED per step, so a tool that is not valid right now is
 *     not merely discouraged, it is unrepresentable. Answering before reading is not
 *     a behaviour to be discouraged by a prompt; it is a call the sampler cannot emit.
 *   - a tool result is BOUNDED and says when it truncated. The full text of a source
 *     goes to the disk cache under a content hash and only a skim enters the prompt,
 *     with an identifier that can fetch any slice back later. Nothing is ever
 *     truncated and discarded — that would destroy evidence, which is the one thing
 *     a research tool exists to preserve.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RESEARCH_TOOLS_HPP
#    define LPL_RESEARCH_TOOLS_HPP

#    include <lpl/research/Engine.hpp>

#    include <string>
#    include <vector>

namespace lpl::research {

// TODO(lot 8): the tool descriptors, and their binding to an agent dispatcher.
//
// Shape to aim for, so that the engine keeps ONE state machine and this is only a
// second way in — never a second implementation:
//
//   searchTool(query, providers)      -> bounded result list + per-provider status
//   readTool(url)                     -> cache identifier + skim view
//   retrieveTool(cacheId, offset, n)  -> any slice of a source already read
//   reflectTool()                     -> the named gaps, as sub-questions
//   answerTool()                      -> an evaluated answer, or the critique
//   reportTool()                      -> the written report's path

} // namespace lpl::research

#endif // LPL_RESEARCH_TOOLS_HPP
