/**
 * @file GrammarSampler.hpp
 * @brief Constrained decoding against a grammar.
 *
 * The sampler masks every token the grammar forbids, so a malformed tool call is
 * not unlikely — it is unrepresentable. This is what makes a small local model
 * reliable enough to drive an engine.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_GRAMMARSAMPLER_HPP
#    define LPL_LPL_INFER_GRAMMARSAMPLER_HPP

#    include <lpl/Foundation.hpp>

namespace lpl::infer {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::infer

#endif // LPL_LPL_INFER_GRAMMARSAMPLER_HPP
