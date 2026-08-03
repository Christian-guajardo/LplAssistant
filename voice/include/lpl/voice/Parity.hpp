/**
 * @file Parity.hpp
 * @brief The constexpr buffer both targets fingerprint.
 *
 * The same PCM must yield the same signature on the host and in ring 0 — which is
 * why the arithmetic is fixed-point rather than float.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOICE_PARITY_HPP
#    define LPL_VOICE_PARITY_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::voice {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::voice

#endif // LPL_VOICE_PARITY_HPP
