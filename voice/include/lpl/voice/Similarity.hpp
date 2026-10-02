/**
 * @file Similarity.hpp
 * @brief A vocal signature, and how two of them are compared.
 *
 * The signature is a plain vector of coefficients: per-band spectral mean and
 * deviation (the timbre) followed by mean and deviation of pitch. Comparison is
 * cosine, so the result is invariant to overall loudness — which is the property
 * that lets the same person be recognised near the microphone and across the room.
 *
 * The threshold is a household setting, not a security parameter. Too low and two
 * siblings share one conversation; too high and walking into the next room loses it.
 * This is explicitly not biometric authentication and the code should keep saying so.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOICE_SIMILARITY_HPP
#    define LPL_VOICE_SIMILARITY_HPP

#    include <cstdint>
#    include <vector>

namespace lpl::voice {

/// Per-band spectral statistics followed by pitch statistics.
using Signature = std::vector<float>;

/// Default acceptance threshold. Overridable per household.
constexpr float kDefaultThreshold = 0.80f;

/// Cosine similarity in [-1, 1]; -1 when the sizes disagree or either is empty.
/// Mismatched sizes are a legitimate runtime case (a profile enrolled by an older
/// build), so this reports rather than asserts.
[[nodiscard]] float cosineSimilarity(const Signature &left, const Signature &right);

} // namespace lpl::voice

#endif // LPL_VOICE_SIMILARITY_HPP
