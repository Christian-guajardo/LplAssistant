/**
 * @file Voiceprint.hpp
 * @brief Extracting a lightweight vocal signature from raw audio.
 *
 * Long-term spectral statistics plus a pitch estimate — no neural model. The need
 * came before the solution, and the need is only to tell a few members of a
 * household apart so that the right conversation continues when someone walks into
 * another room.
 *
 * Deliberately unchanged from the original implementation: it works, and replacing
 * it with a learned embedding would buy accuracy nobody asked for at a cost ring 0
 * cannot pay. What DID change is where it lives — this module is freestanding, so a
 * satellite could eventually identify a speaker without a round trip.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOICE_VOICEPRINT_HPP
#    define LPL_VOICE_VOICEPRINT_HPP

#    include <lpl/voice/Similarity.hpp>

#    include <cstdint>
#    include <vector>

namespace lpl::voice {

/// Sample rate every stage of the voice path agrees on.
constexpr int kSampleRate = 16000;

/// Computes a signature from pulse-code-modulated 16-bit mono audio at
/// @ref kSampleRate. Returns an EMPTY signature when the utterance is too short
/// or too noisy to characterise — an empty result is a legitimate answer here and
/// callers must treat it as "unknown speaker", never as a zero vector.
[[nodiscard]] Signature computeVoiceprint(const std::vector<std::int16_t> &pulseCodeModulation);

} // namespace lpl::voice

#endif // LPL_VOICE_VOICEPRINT_HPP
