/**
 * @file Parity.hpp
 * @brief The constexpr exchange all three implementations must agree on.
 *
 * A fixed sequence of datagrams must produce the same decisions on the host oracle
 * and in ring 0. Three consumers of one protocol is exactly the situation where a
 * gate stops being bureaucracy.
 *
 * What is folded is a whole exchange and not a function: silence, a wake word, an
 * utterance, the hangover that closes it, a reply playing, and that reply coming back
 * into the microphone. Every decision a node makes appears in it — when to start
 * sending, when to stop, when the word was heard, when it is hearing itself, and what
 * state it was in while all that happened.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_PARITY_HPP
#    define LPL_SATELLITE_PARITY_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/satellite/Protocol.hpp>

namespace lpl::satellite {

/**
 * @brief Frames the canonical exchange runs for.
 *
 * Eighty at forty milliseconds is 3.2 seconds — long enough to contain a seven
 * hundred millisecond hangover with room either side, which is what makes the
 * utterance actually close rather than run off the end of the timeline.
 */
[[nodiscard]] constexpr core::u32 parityFrameCount() noexcept { return 80u; }

/// First frame of the wake word in the canonical timeline.
[[nodiscard]] constexpr core::u32 parityWakeFirstFrame() noexcept { return 10u; }

/// Frames the wake word spans, and therefore the template's length.
[[nodiscard]] constexpr core::u32 parityWakeFrames() noexcept { return 6u; }

/**
 * @brief Average per-band distance still counted as the wake word.
 *
 * Zero would work on a synthetic timeline and would be a gate that proves nothing
 * about a real room. Ninety-six out of a shape that sums to 1024 is roughly a tenth
 * of the spectrum moving, which is what a different distance from the microphone
 * does to these bands.
 */
[[nodiscard]] constexpr core::u32 parityWakeTolerance() noexcept { return 96u; }

/// Distance below which a capture is the node hearing its own playback.
[[nodiscard]] constexpr core::u32 parityEchoTolerance() noexcept { return 96u; }

/**
 * @brief Writes one frame of the canonical timeline.
 *
 * Deterministic from the frame index alone — no state, no clock, no stream carried
 * between calls. That is what lets the kernel and the host generate the same audio
 * without shipping a wave file to both, and it is the same reason the world gate
 * derives a world from a seed instead of loading one.
 *
 * The timeline is: silence, a wake word, an utterance, the hangover, a reply playing
 * back into the room, then silence again.
 *
 * @param frame   Index below @ref parityFrameCount.
 * @param out     Receives @p count samples.
 * @param count   Samples wanted; @ref kFrameSamples in the canonical run.
 */
void synthesiseFrame(core::u32 frame, core::i16 *out, core::u32 count) noexcept;

/**
 * @brief Is this frame part of the reply the node plays?
 * @param frame Index.
 * @return true while playback is active.
 */
[[nodiscard]] bool parityFrameIsPlayback(core::u32 frame) noexcept;

/**
 * @struct SatelliteFoldResult
 * @brief The signatures the kernel must reproduce.
 *
 * Free of Fixed32 and bool, like every other fold result in the project, so a kernel
 * smoke can copy it field by field into a plain C struct.
 */
struct SatelliteFoldResult {
    core::u32 featureSignature{0u};  ///< Fold of every frame's spectral shape.
    core::u32 levelSignature{0u};    ///< Fold of every frame's root-mean-square level.
    core::u32 eventSignature{0u};    ///< Fold of the voice-activity decisions.
    core::u32 wireSignature{0u};     ///< Fold of every datagram the node emitted.
    core::u32 stateSignature{0u};    ///< Fold of the power state after each frame.
    core::u32 templateSignature{0u}; ///< Fold of the armed wake-word template.
    core::u32 framesEmitted{0u};     ///< Audio datagrams sent.
    core::u32 utterances{0u};        ///< Utterances the detector closed.
    core::u32 detections{0u};        ///< Times the wake word matched.
    core::u32 wakeFrame{0u};         ///< Frame at which it first matched.
    core::u32 wakeDistance{0u};      ///< Distance at that match.
    core::u32 echoesRejected{0u};    ///< Captures identified as the node's own voice.
    core::u32 speechDistance{0u};    ///< Wake distance during the utterance — must be far.
    core::u32 transitions{0u};       ///< Power state changes.
    core::u32 idlePermille{0u};      ///< Share of the timeline spent idle, in thousandths.
    core::u32 dutyPermille{0u};      ///< Share of the timeline the processor was awake.
    core::u32 taggedAudioIsAudio{0u};///< 1 when a payload beginning "TXT:" still decodes as audio.
};

/**
 * @brief Runs the canonical exchange and folds every stage of it.
 *
 * @param out Receives the signatures.
 */
void foldSatelliteState(SatelliteFoldResult &out);

} // namespace lpl::satellite

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_SATELLITE_PARITY_HPP
