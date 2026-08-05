/**
 * @file VoiceActivity.hpp
 * @brief Deciding that someone is speaking, on energy alone.
 *
 * Hysteresis plus a hangover window: two thresholds so a pause inside a sentence
 * does not end the utterance, and a tail so the last syllable survives. No model,
 * no allocation — this has to run on a microcontroller with a few hundred kilobytes
 * and it has to run all the time.
 *
 * "All the time" is the constraint that shapes everything here. This is the only
 * code on a satellite that never sleeps, so its cost per frame is the node's idle
 * power. One pass over the samples, one integer square root, no transcendental and
 * no second buffer.
 *
 * The levels are Fixed32 fractions of full scale rather than floats, for the usual
 * reason and one more: they decide whether a datagram leaves the node, so two nodes
 * rounding differently would disagree about whether anyone spoke.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_VOICEACTIVITY_HPP
#    define LPL_SATELLITE_VOICEACTIVITY_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/math/FixedPoint.hpp>

namespace lpl::satellite {

/**
 * @struct VoiceActivityParams
 * @brief The four numbers a detector is.
 *
 * Every value is the one the working hosted node uses, kept rather than re-tuned:
 * they were arrived at against a real microphone in a real room, and a fresh guess
 * would only be a fresh guess.
 */
struct VoiceActivityParams {
    /// Level at which an utterance starts. 0.020 of full scale.
    math::Fixed32 startLevel{math::Fixed32::fromRaw(1311)};

    /// Level below which silence accumulates. 0.012 — lower than @c startLevel, and
    /// that gap IS the hysteresis: a pause between words sits between the two and
    /// neither restarts nor ends the utterance.
    math::Fixed32 keepLevel{math::Fixed32::fromRaw(786)};

    /// Both levels are multiplied by this while the node's own speaker is playing.
    /// 2.5, because the speaker excites the microphone and only a nearby voice —
    /// somebody actually interrupting — should get through.
    math::Fixed32 playbackBoost{math::Fixed32::fromRaw(163840)};

    /// Silence tolerated before the utterance is closed.
    core::u32 hangoverMilliseconds{700u};

    /// Frames the caller must keep so the attack is not clipped off the front.
    core::u32 prerollFrames{4u};
};

/**
 * @enum VoiceEvent
 * @brief What a frame changed.
 */
enum class VoiceEvent : core::u32 {
    Silence = 0u,        ///< Nothing, and nothing was happening.
    UtteranceBegan = 1u, ///< This frame crossed the start level: flush the preroll.
    Speaking = 2u,       ///< Inside an utterance: send this frame.
    UtteranceEnded = 3u, ///< The hangover expired: send the end-of-utterance datagram.
};

/**
 * @brief Root-mean-square level of a frame, as a fraction of full scale.
 *
 * Accumulated in 64 bits and rooted once. Squaring 640 samples of a 16-bit signal
 * needs 39 bits before the mean, so a 32-bit accumulator would wrap on a loud frame
 * — and the wrap would read as silence, which is the worst possible direction for
 * the error to go.
 *
 * @param samples PCM16 mono.
 * @param count   How many.
 * @return The level in [0, 1].
 */
[[nodiscard]] math::Fixed32 frameLevel(const core::i16 *samples, core::u32 count) noexcept;

/**
 * @class VoiceActivity
 * @brief The gate, frame by frame.
 */
class VoiceActivity {
public:
    VoiceActivity() = default;

    /**
     * @brief Binds tuning.
     * @param params The four numbers.
     */
    explicit VoiceActivity(const VoiceActivityParams &params) noexcept : _params(params) {}

    /**
     * @brief Feeds one frame.
     * @param samples PCM16 mono.
     * @param count   How many; @ref kFrameSamples in normal operation.
     * @param playing Whether the node's own speaker is active.
     * @return What this frame changed.
     */
    VoiceEvent observe(const core::i16 *samples, core::u32 count, bool playing) noexcept;

    /**
     * @brief Level of the last frame observed.
     * @return The level, for a diagnostic or a threshold that adapts later.
     */
    [[nodiscard]] math::Fixed32 lastLevel() const noexcept { return _lastLevel; }

    /**
     * @brief Is an utterance open?
     * @return true between @ref VoiceEvent::UtteranceBegan and @c UtteranceEnded.
     */
    [[nodiscard]] bool speaking() const noexcept { return _speaking; }

    /**
     * @brief Milliseconds of silence accumulated inside the current utterance.
     * @return The hangover progress; zero outside an utterance.
     */
    [[nodiscard]] core::u32 silenceMilliseconds() const noexcept { return _silenceMilliseconds; }

    /**
     * @brief Utterances closed since construction.
     * @return The count.
     */
    [[nodiscard]] core::u32 utterances() const noexcept { return _utterances; }

    /**
     * @brief The tuning in force.
     * @return The parameters.
     */
    [[nodiscard]] const VoiceActivityParams &params() const noexcept { return _params; }

    /**
     * @brief Forgets the current utterance without changing the tuning.
     */
    void reset() noexcept;

private:
    VoiceActivityParams _params{};
    math::Fixed32 _lastLevel{};
    core::u32 _silenceMilliseconds{0u};
    core::u32 _utterances{0u};
    bool _speaking{false};
};

} // namespace lpl::satellite

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_SATELLITE_VOICEACTIVITY_HPP
