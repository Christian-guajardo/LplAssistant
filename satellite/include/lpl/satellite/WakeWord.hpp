/**
 * @file WakeWord.hpp
 * @brief The gate that keeps the network quiet.
 *
 * A satellite that streamed continuously would saturate the link and the server for
 * nothing. Until the word is heard, nothing leaves the node. On a microcontroller
 * this is a few-kilobyte model; on a desktop it is the same interface with a larger
 * one behind it — and the same interface is what makes them swappable.
 *
 * The features are sub-band energies from a HAAR cascade, and the choice is forced
 * rather than preferred. The usual front end is a mel filterbank over an FFT, and an
 * FFT needs cosines — a transcendental, which nothing linked into the kernel is
 * allowed to compute. A Haar cascade splits a frame into octave bands with nothing
 * but adds, subtracts and shifts: exact on every target, and cheap enough to run on
 * a node whose whole job is to not spend energy.
 *
 * The bands are NORMALISED to a constant sum, so what is matched is the SHAPE of the
 * spectrum and not its loudness. A wake word said quietly from across the room and
 * the same word said into the microphone must land on the same template; a feature
 * that carried level would make the gate a volume knob.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_WAKEWORD_HPP
#    define LPL_SATELLITE_WAKEWORD_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

namespace lpl::satellite {

/// Octave bands one frame is split into: three high halves and the residual low.
inline constexpr core::u32 kBandCount = 4u;

/// What the four bands of a frame sum to after normalisation.
inline constexpr core::u32 kBandScale = 1024u;

/// Frames a template may span — about two seconds at forty milliseconds each.
inline constexpr core::u32 kMaxTemplateFrames = 48u;

/**
 * @struct FeatureFrame
 * @brief One frame's spectral shape.
 *
 * Four numbers summing to @ref kBandScale. Sixteen bits each rather than eight,
 * because a band that holds nine tenths of the energy needs more than 255 levels
 * before the quiet bands all collapse to the same value.
 */
struct FeatureFrame {
    core::u16 bands[kBandCount]{};
};

/**
 * @brief Splits a frame into normalised octave-band energies.
 *
 * Three Haar levels: each pass replaces neighbouring pairs with their sum and their
 * difference, keeps the difference as that octave's detail, and recurses on the sum.
 * Energies are accumulated in 64 bits before normalisation because a loud frame's
 * squared coefficients exceed 32 bits well before it clips.
 *
 * A silent frame has no shape to report and comes back as an even split, which is
 * the shape furthest from any real template — so silence never matches.
 *
 * @param samples PCM16 mono.
 * @param count   How many; at least eight, or the cascade has nothing to split.
 * @param out     Receives the four bands.
 */
void extractFeatures(const core::i16 *samples, core::u32 count, FeatureFrame &out) noexcept;

/**
 * @brief Manhattan distance between two shapes.
 *
 * L1 and not L2: the bands already sum to a constant, so a squared distance would
 * only weight the loudest band more, and the loudest band is the one a room's
 * reverberation moves most.
 *
 * @param lhs First shape.
 * @param rhs Second shape.
 * @return The distance, 0 for identical shapes and at most 2 * @ref kBandScale.
 */
[[nodiscard]] core::u32 shapeDistance(const FeatureFrame &lhs, const FeatureFrame &rhs) noexcept;

/**
 * @class WakeWord
 * @brief Matches a rolling window of frames against one template.
 */
class WakeWord {
public:
    WakeWord() = default;

    /**
     * @brief Loads a template.
     *
     * Copied into the object rather than referenced: a node arms this once at boot
     * from a table that may live in a boot module, and holding a pointer into
     * something the loader may reuse is how a wake word starts matching noise.
     *
     * @param frames    The template, in order.
     * @param count     Its length, up to @ref kMaxTemplateFrames.
     * @param tolerance Average per-frame distance still counted as a match.
     * @return false when the count is out of range.
     */
    bool arm(const FeatureFrame *frames, core::u32 count, core::u32 tolerance) noexcept;

    /**
     * @brief Feeds one frame.
     *
     * Fires at most once per crossing: after a match the window is cleared, so a
     * word held over several frames triggers one wake and not five.
     *
     * @param samples PCM16 mono.
     * @param count   How many.
     * @return true when the template just matched.
     */
    bool observe(const core::i16 *samples, core::u32 count) noexcept;

    /// The value @ref lastDistance reports when no full window has been measured.
    static constexpr core::u32 kNoMeasurement = kBandScale * 2u;

    /**
     * @brief Average per-frame distance at the last full window.
     *
     * @ref kNoMeasurement until the window fills, and again after a detection empties
     * it. Callers that average or minimise over this MUST skip that value — it is the
     * absence of a measurement, not a very large one.
     *
     * @return The distance, or @ref kNoMeasurement.
     */
    [[nodiscard]] core::u32 lastDistance() const noexcept { return _lastDistance; }

    /**
     * @brief Distance at the last successful match.
     *
     * Separate from @ref lastDistance because they answer different questions and a
     * single field cannot answer both: a match empties the window, so the moment
     * after a detection there is no current measurement at all — which is exactly
     * when a caller most wants to know how good the match was.
     *
     * @return The distance, or @ref kNoMeasurement before the first detection.
     */
    [[nodiscard]] core::u32 matchDistance() const noexcept { return _matchDistance; }

    /**
     * @brief Times the template has matched.
     * @return The count.
     */
    [[nodiscard]] core::u32 detections() const noexcept { return _detections; }

    /**
     * @brief Is a template loaded?
     * @return true after a successful @ref arm.
     */
    [[nodiscard]] bool armed() const noexcept { return _templateFrames != 0u; }

    /**
     * @brief Empties the rolling window without dropping the template.
     */
    void reset() noexcept;

private:
    FeatureFrame _template[kMaxTemplateFrames]{};
    FeatureFrame _window[kMaxTemplateFrames]{};
    core::u32 _templateFrames{0u};
    core::u32 _filled{0u};
    core::u32 _cursor{0u};
    core::u32 _tolerance{0u};
    core::u32 _lastDistance{kBandScale * 2u};
    core::u32 _matchDistance{kBandScale * 2u};
    core::u32 _detections{0u};
};

} // namespace lpl::satellite

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_SATELLITE_WAKEWORD_HPP
