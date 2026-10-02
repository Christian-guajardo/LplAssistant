/**
 * @file Duplex.hpp
 * @brief Listening while speaking, and not answering oneself.
 *
 * A satellite that stops listening while it talks cannot be interrupted, which is
 * the one thing a voice assistant most needs to allow. So it listens through its own
 * playback — and then has to recognise its own voice coming back. Not by DSP echo
 * cancellation but by CONTENT: an utterance arriving inside the playback window whose
 * words are mostly the words just spoken is an echo. Cheap, and it needs no second
 * microphone.
 *
 * "Content" on a node that has no speech recogniser means the only content it has:
 * the spectral shape of what it just played. So this reuses @ref extractFeatures —
 * the same four bands the wake word matches on. That is not a shortcut, it is the
 * point: one feature extractor with three consumers cannot disagree with itself, and
 * a node whose whole job is to spend nothing cannot afford a second front end.
 *
 * The comparison is against a RING of recently played frames rather than the
 * aligned one, because the echo comes back late by however long the room is —
 * a delay a node cannot know and does not need to, if it searches a window.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_DUPLEX_HPP
#    define LPL_SATELLITE_DUPLEX_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/satellite/WakeWord.hpp>

namespace lpl::satellite {

/**
 * @brief Played frames kept for comparison.
 *
 * Sixteen frames is 640 milliseconds at the protocol's rate — comfortably longer
 * than the reverberation time of a living room, and short enough that the node is
 * not still suspecting itself half a second after it stopped talking.
 */
inline constexpr core::u32 kEchoWindowFrames = 16u;

/**
 * @class Duplex
 * @brief Tells the node's own voice from somebody else's.
 */
class Duplex {
public:
    Duplex() = default;

    /**
     * @brief Sets how close a shape must be to count as an echo.
     * @param tolerance Manhattan distance below which a capture is the node itself.
     */
    explicit Duplex(core::u32 tolerance) noexcept : _tolerance(tolerance) {}

    /**
     * @brief Records a frame the node is about to play.
     * @param samples PCM16 mono.
     * @param count   How many.
     */
    void notePlayed(const core::i16 *samples, core::u32 count) noexcept;

    /**
     * @brief Is this captured frame the node hearing itself?
     *
     * Returns false when nothing has been played, and that is deliberate rather than
     * defensive: with an empty window every capture would be infinitely far from
     * everything, which is the right answer, but saying so explicitly keeps a silent
     * node from depending on the distance of an uninitialised shape.
     *
     * @param samples PCM16 mono.
     * @param count   How many.
     * @return true when it matches something recently played.
     */
    [[nodiscard]] bool isEcho(const core::i16 *samples, core::u32 count) noexcept;

    /**
     * @brief Closest distance found at the last @ref isEcho.
     * @return The distance.
     */
    [[nodiscard]] core::u32 lastDistance() const noexcept { return _lastDistance; }

    /**
     * @brief Captures rejected as the node's own voice.
     * @return The count.
     */
    [[nodiscard]] core::u32 rejected() const noexcept { return _rejected; }

    /**
     * @brief Forgets what was played.
     *
     * Called when playback stops: the room goes quiet, and a node that kept
     * suspecting the last thing it said would deafen itself to a reply.
     */
    void clear() noexcept;

private:
    FeatureFrame _played[kEchoWindowFrames]{};
    core::u32 _filled{0u};
    core::u32 _cursor{0u};
    core::u32 _tolerance{96u};
    core::u32 _lastDistance{kBandScale * 2u};
    core::u32 _rejected{0u};
};

} // namespace lpl::satellite

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_SATELLITE_DUPLEX_HPP
