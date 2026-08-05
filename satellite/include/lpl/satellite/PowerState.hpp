/**
 * @file PowerState.hpp
 * @brief What a node does when nobody is talking to it.
 *
 * Most of a satellite's life is silence. The wake-word gate is the only thing that
 * must stay awake, so everything else may be clock-gated, and the processor may sleep
 * between audio buffers rather than spinning on a periodic tick. This is the module's
 * declaration of what may be turned off and what may not.
 *
 * It is a DECLARATION and not a driver: this module does not know how a clock is
 * gated on the target it happens to be running on. It says which of four states the
 * node is in and what that state permits, and the host — the kernel's power floor, a
 * microcontroller's sleep modes, or nothing at all on a workstation — obeys as far as
 * its hardware allows. Splitting it this way is what lets one policy serve three
 * machines that share no register.
 *
 * The accounting is not decoration. "This node costs nothing when idle" is a claim,
 * and a claim about energy that is not measured is a wish: @ref PowerState::dutyCyclePermille
 * is the number that makes it checkable, and it is the number the satellite profile
 * prints.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_POWERSTATE_HPP
#    define LPL_SATELLITE_POWERSTATE_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/satellite/VoiceActivity.hpp>

namespace lpl::satellite {

/**
 * @enum NodeState
 * @brief The four things a node can be doing.
 */
enum class NodeState : core::u32 {
    /// Silence. Nothing has been heard; the deepest sleep the hardware offers.
    Idle = 0u,
    /// Somebody is speaking but the wake word has not been heard. Nothing leaves.
    Listening = 1u,
    /// The word was heard: datagrams are going out on a forty-millisecond cadence.
    Streaming = 2u,
    /// A reply is playing, and the microphone stays open so it can be interrupted.
    Speaking = 3u,
};

/**
 * @struct PowerPolicy
 * @brief What a state permits.
 */
struct PowerPolicy {
    /**
     * May the periodic tick be stopped entirely?
     *
     * True everywhere on this profile, and that is the whole reason a satellite is
     * allowed to be tickless when the engine profiles are not: it instantiates no
     * World, so there is no authoritative tick whose cadence is part of a contract.
     */
    bool mayStopPeriodicTick{true};

    /// May the processor sleep until a device writes the capture buffer?
    bool maySleepUntilAudio{true};

    /// May the clock be scaled down?
    bool mayScaleDownClock{true};

    /// Must the link stay up? False lets a host power-gate the network entirely.
    bool networkMustStayUp{false};

    /**
     * Longest single sleep, in microseconds.
     *
     * Bounded by the codec and not by a policy: a capture buffer fills every
     * @ref kFrameMilliseconds whether anybody is speaking or not, so sleeping past
     * that only means arriving late to a buffer that is already full.
     */
    core::u32 wakeIntervalMicroseconds{40000u};
};

/**
 * @brief What a state permits.
 * @param state The node's state.
 * @return Its policy.
 */
[[nodiscard]] PowerPolicy policyFor(NodeState state) noexcept;

/**
 * @class PowerState
 * @brief The state machine, and the accounting that makes its claim checkable.
 */
class PowerState {
public:
    PowerState() = default;

    /**
     * @brief Feeds a voice-activity decision.
     * @param event What the last frame changed.
     */
    void onVoice(VoiceEvent event) noexcept;

    /**
     * @brief The wake word was heard: the node starts streaming.
     */
    void onWakeWord() noexcept;

    /**
     * @brief A reply began playing.
     */
    void onPlaybackBegan() noexcept;

    /**
     * @brief The reply finished, or was cut off.
     */
    void onPlaybackEnded() noexcept;

    /**
     * @brief Accounts for elapsed time.
     *
     * Called by the host with what its clock measured, because this module has none
     * — and must not have one: a node that read a wall clock would stop being
     * replayable, and the gate folds this state machine.
     *
     * @param microseconds Time since the last call.
     * @param awake        Whether the processor was running rather than halted.
     */
    void advance(core::u32 microseconds, bool awake) noexcept;

    /**
     * @brief The node's state.
     * @return What it is doing.
     */
    [[nodiscard]] NodeState state() const noexcept { return _state; }

    /**
     * @brief What the current state permits.
     * @return The policy.
     */
    [[nodiscard]] PowerPolicy policy() const noexcept { return policyFor(_state); }

    /**
     * @brief State changes since construction.
     * @return The count.
     */
    [[nodiscard]] core::u32 transitions() const noexcept { return _transitions; }

    /**
     * @brief Microseconds spent in one state.
     * @param state Which one.
     * @return The total.
     */
    [[nodiscard]] core::u64 microsecondsIn(NodeState state) const noexcept;

    /**
     * @brief Fraction of elapsed time the processor was awake, in thousandths.
     *
     * The number that decides whether this profile earned its name. A node that idles
     * at a few per mille is one that can live on a battery; one that idles at five
     * hundred is a node with a spin loop in it somewhere.
     *
     * @return Awake time over elapsed time, 0 to 1000; 1000 when nothing elapsed yet.
     */
    [[nodiscard]] core::u32 dutyCyclePermille() const noexcept;

    /**
     * @brief Total time accounted for.
     * @return Microseconds.
     */
    [[nodiscard]] core::u64 elapsedMicroseconds() const noexcept { return _elapsed; }

    /**
     * @brief Time the processor was awake.
     * @return Microseconds.
     */
    [[nodiscard]] core::u64 awakeMicroseconds() const noexcept { return _awake; }

private:
    /**
     * @brief Moves to a state, counting the change.
     * @param next Where to go.
     */
    void enter(NodeState next) noexcept;

    NodeState _state{NodeState::Idle};
    core::u64 _inState[4]{};
    core::u64 _elapsed{0u};
    core::u64 _awake{0u};
    core::u32 _transitions{0u};
    bool _playing{false};
};

} // namespace lpl::satellite

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_SATELLITE_POWERSTATE_HPP
