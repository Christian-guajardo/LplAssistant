/**
 * @file PowerState.cpp
 * @brief What a node does when nobody is talking to it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/satellite/PowerState.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/satellite/Protocol.hpp>

namespace lpl::satellite {

PowerPolicy policyFor(NodeState state) noexcept
{
    PowerPolicy policy{};
    policy.wakeIntervalMicroseconds = kFrameMilliseconds * 1000u;

    switch (state)
    {
        case NodeState::Idle:
            // Nothing is happening and nothing is expected. Everything off but the
            // codec, which must keep filling a buffer or there is nothing to wake for.
            policy.mayStopPeriodicTick = true;
            policy.maySleepUntilAudio = true;
            policy.mayScaleDownClock = true;
            policy.networkMustStayUp = false;
            break;

        case NodeState::Listening:
            // Somebody is speaking. Still nothing on the wire — that is the wake
            // word's whole job — so the link may stay down, but the clock may not
            // drop: the correlator runs on every frame now and must finish inside one.
            policy.mayStopPeriodicTick = true;
            policy.maySleepUntilAudio = true;
            policy.mayScaleDownClock = false;
            policy.networkMustStayUp = false;
            break;

        case NodeState::Streaming:
            // A datagram every forty milliseconds. Sleeping until the next buffer is
            // still correct — the cadence IS the buffer — but the link is up and the
            // clock stays where it is.
            policy.mayStopPeriodicTick = true;
            policy.maySleepUntilAudio = true;
            policy.mayScaleDownClock = false;
            policy.networkMustStayUp = true;
            break;

        case NodeState::Speaking:
            // Playing and capturing at once. The microphone stays open so the
            // sovereign can cut in, which is the one thing a voice assistant must
            // never make impossible.
            policy.mayStopPeriodicTick = true;
            policy.maySleepUntilAudio = true;
            policy.mayScaleDownClock = false;
            policy.networkMustStayUp = true;
            break;
    }
    return policy;
}

void PowerState::enter(NodeState next) noexcept
{
    if (next == _state)
        return;
    _state = next;
    ++_transitions;
}

void PowerState::onVoice(VoiceEvent event) noexcept
{
    switch (event)
    {
        case VoiceEvent::UtteranceBegan:
            // Speaking outranks listening: a reply is playing and somebody just
            // started talking over it, which is a barge-in and not a new state.
            if (!_playing)
                enter(NodeState::Listening);
            break;
        case VoiceEvent::UtteranceEnded:
            enter(_playing ? NodeState::Speaking : NodeState::Idle);
            break;
        case VoiceEvent::Silence:
            if (!_playing && _state == NodeState::Listening)
                enter(NodeState::Idle);
            break;
        case VoiceEvent::Speaking:
            break;
    }
}

void PowerState::onWakeWord() noexcept { enter(NodeState::Streaming); }

void PowerState::onPlaybackBegan() noexcept
{
    _playing = true;
    enter(NodeState::Speaking);
}

void PowerState::onPlaybackEnded() noexcept
{
    _playing = false;
    enter(NodeState::Idle);
}

void PowerState::advance(core::u32 microseconds, bool awake) noexcept
{
    _inState[static_cast<core::u32>(_state)] += microseconds;
    _elapsed += microseconds;
    if (awake)
        _awake += microseconds;
}

core::u64 PowerState::microsecondsIn(NodeState state) const noexcept
{
    return _inState[static_cast<core::u32>(state)];
}

core::u32 PowerState::dutyCyclePermille() const noexcept
{
    // Nothing measured yet reads as fully awake rather than as fully asleep. An
    // uninitialised counter that claimed perfect efficiency is the one answer a power
    // measurement must never give by default.
    if (_elapsed == 0u)
        return 1000u;
    return static_cast<core::u32>((_awake * 1000u) / _elapsed);
}

} // namespace lpl::satellite

#endif // LPL_HAS_FOUNDATION
