/**
 * @file VoiceActivity.cpp
 * @brief Deciding that someone is speaking, on energy alone.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/satellite/VoiceActivity.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/math/FixedMath.hpp>
#    include <lpl/satellite/Protocol.hpp>

namespace lpl::satellite {

math::Fixed32 frameLevel(const core::i16 *samples, core::u32 count) noexcept
{
    if (samples == nullptr || count == 0u)
        return math::Fixed32::zero();

    core::u64 sumOfSquares = 0u;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const core::i64 value = static_cast<core::i64>(samples[i]);
        sumOfSquares += static_cast<core::u64>(value * value);
    }

    // mean(v^2) with v = s / 32768, expressed in Q16.16 before the root. The shift
    // and the divide are grouped so the numerator keeps its bits: 32768 squared is
    // 2^30, and dividing by it first would leave zero for anything below a quarter
    // of full scale — which is every voice ever recorded.
    const core::u64 meanRaw = (sumOfSquares << 16) / (static_cast<core::u64>(count) << 30);
    return math::fixedSqrt(math::Fixed32::fromRaw(static_cast<core::i32>(meanRaw)));
}

VoiceEvent VoiceActivity::observe(const core::i16 *samples, core::u32 count, bool playing) noexcept
{
    _lastLevel = frameLevel(samples, count);

    const math::Fixed32 boost = playing ? _params.playbackBoost : math::Fixed32::one();
    const math::Fixed32 startLevel = _params.startLevel * boost;
    const math::Fixed32 keepLevel = _params.keepLevel * boost;

    // Milliseconds are derived from the frame, not assumed: a node whose driver hands
    // over a different buffer size still counts its hangover in real time.
    const core::u32 frameMilliseconds = count == 0u ? kFrameMilliseconds : (count * 1000u) / kSampleRateHz;

    if (!_speaking)
    {
        if (_lastLevel < startLevel)
            return VoiceEvent::Silence;
        _speaking = true;
        _silenceMilliseconds = 0u;
        return VoiceEvent::UtteranceBegan;
    }

    _silenceMilliseconds = _lastLevel < keepLevel ? _silenceMilliseconds + frameMilliseconds : 0u;
    if (_silenceMilliseconds < _params.hangoverMilliseconds)
        return VoiceEvent::Speaking;

    _speaking = false;
    _silenceMilliseconds = 0u;
    ++_utterances;
    return VoiceEvent::UtteranceEnded;
}

void VoiceActivity::reset() noexcept
{
    _lastLevel = math::Fixed32::zero();
    _silenceMilliseconds = 0u;
    _speaking = false;
}

} // namespace lpl::satellite

#endif // LPL_HAS_FOUNDATION
