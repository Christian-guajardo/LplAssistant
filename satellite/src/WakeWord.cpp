/**
 * @file WakeWord.cpp
 * @brief The gate that keeps the network quiet.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/satellite/WakeWord.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::satellite {

namespace {

/// Haar levels applied, giving three detail bands and one residual.
constexpr core::u32 kHaarLevels = 3u;

/// Longest frame the cascade will work on.
constexpr core::u32 kMaxFrameSamples = 1024u;

/**
 * @brief The cascade's working lanes.
 *
 * File scope and not a local array, because four kibibytes on the stack is four
 * kibibytes the kernel stack does not have — the same overflow a 1024-entity scene
 * caused the first time it was declared inside a function. Safe as shared state
 * because a node extracts features one frame at a time, on one thread, by
 * construction: the whole point of this module is that it does not spend cycles.
 */
core::i32 gLane[kMaxFrameSamples];

} // namespace

void extractFeatures(const core::i16 *samples, core::u32 count, FeatureFrame &out) noexcept
{
    for (core::u32 b = 0u; b < kBandCount; ++b)
        out.bands[b] = 0u;

    if (samples == nullptr || count < (core::u32{1} << kHaarLevels))
        return;

    // The cascade works on a copy in 32-bit lanes: sums grow by one bit per level,
    // and doing it in place over the caller's i16 buffer would both clip and destroy
    // audio somebody else still has to send.
    core::i32 *const lane = gLane;
    core::u32 length = count < kMaxFrameSamples ? count : kMaxFrameSamples;
    for (core::u32 i = 0u; i < length; ++i)
        lane[i] = static_cast<core::i32>(samples[i]);

    core::u64 energy[kBandCount] = {};

    for (core::u32 level = 0u; level < kHaarLevels; ++level)
    {
        const core::u32 half = length / 2u;
        if (half == 0u)
            break;

        core::u64 detail = 0u;
        for (core::u32 i = 0u; i < half; ++i)
        {
            const core::i32 a = lane[2u * i];
            const core::i32 b = lane[2u * i + 1u];
            const core::i32 difference = a - b;
            detail += static_cast<core::u64>(static_cast<core::i64>(difference) * difference);
            // The sum is halved rather than kept, so the residual stays on the same
            // scale as the input and the four energies remain comparable.
            lane[i] = (a + b) / 2;
        }
        // Band 0 is the highest octave: the first level splits the top half of the
        // spectrum off, so the levels come out in descending frequency order.
        energy[level] = detail / half;
        length = half;
    }

    core::u64 residual = 0u;
    for (core::u32 i = 0u; i < length; ++i)
        residual += static_cast<core::u64>(static_cast<core::i64>(lane[i]) * lane[i]);
    energy[kBandCount - 1u] = length == 0u ? 0u : residual / length;

    core::u64 total = 0u;
    for (core::u32 b = 0u; b < kBandCount; ++b)
        total += energy[b];

    if (total == 0u)
    {
        // Silence has no shape. An even split is the answer furthest from any real
        // template, so a quiet room never trips the gate.
        for (core::u32 b = 0u; b < kBandCount; ++b)
            out.bands[b] = static_cast<core::u16>(kBandScale / kBandCount);
        return;
    }

    // Largest-remainder apportionment, so the bands sum to kBandScale exactly. Plain
    // truncation loses up to three units, and a shape whose total drifts with the
    // input would make a distance threshold mean something different frame to frame.
    core::u32 assigned = 0u;
    core::u64 remainder[kBandCount] = {};
    for (core::u32 b = 0u; b < kBandCount; ++b)
    {
        const core::u64 scaled = energy[b] * kBandScale;
        out.bands[b] = static_cast<core::u16>(scaled / total);
        remainder[b] = scaled % total;
        assigned += out.bands[b];
    }
    while (assigned < kBandScale)
    {
        core::u32 best = 0u;
        for (core::u32 b = 1u; b < kBandCount; ++b)
            if (remainder[b] > remainder[best])
                best = b;
        ++out.bands[best];
        remainder[best] = 0u;
        ++assigned;
    }
}

core::u32 shapeDistance(const FeatureFrame &lhs, const FeatureFrame &rhs) noexcept
{
    core::u32 distance = 0u;
    for (core::u32 b = 0u; b < kBandCount; ++b)
    {
        const core::u32 a = lhs.bands[b];
        const core::u32 c = rhs.bands[b];
        distance += a > c ? a - c : c - a;
    }
    return distance;
}

bool WakeWord::arm(const FeatureFrame *frames, core::u32 count, core::u32 tolerance) noexcept
{
    if (frames == nullptr || count == 0u || count > kMaxTemplateFrames)
        return false;

    for (core::u32 i = 0u; i < count; ++i)
        _template[i] = frames[i];
    _templateFrames = count;
    _tolerance = tolerance;
    reset();
    return true;
}

bool WakeWord::observe(const core::i16 *samples, core::u32 count) noexcept
{
    if (_templateFrames == 0u)
        return false;

    FeatureFrame shape{};
    extractFeatures(samples, count, shape);

    _window[_cursor] = shape;
    _cursor = (_cursor + 1u) % _templateFrames;
    if (_filled < _templateFrames)
        ++_filled;
    if (_filled < _templateFrames)
        return false;

    // The window is a ring, so the oldest frame sits at the cursor. Walking from
    // there aligns it with template frame zero without moving any data.
    core::u32 total = 0u;
    for (core::u32 i = 0u; i < _templateFrames; ++i)
        total += shapeDistance(_window[(_cursor + i) % _templateFrames], _template[i]);

    _lastDistance = total / _templateFrames;
    if (_lastDistance > _tolerance)
        return false;

    ++_detections;
    _matchDistance = _lastDistance;
    // Cleared, so one spoken word is one wake. Without this the window keeps
    // matching for as many frames as the word is long, and the node would open a
    // stream per frame.
    //
    // Through reset() and not by zeroing the two indices by hand, which is what this
    // did first: the distance then kept the value it had at the match — zero — while
    // the window refilled, so anything reading lastDistance() over the next few
    // frames was reading the match again rather than measuring the frames after it.
    reset();
    return true;
}

void WakeWord::reset() noexcept
{
    _filled = 0u;
    _cursor = 0u;
    _lastDistance = kBandScale * 2u;
}

} // namespace lpl::satellite

#endif // LPL_HAS_FOUNDATION
