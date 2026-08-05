/**
 * @file Duplex.cpp
 * @brief Listening while speaking, and not answering oneself.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/satellite/Duplex.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::satellite {

void Duplex::notePlayed(const core::i16 *samples, core::u32 count) noexcept
{
    FeatureFrame shape{};
    extractFeatures(samples, count, shape);

    _played[_cursor] = shape;
    _cursor = (_cursor + 1u) % kEchoWindowFrames;
    if (_filled < kEchoWindowFrames)
        ++_filled;
}

bool Duplex::isEcho(const core::i16 *samples, core::u32 count) noexcept
{
    _lastDistance = kBandScale * 2u;
    if (_filled == 0u)
        return false;

    FeatureFrame shape{};
    extractFeatures(samples, count, shape);

    // Nearest played frame, not the aligned one: the echo comes back delayed by
    // however far the wall is, which the node cannot know and does not have to.
    for (core::u32 i = 0u; i < _filled; ++i)
    {
        const core::u32 distance = shapeDistance(shape, _played[i]);
        if (distance < _lastDistance)
            _lastDistance = distance;
    }

    if (_lastDistance > _tolerance)
        return false;

    ++_rejected;
    return true;
}

void Duplex::clear() noexcept
{
    _filled = 0u;
    _cursor = 0u;
    _lastDistance = kBandScale * 2u;
}

} // namespace lpl::satellite

#endif // LPL_HAS_FOUNDATION
