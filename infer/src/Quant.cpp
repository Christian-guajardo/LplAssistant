/**
 * @file Quant.cpp
 * @brief Integer quantisation of weights and activations.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Quant.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::infer {

void quantiseRow(const math::Fixed32 *values, core::u32 count, QuantBlock *out) noexcept
{
    if (values == nullptr || out == nullptr)
        return;

    const core::u32 blocks = quantBlockCount(count);
    for (core::u32 b = 0u; b < blocks; ++b)
    {
        QuantBlock &block = out[b];
        const core::u32 base = b * kQuantBlockLanes;
        const core::u32 lanes = (base + kQuantBlockLanes <= count) ? kQuantBlockLanes : (count - base);

        core::i32 maxAbs = 0;
        for (core::u32 i = 0u; i < lanes; ++i)
        {
            const core::i32 magnitude = values[base + i].abs().raw();
            if (magnitude > maxAbs)
                maxAbs = magnitude;
        }

        block.scaleRaw = maxAbs;
        for (core::u32 i = 0u; i < kQuantBlockLanes; ++i)
            block.lanes[i] = 0;

        // An all-zero block has no scale to divide by, and every lane is already
        // zero — which is the correct answer, not a special case to apologise for.
        if (maxAbs == 0)
            continue;

        for (core::u32 i = 0u; i < lanes; ++i)
        {
            const core::i64 raw = static_cast<core::i64>(values[base + i].raw());
            const core::i64 half = static_cast<core::i64>(maxAbs) / 2;
            const core::i64 bias = raw >= 0 ? half : -half;
            core::i64 level = (raw * kQuantLevels + bias) / static_cast<core::i64>(maxAbs);
            if (level > kQuantLevels)
                level = kQuantLevels;
            if (level < -kQuantLevels)
                level = -kQuantLevels;
            block.lanes[i] = static_cast<core::i8>(level);
        }
    }
}

math::Fixed32 dequantiseLane(const QuantBlock &block, core::u32 lane) noexcept
{
    if (lane >= kQuantBlockLanes)
        return math::Fixed32::zero();
    const core::i64 product = static_cast<core::i64>(block.lanes[lane]) * static_cast<core::i64>(block.scaleRaw);
    return math::Fixed32::fromRaw(static_cast<core::i32>(product / kQuantLevels));
}

math::Fixed32 dotQuantised(const QuantBlock *lhs, const QuantBlock *rhs, core::u32 blocks) noexcept
{
    if (lhs == nullptr || rhs == nullptr)
        return math::Fixed32::zero();

    core::i64 accumulator = 0;
    for (core::u32 b = 0u; b < blocks; ++b)
    {
        core::i32 lanes = 0;
        for (core::u32 i = 0u; i < kQuantBlockLanes; ++i)
            lanes += static_cast<core::i32>(lhs[b].lanes[i]) * static_cast<core::i32>(rhs[b].lanes[i]);

        if (lanes == 0)
            continue;

        // The two scales multiply as Q16.16 values, then the level counts divide
        // back out. Done in this order the intermediate keeps its bits: dividing
        // first would throw away the whole block on a small scale.
        const core::i64 scales = (static_cast<core::i64>(lhs[b].scaleRaw) * static_cast<core::i64>(rhs[b].scaleRaw)) >>
                                 math::Fixed32::kFracBits;
        accumulator += (scales * static_cast<core::i64>(lanes)) / (kQuantLevels * kQuantLevels);
    }

    if (accumulator > math::Fixed32::max().raw())
        return math::Fixed32::max();
    if (accumulator < math::Fixed32::min().raw())
        return math::Fixed32::min();
    return math::Fixed32::fromRaw(static_cast<core::i32>(accumulator));
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
