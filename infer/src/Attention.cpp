/**
 * @file Attention.cpp
 * @brief Scaled dot-product attention over quantised tensors.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Attention.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/FixedMath.hpp>

namespace lpl::infer {

namespace {

/// Two pi in Q16.16, for reducing a rotary angle before it reaches the CORDIC.
const math::Fixed32 kTwoPi = math::Fixed32::pi() * math::Fixed32::fromInt(2);

/**
 * @brief Brings an angle into [0, 2pi).
 *
 * Done here rather than left to the CORDIC because the rotary angle grows with the
 * position and a window of a few dozen tokens already exceeds any range a rotation
 * kernel reduces for free. Subtraction rather than a division: the loop runs a few
 * dozen times at worst and every step is exact, where a division would round once
 * and make two positions that differ by a full turn disagree.
 *
 * @param angle Angle in radians.
 * @return The equivalent angle in [0, 2pi).
 */
math::Fixed32 wrapAngle(math::Fixed32 angle) noexcept
{
    while (angle >= kTwoPi)
        angle -= kTwoPi;
    while (angle < math::Fixed32::zero())
        angle += kTwoPi;
    return angle;
}

/**
 * @brief Dot product of two runs, accumulated at full width.
 * @param lhs   First run.
 * @param rhs   Second run.
 * @param count Entries.
 * @return The dot product, saturating.
 */
math::Fixed32 dotFixed(const math::Fixed32 *lhs, const math::Fixed32 *rhs, core::u32 count) noexcept
{
    core::i64 accumulator = 0;
    for (core::u32 i = 0u; i < count; ++i)
        accumulator += (static_cast<core::i64>(lhs[i].raw()) * static_cast<core::i64>(rhs[i].raw())) >>
                       math::Fixed32::kFracBits;

    if (accumulator > math::Fixed32::max().raw())
        return math::Fixed32::max();
    if (accumulator < math::Fixed32::min().raw())
        return math::Fixed32::min();
    return math::Fixed32::fromRaw(static_cast<core::i32>(accumulator));
}

} // namespace

void rmsNorm(ConstVectorView input, ConstVectorView gain, VectorView out) noexcept
{
    LPL_VERIFY(input.count == gain.count && out.count == input.count);
    if (input.count == 0u)
        return;

    core::u64 sumOfSquares = 0u;
    for (core::u32 i = 0u; i < input.count; ++i)
    {
        const core::i64 raw = static_cast<core::i64>(input.at(i).raw());
        sumOfSquares += static_cast<core::u64>(raw * raw);
    }

    // Mean is Q32; shifting to Q16.16 before the root keeps the intermediate inside
    // what fixedSqrt takes. A vector of exact zeros has no direction to preserve, so
    // the norm is one rather than a division by zero.
    const core::u64 mean = sumOfSquares / input.count;
    const math::Fixed32 meanFixed = math::Fixed32::fromRaw(static_cast<core::i32>(mean >> math::Fixed32::kFracBits));
    math::Fixed32 root = math::fixedSqrt(meanFixed);
    if (root.raw() == 0)
        root = math::Fixed32::one();

    for (core::u32 i = 0u; i < input.count; ++i)
        out.at(i) = (input.at(i) / root) * gain.at(i);
}

void applyRotaryEmbedding(VectorView vector, core::u32 heads, core::u32 headDim, core::u32 position) noexcept
{
    LPL_VERIFY(vector.count == heads * headDim);
    const core::u32 pairs = headDim / 2u;

    for (core::u32 pair = 0u; pair < pairs; ++pair)
    {
        // theta = 2^-pair. Exact, and the reason this ladder is base two.
        const math::Fixed32 frequency = math::Fixed32::fromRaw(math::Fixed32::kOne >> pair);
        const math::Fixed32 angle = wrapAngle(math::Fixed32::fromInt(static_cast<core::i32>(position)) * frequency);

        math::Fixed32 sine{};
        math::Fixed32 cosine{};
        math::Cordic::sincos(angle, sine, cosine);

        for (core::u32 head = 0u; head < heads; ++head)
        {
            const core::u32 base = head * headDim + pair * 2u;
            const math::Fixed32 even = vector.at(base);
            const math::Fixed32 odd = vector.at(base + 1u);
            vector.at(base) = even * cosine - odd * sine;
            vector.at(base + 1u) = even * sine + odd * cosine;
        }
    }
}

void softmaxInPlace(VectorView scores) noexcept
{
    if (scores.count == 0u)
        return;

    math::Fixed32 highest = scores.at(0);
    for (core::u32 i = 1u; i < scores.count; ++i)
        if (scores.at(i) > highest)
            highest = scores.at(i);

    core::i64 total = 0;
    for (core::u32 i = 0u; i < scores.count; ++i)
    {
        const math::Fixed32 weight = math::fixedExp(scores.at(i) - highest);
        scores.at(i) = weight;
        total += weight.raw();
    }

    // Every weight underflowed to zero, which Q16.16 reaches once the gap past the
    // maximum exceeds about eleven. Uniform is the honest answer: the format cannot
    // tell these positions apart any more, and pretending one of them won would be
    // reporting a distinction that was lost.
    if (total == 0)
    {
        const math::Fixed32 share = math::Fixed32::one() / math::Fixed32::fromInt(static_cast<core::i32>(scores.count));
        for (core::u32 i = 0u; i < scores.count; ++i)
            scores.at(i) = share;
        return;
    }

    for (core::u32 i = 0u; i < scores.count; ++i)
        scores.at(i) = math::Fixed32::fromRaw(
            static_cast<core::i32>((static_cast<core::i64>(scores.at(i).raw()) << math::Fixed32::kFracBits) / total));
}

void attentionBlock(const ModelConfig &shape, const LayerWeights &weights, KvCache &cache, core::u32 layer,
                    core::u32 slot, core::u32 position, VectorView residual, VectorView normed, VectorView query,
                    VectorView scores, VectorView blend, QuantBlock *blocks) noexcept
{
    const core::u32 headDim = shape.headDim();
    const VectorView key = cache.key(layer, slot);
    const VectorView value = cache.value(layer, slot);

    rmsNorm(residual, weights.attentionNorm, normed);
    quantiseRow(normed.values, normed.count, blocks);
    quantisedMatVecPrepared(weights.query, blocks, query);
    quantisedMatVecPrepared(weights.key, blocks, key);
    quantisedMatVecPrepared(weights.value, blocks, value);

    applyRotaryEmbedding(query, shape.heads, headDim, position);
    applyRotaryEmbedding(key, shape.heads, headDim, position);

    // 1 / sqrt(headDim). Computed rather than tabulated: a table would be a second
    // statement of the head width, and the two would drift the day one changed.
    const math::Fixed32 scale =
        math::Fixed32::one() / math::fixedSqrt(math::Fixed32::fromInt(static_cast<core::i32>(headDim)));

    const core::u32 attended = cache.length();
    for (core::u32 head = 0u; head < shape.heads; ++head)
    {
        const core::u32 base = head * headDim;
        const VectorView window = scores.slice(0u, attended);

        for (core::u32 t = 0u; t < attended; ++t)
        {
            const VectorView past = cache.key(layer, t);
            window.at(t) = dotFixed(query.values + base, past.values + base, headDim) * scale;
        }
        softmaxInPlace(window);

        for (core::u32 i = 0u; i < headDim; ++i)
            blend.at(base + i) = math::Fixed32::zero();
        for (core::u32 t = 0u; t < attended; ++t)
        {
            const VectorView past = cache.value(layer, t);
            const math::Fixed32 weight = window.at(t);
            for (core::u32 i = 0u; i < headDim; ++i)
                blend.at(base + i) += weight * past.at(base + i);
        }
    }

    quantiseRow(blend.values, blend.count, blocks);
    quantisedMatVecPrepared(weights.output, blocks, normed);
    for (core::u32 i = 0u; i < residual.count; ++i)
        residual.at(i) = residual.at(i).saturatingAdd(normed.at(i));
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
