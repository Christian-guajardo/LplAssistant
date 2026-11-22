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

#    include <bit>

#    include <lpl/math/Cordic.hpp>
#    include <lpl/math/FixedMath.hpp>

namespace lpl::infer {

namespace {

constexpr core::i32 kLargestRaw = 0x7FFFFFFF; ///< The largest raw word of a Fixed32.
constexpr core::u32 kRadicandBits = 62u;      ///< Bits of a radicand whose root stays below 2^31.

/**
 * @brief A sum of squares in units of 2^-32, on two words: sixty-four values near the Fixed32 range pass 2^64.
 */
struct SquareSum {
    core::u64 high{0u}; ///< The bits past the low word.
    core::u64 low{0u};  ///< The low 64 bits.
};

/**
 * @brief The squares of @p input's raw words, summed exactly.
 */
SquareSum sumOfSquares(ConstVectorView input) noexcept
{
    SquareSum sum{};

    for (core::u32 i = 0u; i < input.count; ++i)
    {
        const core::i64 raw = static_cast<core::i64>(input.at(i).raw());
        const core::u64 square = static_cast<core::u64>(raw * raw);
        sum.low += square;
        if (sum.low < square)
            ++sum.high;
    }
    return sum;
}

/**
 * @brief @p sum divided by @p count, in two steps of 32 bits: each square is at most 2^62, so the high word stays
 *        below the count and the mean below 2^62.
 */
core::u64 meanOf(SquareSum sum, core::u32 count) noexcept
{
    const core::u64 upper = (sum.high << 32) | (sum.low >> 32);
    const core::u64 lower = ((upper % count) << 32) | (sum.low & 0xFFFFFFFFu);
    return ((upper / count) << 32) + lower / count;
}

/**
 * @brief The root of a mean of squares taken with extra fraction bits: @c root is the root mean square times
 *        2^(16 + shift).
 */
struct ScaledRoot {
    core::u64 root{0u};  ///< floor(sqrt(mean << 2 shift)), the mean in units of 2^-32.
    core::u32 shift{0u}; ///< Fraction bits past Q16.16.
};

/**
 * @brief The root of @p meanSquare, in units of 2^-32, with as many fraction bits as 62 bits of radicand leave: a
 *        root mean square of 0.0035 is 231 raw units in Q16.16, too coarse to divide by, and 2^30 units here.
 */
ScaledRoot rootOf(core::u64 meanSquare) noexcept
{
    const core::u32 width = static_cast<core::u32>(std::bit_width(meanSquare));
    const core::u32 shift = width >= kRadicandBits ? 0u : (kRadicandBits - width) / 2u;
    return ScaledRoot{math::integerSqrt(meanSquare << (2u * shift)), shift};
}

/**
 * @brief @p value divided by the root mean square @p root, in Q16.16, saturated: the quotient of a value by the root
 *        mean square of a vector it belongs to is at most the root of the vector's length.
 */
math::Fixed32 divideByRoot(math::Fixed32 value, ScaledRoot root) noexcept
{
    const core::i64 quotient =
        (static_cast<core::i64>(value.raw()) * (core::i64{1} << (math::Fixed32::kFracBits + root.shift))) /
        static_cast<core::i64>(root.root);
    if (quotient > kLargestRaw)
        return math::Fixed32::fromRaw(kLargestRaw);
    if (quotient < -kLargestRaw)
        return math::Fixed32::fromRaw(-kLargestRaw);
    return math::Fixed32::fromRaw(static_cast<core::i32>(quotient));
}

/** Two pi in Q16.16, 411774 raw units: a whole turn of a rotary angle, the period CORDIC's own reduction uses. */
constexpr core::u64 kTwoPiRaw = 2u * static_cast<core::u64>(math::Fixed32::pi().raw());

/**
 * @brief How far a pair that turns by @p turnsPerPosition per position has turned at @p position, in units of 2^-32
 *        turn: modulo a whole turn, which unsigned 32-bit arithmetic wraps exactly, at every position.
 */
constexpr core::u32 turnsAt(core::u32 position, core::u32 turnsPerPosition) noexcept
{
    return position * turnsPerPosition;
}

/**
 * @brief The angle of @p turns units of 2^-32 turn, in Q16.16 radians, in [0, 2 pi).
 */
math::Fixed32 angleOfTurns(core::u32 turns) noexcept
{
    return math::Fixed32::fromRaw(static_cast<core::i32>((static_cast<core::u64>(turns) * kTwoPiRaw) >> 32));
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
        accumulator +=
            (static_cast<core::i64>(lhs[i].raw()) * static_cast<core::i64>(rhs[i].raw())) >> math::Fixed32::kFracBits;

    if (accumulator > math::Fixed32::max().raw())
        return math::Fixed32::max();
    if (accumulator < math::Fixed32::min().raw())
        return math::Fixed32::min();
    return math::Fixed32::fromRaw(static_cast<core::i32>(accumulator));
}

} // namespace

void rmsNorm(ConstVectorView input, ConstVectorView gain, core::u32 epsilon, VectorView out) noexcept
{
    LPL_VERIFY(input.count == gain.count && out.count == input.count);
    if (input.count == 0u)
        return;

    const core::u64 meanSquare = meanOf(sumOfSquares(input), input.count) + epsilon;
    if (meanSquare == 0u)
    {
        for (core::u32 i = 0u; i < input.count; ++i)
            out.at(i) = math::Fixed32::zero();
        return;
    }

    const ScaledRoot root = rootOf(meanSquare);
    for (core::u32 i = 0u; i < input.count; ++i)
        out.at(i) = divideByRoot(input.at(i), root) * gain.at(i);
}

void applyRotaryEmbedding(VectorView vector, core::u32 heads, core::u32 headDim, RotaryFrequencies frequencies,
                          core::u32 position) noexcept
{
    LPL_VERIFY(vector.count == heads * headDim && frequencies.pairs == headDim / 2u);

    for (core::u32 pair = 0u; pair < frequencies.pairs; ++pair)
    {
        math::Fixed32 sine{};
        math::Fixed32 cosine{};
        math::Cordic::sincos(angleOfTurns(turnsAt(position, frequencies.turnsPerPosition[pair])), sine, cosine);

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

void attentionBlock(const ModelConfig &shape, const LayerWeights &weights, RotaryFrequencies rotary, KvCache &cache,
                    core::u32 layer, core::u32 slot, core::u32 position, VectorView residual, VectorView normed,
                    VectorView query, VectorView scores, VectorView blend, QuantBlock *blocks) noexcept
{
    const core::u32 headDim = shape.headDim();
    const VectorView key = cache.key(layer, slot);
    const VectorView value = cache.value(layer, slot);

    rmsNorm(residual, weights.attentionNorm, shape.normEpsilon, normed);
    quantiseRow(normed.values, normed.count, blocks);
    quantisedMatVecPrepared(weights.query, blocks, query);
    quantisedMatVecPrepared(weights.key, blocks, key);
    quantisedMatVecPrepared(weights.value, blocks, value);

    applyRotaryEmbedding(query, shape.heads, headDim, rotary, position);
    applyRotaryEmbedding(key, shape.heads, headDim, rotary, position);

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
