/**
 * @file Quant.hpp
 * @brief Integer quantisation of weights and activations.
 *
 * The book already has a section on INT8 for inference; this is that section made
 * real. Integer arithmetic is reproducible across targets, which float would not
 * be at this scale.
 *
 * The shape is the one llama.cpp calls Q8_0 and it is chosen for a reason that is
 * not compression. A whole row shares one scale only if every value in it has a
 * comparable magnitude, which weights do not: one outlier column drags the scale up
 * and quantises everything else to zero. Blocking the row into groups of thirty-two
 * and giving each its own scale bounds how far any single outlier can reach, and it
 * is what makes eight bits enough to keep a forward pass recognisable.
 *
 * Nothing here is floating point, including the scale. A scale in @c float would put
 * the whole model one rounding mode away from a different answer.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_QUANT_HPP
#    define LPL_LPL_INFER_QUANT_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/math/FixedPoint.hpp>

namespace lpl::infer {

/**
 * @brief Values sharing one scale.
 *
 * Thirty-two, so a block's dot product cannot overflow a 32-bit accumulator:
 * 32 * 127 * 127 is 516128, three orders of magnitude below the limit. That
 * headroom is what lets the inner loop stay in @c i32 on a target with no wide
 * registers to spare.
 */
inline constexpr core::u32 kQuantBlockLanes = 32u;

/// Largest magnitude a lane may hold. 127 and not 128: the range stays symmetric,
/// so negating a quantised weight is exact rather than saturating on one side.
inline constexpr core::i32 kQuantLevels = 127;

/**
 * @struct QuantBlock
 * @brief Thirty-two eight-bit lanes and the magnitude they are measured against.
 *
 * @c scaleRaw is the block's largest absolute value in Q16.16, not a step size.
 * Storing the maximum rather than max/127 keeps both directions exact integer
 * arithmetic: quantising divides by it, dequantising multiplies by it, and neither
 * needs a reciprocal that would have to be rounded once and then reused.
 */
struct QuantBlock {
    core::i32 scaleRaw{0};
    core::i8 lanes[kQuantBlockLanes]{};
};

/**
 * @brief Blocks needed to hold @p values entries.
 *
 * @param values Entries in the row.
 * @return ceil(values / kQuantBlockLanes).
 */
[[nodiscard]] constexpr core::u32 quantBlockCount(core::u32 values) noexcept
{
    return (values + kQuantBlockLanes - 1u) / kQuantBlockLanes;
}

/**
 * @brief Quantises a row of Fixed32 into blocks.
 *
 * Rounds half away from zero, stated because it is part of the format: half toward
 * even and half away from zero disagree on exactly the values that sit between two
 * levels, and a model quantised under one rule and run under the other would drift
 * without anything reporting an error.
 *
 * A trailing partial block is zero-filled in both lanes and scale, so a dot product
 * over it contributes nothing rather than reading whatever the arena held before.
 *
 * @param values Row to quantise.
 * @param count  Entries in it.
 * @param out    Receives @ref quantBlockCount(count) blocks.
 */
void quantiseRow(const math::Fixed32 *values, core::u32 count, QuantBlock *out) noexcept;

/**
 * @brief Reconstructs one lane.
 *
 * @param block Block holding it.
 * @param lane  Lane index, below @ref kQuantBlockLanes.
 * @return The value the lane stands for.
 */
[[nodiscard]] math::Fixed32 dequantiseLane(const QuantBlock &block, core::u32 lane) noexcept;

/**
 * @brief Dot product of two quantised rows.
 *
 * Two levels of accumulation, and the split is the whole point. Inside a block the
 * lanes are plain integers and sum exactly in @c i32; across blocks the partial sums
 * carry different scales and are folded into a 64-bit Q16.16 accumulator. Neither
 * step rounds until the last one, so the result depends on the values and not on the
 * order a compiler chose to emit them in.
 *
 * @param lhs    First row.
 * @param rhs    Second row, quantised the same way.
 * @param blocks Blocks in each.
 * @return The dot product, saturating rather than wrapping.
 */
[[nodiscard]] math::Fixed32 dotQuantised(const QuantBlock *lhs, const QuantBlock *rhs, core::u32 blocks) noexcept;

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_QUANT_HPP
