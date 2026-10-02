/**
 * @file FeedForward.hpp
 * @brief The position-wise network.
 *
 * The other half of a block; separated so both can be folded independently.
 *
 * Gated, in the SwiGLU arrangement: two projections up, one of them passed through a
 * smooth gate, multiplied together, one projection back down. The gate is what makes
 * it worth three matrices instead of two — a block can suppress a channel outright
 * rather than only scale it, which is how a network of this size represents "this
 * feature does not apply here" at all.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_FEEDFORWARD_HPP
#    define LPL_LPL_INFER_FEEDFORWARD_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/Model.hpp>

namespace lpl::infer {

/**
 * @brief The sigmoid-weighted linear unit, @f$x \cdot \sigma(x)@f$.
 *
 * Smooth where a rectifier has a corner, and the corner is the point: an eight-bit
 * activation lands on a small number of distinct levels, so a hard threshold makes
 * neighbouring levels behave discontinuously differently. A smooth gate spreads that
 * decision across several levels, where quantisation can still represent it.
 *
 * @param value Input.
 * @return The gated value.
 */
[[nodiscard]] math::Fixed32 sigmoidLinearUnit(math::Fixed32 value) noexcept;

/**
 * @brief One block's position-wise network, added into the residual stream.
 *
 * @param shape    The model's extents.
 * @param weights  This block's tensors.
 * @param residual The stream, read and updated.
 * @param normed   Scratch, @c dim long.
 * @param gate     Scratch, @c ffnHidden long.
 * @param up       Scratch, @c ffnHidden long.
 * @param blocks   Scratch, at least @ref quantBlockCount(max(dim, ffnHidden)) blocks.
 */
void feedForwardBlock(const ModelConfig &shape, const LayerWeights &weights, VectorView residual, VectorView normed,
                      VectorView gate, VectorView up, QuantBlock *blocks) noexcept;

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_FEEDFORWARD_HPP
