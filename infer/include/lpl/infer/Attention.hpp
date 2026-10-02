/**
 * @file Attention.hpp
 * @brief Scaled dot-product attention over quantised tensors.
 *
 * Integer accumulation, saturating rather than wrapping.
 *
 * Two pieces of this file are choices rather than transcriptions, and both are
 * forced by the determinism contract:
 *
 *   - the softmax goes through @c math::fixedExp, because libm is unavailable to
 *     anything linked into the kernel and a float exponential would put the whole
 *     distribution one rounding mode away from a different answer;
 *   - the rotary frequency ladder is base two, not base ten thousand. The usual
 *     @f$\theta_j = 10000^{-2j/d}@f$ needs a transcendental power that this module
 *     is not allowed to compute. Halving per pair gives the same thing the ladder is
 *     for — a spread of wavelengths from one position to the whole window — with an
 *     exponent the hardware expresses exactly as a shift.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_ATTENTION_HPP
#    define LPL_LPL_INFER_ATTENTION_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/KvCache.hpp>
#        include <lpl/infer/Model.hpp>

namespace lpl::infer {

/**
 * @brief Root-mean-square normalisation with per-channel gains.
 *
 * The sum of squares accumulates in 64 bits at Q32 and is only narrowed once, at the
 * root. Narrowing each term first would lose every value below a raw unit — which,
 * on a residual stream whose entries are fractions, is most of them.
 *
 * @param input Values to normalise.
 * @param gain  Per-channel multipliers; same length as @p input.
 * @param out   Receives the result; may alias @p input.
 */
void rmsNorm(ConstVectorView input, ConstVectorView gain, VectorView out) noexcept;

/**
 * @brief Rotates each head's channel pairs by an angle proportional to @p position.
 *
 * Relative position falls out of the geometry: the dot product of two rotated
 * vectors depends on the DIFFERENCE of their angles, so a key cached at position
 * three and a query at position seven see a separation of four wherever the window
 * has since slid to.
 *
 * @param vector   Query or key run, @c heads * headDim long.
 * @param heads    Attention heads.
 * @param headDim  Channels per head; pairs are (2i, 2i+1) within a head.
 * @param position Absolute position of the token.
 */
void applyRotaryEmbedding(VectorView vector, core::u32 heads, core::u32 headDim, core::u32 position) noexcept;

/**
 * @brief Turns scores into a distribution, in place.
 *
 * Subtracts the maximum first. Not for numerical comfort — for correctness of the
 * bounded exponential: @c fixedExp saturates above about fifteen, so a raw score of
 * twenty and a raw score of forty would both come back as the same clamped value and
 * two clearly different tokens would end up equally likely.
 *
 * @param scores Values to normalise; length is the attended window.
 */
void softmaxInPlace(VectorView scores) noexcept;

/**
 * @brief One block's attention, added into the residual stream.
 *
 * @param shape    The model's extents.
 * @param weights  This block's tensors.
 * @param cache    Where this position's key and value are stored, and the past read.
 * @param layer    Block index.
 * @param slot     Cache slot for this token, from @ref KvCache::reserve.
 * @param position Absolute position, for the rotary embedding.
 * @param residual The stream, read and updated.
 * @param normed   Scratch, @c dim long.
 * @param query    Scratch, @c dim long.
 * @param scores   Scratch, at least @c cache.length() long.
 * @param blend    Scratch, @c dim long.
 * @param blocks   Scratch, @ref quantBlockCount(dim) blocks.
 */
void attentionBlock(const ModelConfig &shape, const LayerWeights &weights, KvCache &cache, core::u32 layer,
                    core::u32 slot, core::u32 position, VectorView residual, VectorView normed, VectorView query,
                    VectorView scores, VectorView blend, QuantBlock *blocks) noexcept;

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_ATTENTION_HPP
