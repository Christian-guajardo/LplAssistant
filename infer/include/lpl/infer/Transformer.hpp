/**
 * @file Transformer.hpp
 * @brief Assembling blocks into a forward pass.
 *
 * Layer order and residual placement are part of the fold: two arrangements that are
 * mathematically equal are not bitwise equal.
 *
 * Pre-normalisation — each sub-layer normalises its input and adds its output to an
 * untouched residual — rather than the post-normalisation of the original paper. The
 * residual path then carries no normalisation at all, which is what keeps a stack of
 * blocks from scaling the stream by a compounding factor. At Q16.16 that compounding
 * is not a training inconvenience: it saturates.
 *
 * Every buffer the pass needs is claimed once by @ref Transformer::allocate. Nothing
 * between the first token and the last touches an allocator.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_TRANSFORMER_HPP
#    define LPL_LPL_INFER_TRANSFORMER_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/KvCache.hpp>
#        include <lpl/infer/Model.hpp>

namespace lpl::infer {

/**
 * @class Transformer
 * @brief The forward pass and the scratch it runs in.
 */
class Transformer {
public:
    Transformer() = default;

    /**
     * @brief Claims every working buffer up front.
     * @param arena Storage.
     * @param model The weights this pass will run.
     * @return false when the model is unbuilt or the arena is exhausted.
     */
    bool allocate(TensorArena &arena, const Model &model);

    /**
     * @brief Runs one token through every block.
     *
     * @param token    Identifier to embed.
     * @param position Absolute position, for the rotary embedding.
     * @param cache    The window; a slot is reserved here.
     * @param outLogits Receives @c vocabSize scores.
     * @return false when the cache refused the position.
     */
    [[nodiscard]] bool forward(core::u32 token, core::u32 position, KvCache &cache, VectorView outLogits) noexcept;

    /**
     * @brief The residual stream after the last @ref forward.
     *
     * Exposed for the gate rather than for callers: folding the stream as well as the
     * logits is what distinguishes "the two targets picked the same token" from "the
     * two targets computed the same thing", and the first can hold while the second
     * fails.
     *
     * @return The stream.
     */
    [[nodiscard]] ConstVectorView residual() const noexcept { return _residual; }

private:
    const Model *_model{nullptr};
    VectorView _residual{};
    VectorView _normed{};
    VectorView _query{};
    VectorView _blend{};
    VectorView _scores{};
    VectorView _gate{};
    VectorView _up{};
    QuantBlock *_blocks{nullptr};
};

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_TRANSFORMER_HPP
