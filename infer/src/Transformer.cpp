/**
 * @file Transformer.cpp
 * @brief Assembling blocks into a forward pass.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Transformer.hpp>

#include <lpl/infer/Attention.hpp>
#include <lpl/infer/FeedForward.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::infer {

bool Transformer::allocate(TensorArena &arena, const Model &model)
{
    if (!model.ready())
        return false;

    const ModelConfig &shape = model.config();
    const core::u32 widest = shape.dim > shape.ffnHidden ? shape.dim : shape.ffnHidden;

    math::Fixed32 *const residual = arena.claim<math::Fixed32>(shape.dim);
    math::Fixed32 *const normed = arena.claim<math::Fixed32>(shape.dim);
    math::Fixed32 *const query = arena.claim<math::Fixed32>(shape.dim);
    math::Fixed32 *const blend = arena.claim<math::Fixed32>(shape.dim);
    math::Fixed32 *const scores = arena.claim<math::Fixed32>(shape.contextLength);
    math::Fixed32 *const gate = arena.claim<math::Fixed32>(shape.ffnHidden);
    math::Fixed32 *const up = arena.claim<math::Fixed32>(shape.ffnHidden);
    QuantBlock *const blocks = arena.claim<QuantBlock>(quantBlockCount(widest));
    if (residual == nullptr || normed == nullptr || query == nullptr || blend == nullptr || scores == nullptr ||
        gate == nullptr || up == nullptr || blocks == nullptr)
        return false;

    _model = &model;
    _residual = VectorView{residual, shape.dim};
    _normed = VectorView{normed, shape.dim};
    _query = VectorView{query, shape.dim};
    _blend = VectorView{blend, shape.dim};
    _scores = VectorView{scores, shape.contextLength};
    _gate = VectorView{gate, shape.ffnHidden};
    _up = VectorView{up, shape.ffnHidden};
    _blocks = blocks;
    return true;
}

bool Transformer::forward(core::u32 token, core::u32 position, KvCache &cache, VectorView outLogits) noexcept
{
    LPL_VERIFY(_model != nullptr);
    const ModelConfig &shape = _model->config();
    LPL_VERIFY(outLogits.count == shape.vocabSize);

    const core::u32 slot = cache.reserve();
    if (slot == kNoSlot)
        return false;

    _model->embed(token, _residual);

    for (core::u32 l = 0u; l < shape.layers; ++l)
    {
        attentionBlock(shape, _model->layer(l), cache, l, slot, position, _residual, _normed, _query, _scores, _blend,
                       _blocks);
        feedForwardBlock(shape, _model->layer(l), _residual, _normed, _gate, _up, _blocks);
    }

    // The output head is the embedding table read the other way round. One matrix,
    // two directions — so a token's representation and its score cannot disagree.
    rmsNorm(_residual, _model->finalNorm(), _normed);
    quantiseRow(_normed.values, _normed.count, _blocks);
    quantisedMatVecPrepared(_model->embedding(), _blocks, outLogits);
    return true;
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
