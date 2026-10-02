/**
 * @file FeedForward.cpp
 * @brief The position-wise network.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/FeedForward.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/infer/Attention.hpp>
#    include <lpl/math/FixedMath.hpp>

namespace lpl::infer {

math::Fixed32 sigmoidLinearUnit(math::Fixed32 value) noexcept
{
    const math::Fixed32 denominator = math::Fixed32::one() + math::fixedExp(-value);
    if (denominator.raw() == 0)
        return math::Fixed32::zero();
    return value * (math::Fixed32::one() / denominator);
}

void feedForwardBlock(const ModelConfig &shape, const LayerWeights &weights, VectorView residual, VectorView normed,
                      VectorView gate, VectorView up, QuantBlock *blocks) noexcept
{
    rmsNorm(residual, weights.feedForwardNorm, normed);
    quantiseRow(normed.values, normed.count, blocks);
    quantisedMatVecPrepared(weights.gate, blocks, gate);
    quantisedMatVecPrepared(weights.up, blocks, up);

    for (core::u32 i = 0u; i < shape.ffnHidden; ++i)
        gate.at(i) = sigmoidLinearUnit(gate.at(i)) * up.at(i);

    quantiseRow(gate.values, gate.count, blocks);
    quantisedMatVecPrepared(weights.down, blocks, normed);
    for (core::u32 i = 0u; i < residual.count; ++i)
        residual.at(i) = residual.at(i).saturatingAdd(normed.at(i));
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
