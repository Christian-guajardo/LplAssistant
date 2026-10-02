/**
 * @file Tensor.cpp
 * @brief Non-owning views with static rank and checked extents.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Tensor.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::infer {

void quantisedMatVec(const QuantMatrixView &matrix, ConstVectorView input, QuantBlock *scratch,
                     VectorView out) noexcept
{
    LPL_VERIFY(input.count == matrix.columns);
    LPL_VERIFY(out.count == matrix.rows);
    LPL_VERIFY(scratch != nullptr);

    quantiseRow(input.values, input.count, scratch);
    quantisedMatVecPrepared(matrix, scratch, out);
}

void quantisedMatVecPrepared(const QuantMatrixView &matrix, const QuantBlock *input, VectorView out) noexcept
{
    LPL_VERIFY(out.count == matrix.rows);
    LPL_VERIFY(input != nullptr);

    const core::u32 blocks = matrix.blocksPerRow();
    for (core::u32 r = 0u; r < matrix.rows; ++r)
        out.at(r) = dotQuantised(matrix.row(r), input, blocks);
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
