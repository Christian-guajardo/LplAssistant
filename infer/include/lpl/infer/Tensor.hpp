/**
 * @file Tensor.hpp
 * @brief Non-owning views with static rank and checked extents.
 *
 * Views, never containers: nothing in the forward pass is allowed to allocate.
 * Every buffer a view points at was claimed from the arena before the first token,
 * and the views exist so that a shape mismatch is caught where it happens rather
 * than three layers later as a wrong number.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_TENSOR_HPP
#    define LPL_LPL_INFER_TENSOR_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/Quant.hpp>

namespace lpl::infer {

/**
 * @struct VectorView
 * @brief A writable run of activations.
 */
struct VectorView {
    math::Fixed32 *values{nullptr};
    core::u32 count{0u};

    /**
     * @brief Element access, bounds-checked.
     * @param index Position.
     * @return Reference to the element.
     */
    [[nodiscard]] math::Fixed32 &at(core::u32 index) const noexcept
    {
        LPL_VERIFY(values != nullptr && index < count);
        return values[index];
    }

    /**
     * @brief A window onto part of this run.
     * @param offset First element of the window.
     * @param length Elements in it.
     * @return The window, which shares storage with this view.
     */
    [[nodiscard]] VectorView slice(core::u32 offset, core::u32 length) const noexcept
    {
        LPL_VERIFY(offset + length <= count);
        return VectorView{values + offset, length};
    }
};

/**
 * @struct ConstVectorView
 * @brief A read-only run of activations.
 */
struct ConstVectorView {
    const math::Fixed32 *values{nullptr};
    core::u32 count{0u};

    ConstVectorView() = default;

    /**
     * @brief Wraps a raw run.
     * @param data  First element.
     * @param size  Elements.
     */
    ConstVectorView(const math::Fixed32 *data, core::u32 size) noexcept : values(data), count(size) {}

    /**
     * @brief Adopts a writable view.
     * @param view The view to read from.
     */
    ConstVectorView(const VectorView &view) noexcept : values(view.values), count(view.count) {}

    /**
     * @brief Element access, bounds-checked.
     * @param index Position.
     * @return The element.
     */
    [[nodiscard]] math::Fixed32 at(core::u32 index) const noexcept
    {
        LPL_VERIFY(values != nullptr && index < count);
        return values[index];
    }
};

/**
 * @struct QuantMatrixView
 * @brief A quantised weight matrix, row-major.
 *
 * Rows are padded to whole blocks, so a row's block count is a property of the
 * matrix and not of where the row happens to start. That padding is what lets
 * @ref quantisedMatVec index a row by multiplication instead of by carrying an
 * offset table.
 */
struct QuantMatrixView {
    const QuantBlock *blocks{nullptr};
    core::u32 rows{0u};
    core::u32 columns{0u};

    /**
     * @brief Blocks one row occupies.
     * @return The per-row block count.
     */
    [[nodiscard]] core::u32 blocksPerRow() const noexcept { return quantBlockCount(columns); }

    /**
     * @brief First block of a row.
     * @param row Row index.
     * @return Pointer to its blocks.
     */
    [[nodiscard]] const QuantBlock *row(core::u32 row) const noexcept
    {
        LPL_VERIFY(blocks != nullptr && row < rows);
        return blocks + static_cast<core::usize>(row) * blocksPerRow();
    }
};

/**
 * @brief Matrix times vector, both quantised.
 *
 * The input is quantised once into @p scratch and reused for every row, which is
 * the only reason this is a function rather than a loop over @ref dotQuantised:
 * quantising the activation per row would repeat identical work once per output and
 * would let a row's own scale leak into a value that does not belong to it.
 *
 * @param matrix  Weights.
 * @param input   Activations; its length must equal @c matrix.columns.
 * @param scratch Room for @ref quantBlockCount(matrix.columns) blocks.
 * @param out     Receives @c matrix.rows values.
 */
void quantisedMatVec(const QuantMatrixView &matrix, ConstVectorView input, QuantBlock *scratch,
                     VectorView out) noexcept;

/**
 * @brief Matrix times an already-quantised vector.
 *
 * The form a block uses when the same activation feeds several matrices — the query,
 * key and value projections all read one normalised residual. Quantising it once and
 * passing it three times is not only cheaper: it guarantees the three projections
 * saw the same input, which separate quantisations of the same values would give but
 * only by coincidence.
 *
 * @param matrix Weights.
 * @param input  Blocks covering @c matrix.columns values.
 * @param out    Receives @c matrix.rows values.
 */
void quantisedMatVecPrepared(const QuantMatrixView &matrix, const QuantBlock *input, VectorView out) noexcept;

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_TENSOR_HPP
