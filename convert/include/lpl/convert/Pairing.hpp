/**
 * @file Pairing.hpp
 * @brief Which channels of a head a model's rotation turns together, and how its query and key rows are brought to the
 *        pairs ring 0 turns.
 *
 * Ring 0 turns channels 2i and 2i + 1 of each head together (lpl::infer::applyRotaryEmbedding). A `llama` GGUF holds
 * its rows so already: llama.cpp's converter permutes Hugging Face's query and key rows for it (convert_hf_to_gguf.py
 * :1555-1571), and llama.cpp turns consecutive channels (llama.cpp:19154-19169). Other architectures turn channels i
 * and i + d/2 together (NEOX, llama.cpp:19171-19194) and their rows are not permuted: row 2i + s of a head is then
 * read from the source's row s d/2 + i, which turns each pair as the source turns it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CONVERT_PAIRING_HPP
#    define LPL_LPL_CONVERT_PAIRING_HPP

#    include <cstdint>

namespace lpl::convert {

/**
 * @enum RotaryPairing
 * @brief Which channels of a head a model turns together.
 */
enum class RotaryPairing : std::uint8_t {
    Adjacent,  ///< Channels 2i and 2i + 1, as ring 0 turns them: llama.cpp's NORM.
    HalfSplit, ///< Channels i and i + d/2: llama.cpp's NEOX.
};

/**
 * @brief The row of a source query or key matrix that row @p row of the image's takes, for heads of @p headDim
 *        channels paired as @p pairing.
 * @param row     A row of the image's matrix.
 * @param headDim Channels of a head; even for HalfSplit.
 * @param pairing How the source pairs its channels.
 * @return @p row for Adjacent; for HalfSplit, the row of the same head that holds the same channel of the same pair.
 */
[[nodiscard]] std::uint32_t sourceRowOf(std::uint32_t row, std::uint32_t headDim, RotaryPairing pairing) noexcept;

} // namespace lpl::convert

#endif /* LPL_LPL_CONVERT_PAIRING_HPP */
