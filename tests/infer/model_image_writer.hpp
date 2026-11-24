/**
 * @file model_image_writer.hpp
 * @brief Writing a model's image where the tests of infer/ read it back.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_TESTS_INFER_MODEL_IMAGE_WRITER_HPP
#    define LPL_TESTS_INFER_MODEL_IMAGE_WRITER_HPP

#    include <lpl/infer/Model.hpp>
#    include <lpl/infer/TensorArena.hpp>

namespace lpl::infer::tests {

/** The byte the room of an image holds before it is written: a byte the writer leaves unwritten shows. */
inline constexpr core::u8 kUnwrittenByte = 0xA5u;

/**
 * @brief Writes @p model's image into @p arena, over room whose every byte is kUnwrittenByte first.
 *
 * @param arena         Storage for the image.
 * @param model         A model ready to be written.
 * @param alignmentLog2 Each expert's alignment, as Model::writeBlob takes it.
 * @param shift         Bytes the image's first byte lies past a word boundary, below 4.
 * @param size          Receives the image's bytes.
 * @return The image, or nullptr when the writer refused or the arena is short.
 */
[[nodiscard]] core::u8 *writeModelImage(TensorArena &arena, const Model &model, core::u32 alignmentLog2,
                                        core::u32 shift, core::u32 &size);

} // namespace lpl::infer::tests

#endif /* LPL_TESTS_INFER_MODEL_IMAGE_WRITER_HPP */
