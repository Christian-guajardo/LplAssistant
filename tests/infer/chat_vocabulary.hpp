/**
 * @file chat_vocabulary.hpp
 * @brief The parity vocabulary with tokens of a turn after its own, and the parity model over it, for the tests of a
 *        model's turns.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_TESTS_INFER_CHAT_VOCABULARY_HPP
#    define LPL_TESTS_INFER_CHAT_VOCABULARY_HPP

#    include <lpl/infer/Model.hpp>
#    include <lpl/infer/TensorArena.hpp>
#    include <lpl/infer/Vocab.hpp>

namespace lpl::infer::tests {

/** Tokens of the parity vocabulary: the first token appended after them is this one. */
inline constexpr core::u32 kParityVocabularyTokens = 121u;

/** Byte of an image's header that holds its chat format, as LplAssistant#106 places it. */
inline constexpr core::u32 kHeaderChatFormatByte = 76u;

/**
 * @brief Builds the parity vocabulary with @p count tokens after its own, spelled by @p texts, numbered from
 *        kParityVocabularyTokens in their order.
 *
 * @param arena Storage for the text, the entries and the index.
 * @param texts The appended tokens' texts, each terminated.
 * @param count How many.
 * @param out   Receives the table.
 * @return false when the arena cannot hold it.
 */
[[nodiscard]] bool buildParityVocabularyWith(TensorArena &arena, const char *const *texts, core::u32 count, Vocab &out);

/**
 * @brief Derives the parity model's weights over @p vocab, its context @p contextLength positions, in @p format, its
 *        text split as a whole and segmented by longest match.
 *
 * @param arena         Storage for the weights.
 * @param vocab         The table; its size becomes the model's.
 * @param format        How the model puts messages in turns.
 * @param contextLength Positions the cache holds.
 * @param out           Receives the model.
 * @return false when the arena is short or @p vocab lacks a token @p format needs.
 */
[[nodiscard]] bool synthesiseParityModelOver(TensorArena &arena, const Vocab &vocab, ChatFormat format,
                                             core::u32 contextLength, Model &out);

/**
 * @brief Writes @p value little-endian as the chat format of the image whose header starts at @p image.
 * @param image The image.
 * @param value The format's word.
 */
void putChatFormatWord(core::u8 *image, core::u32 value) noexcept;

} // namespace lpl::infer::tests

#endif /* LPL_TESTS_INFER_CHAT_VOCABULARY_HPP */
