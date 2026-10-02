/**
 * @file Tokenizer.hpp
 * @brief Text to tokens and back, deterministically.
 *
 * A tokenizer that differs by one merge produces a different mind. Pinned and
 * folded like any other authoritative table.
 *
 * Greedy longest match, and the choice is stated rather than assumed. Byte-pair
 * encoding proper replays a ranked merge list, which is a second table to keep in
 * agreement with the vocabulary and a second place for the two sides of the gate to
 * disagree. Longest match derives the segmentation from the vocabulary alone: there
 * is one table, so there is one answer.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_TOKENIZER_HPP
#    define LPL_LPL_INFER_TOKENIZER_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/Vocab.hpp>

namespace lpl::infer {

/**
 * @class Tokenizer
 * @brief Segments text against a vocabulary, without allocating.
 */
class Tokenizer {
public:
    Tokenizer() = default;

    /**
     * @brief Binds a vocabulary.
     * @param vocab The table; must outlive the tokenizer.
     */
    explicit Tokenizer(const Vocab &vocab) noexcept : _vocab(&vocab) {}

    /**
     * @brief Segments @p text into tokens.
     *
     * A byte the vocabulary cannot spell is DROPPED, and it is worth being explicit
     * about why that is not a silent failure: the caller is told, through the
     * returned count and @p outSkipped, that its text did not survive intact. The
     * alternative — an unknown-token identifier — would put a symbol in the context
     * that the model has never seen and cannot have learnt anything about.
     *
     * @param text       Bytes to segment.
     * @param bytes      How many.
     * @param out        Receives the identifiers.
     * @param capacity   Room in @p out.
     * @param outSkipped Receives the count of unspellable bytes.
     * @return Tokens written.
     */
    [[nodiscard]] core::u32 encode(const char *text, core::u32 bytes, core::u32 *out, core::u32 capacity,
                                   core::u32 &outSkipped) const noexcept;

    /**
     * @brief Writes the bytes @p tokens stand for.
     *
     * @param tokens   Identifiers.
     * @param count    How many.
     * @param out      Receives the bytes; not terminated.
     * @param capacity Room in @p out.
     * @return Bytes written.
     */
    [[nodiscard]] core::u32 decode(const core::u32 *tokens, core::u32 count, char *out,
                                   core::u32 capacity) const noexcept;

    /**
     * @brief The table this tokenizer segments against.
     * @return The vocabulary, or nullptr when unbound.
     */
    [[nodiscard]] const Vocab *vocab() const noexcept { return _vocab; }

private:
    const Vocab *_vocab{nullptr};
};

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_TOKENIZER_HPP
