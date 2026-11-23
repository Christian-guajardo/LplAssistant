/**
 * @file Tokenizer.hpp
 * @brief Text to tokens and back, deterministically.
 *
 * A tokenizer that differs by one merge produces a different mind. Pinned and
 * folded like any other authoritative table.
 *
 * Text is first split into pieces by the model's pre-tokenizer, then each piece is segmented. A model that carries
 * ranked merges segments a piece as byte-pair encoding does: its bytes become their single-byte tokens, and the
 * adjacent pair of lowest rank merges first, the leftmost first among equal ranks, until no adjacent pair is a merge.
 * A vocabulary without merges segments a piece by greedy longest match, which derives the segmentation from the
 * vocabulary alone: the parity model's vocabulary is spelled that way.
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

/** Returned where a merge's rank is expected and no merge applies. */
inline constexpr core::u32 kNoMerge = 0xFFFFFFFFu;

/**
 * Symbols one word is merged in at once: a word with more symbols is cut and each part merged apart. The longest word
 * measured on 6,382 lines of prose, code and eleven scripts is 80 bytes (LplAssistant#105).
 */
inline constexpr core::u32 kBytePairWorkspaceSymbols = 4096u;

/**
 * @struct BytePairMerge
 * @brief One ranked merge: two adjacent tokens, and the token whose bytes are theirs one after the other.
 */
struct BytePairMerge {
    core::u32 left{0u};   ///< The token on the left.
    core::u32 right{0u};  ///< The token on the right.
    core::u32 result{0u}; ///< The token the pair becomes.
};

/**
 * @enum BytePairMergesVerdict
 * @brief What indexing a merge list found.
 */
enum class BytePairMergesVerdict : core::u32 {
    Indexed = 0u,             ///< Every merge holds, and the index fit.
    UnknownToken = 1u,        ///< A merge names a token past the vocabulary.
    NotTheConcatenation = 2u, ///< A merge's result is not its left token's bytes then its right token's.
    RepeatedPair = 3u,        ///< Two merges join the same pair.
    ArenaExhausted = 4u,      ///< The arena cannot hold the index.
};

/**
 * @class BytePairMerges
 * @brief A model's ranked merges, borrowed where they lie, and the index from a pair to its rank, in arena storage.
 */
class BytePairMerges {
public:
    BytePairMerges() = default;

    /**
     * @brief Indexes @p count merges, the first of the lowest rank, each checked against @p vocab.
     *
     * The merges are borrowed: they must outlive the table and must not change. The index claims a power of two of
     * 32-bit slots, at least twice the merges, so that looking a pair up probes fewer than three slots on average.
     *
     * @param arena  Storage for the index; nothing is claimed for no merge.
     * @param merges The merges in the order of their ranks; nullptr only when @p count is 0.
     * @param count  Merges.
     * @param vocab  The vocabulary every token of a merge belongs to.
     * @return Indexed; otherwise what was refused, and the table holds no merge.
     */
    [[nodiscard]] BytePairMergesVerdict borrow(TensorArena &arena, const BytePairMerge *merges, core::u32 count,
                                               const Vocab &vocab);

    /**
     * @brief Merges in the table.
     * @return The count; 0 for a table that holds none.
     */
    [[nodiscard]] core::u32 size() const noexcept { return _count; }

    /**
     * @brief The merges, in the order of their ranks.
     * @return The list, or nullptr when the table holds none.
     */
    [[nodiscard]] const BytePairMerge *merges() const noexcept { return _merges; }

    /**
     * @brief The rank of the merge that joins @p left then @p right.
     * @param left  The token on the left.
     * @param right The token on the right.
     * @return The rank, or @ref kNoMerge when no merge joins them.
     */
    [[nodiscard]] core::u32 rank(core::u32 left, core::u32 right) const noexcept;

private:
    const BytePairMerge *_merges{nullptr};
    const core::u32 *_slots{nullptr};
    core::u32 _count{0u};
    core::u32 _slotMask{0u};
};

/**
 * @enum PreTokenizer
 * @brief How a model splits text into the pieces it segments apart.
 */
enum class PreTokenizer : core::u32 {
    WholeText = 0u,                     ///< The text is one piece.
    IndividualDigitsThenByteLevel = 1u, ///< Each number, then GPT-2's pattern (preTokenizedPieceBytes): SmolLM2's.
};

/**
 * @brief Whether @p value names a pre-tokenizer this reader knows.
 * @param value A word read from an image.
 * @return true for the value of a @ref PreTokenizer.
 */
[[nodiscard]] bool isKnownPreTokenizer(core::u32 value) noexcept;

/**
 * @enum CharacterClass
 * @brief What a pre-tokenizer's pattern makes of a code point.
 */
enum class CharacterClass : core::u8 {
    Other = 0u,  ///< None of the three below, an invalid byte of UTF-8 included.
    Letter = 1u, ///< General category L*: \p{L}.
    Number = 2u, ///< General category N*: \p{N}.
    Space = 3u,  ///< The property White_Space: \s.
};

/**
 * @brief The class of @p codePoint, from the Unicode Character Database 16.0.0, which Hugging Face tokenizers 0.22.1
 *        reads.
 * @param codePoint Any value; one past U+10FFFF is other.
 * @return Its class.
 */
[[nodiscard]] CharacterClass characterClassOf(core::u32 codePoint) noexcept;

/**
 * @brief The bytes of the piece that starts at @p at of @p text, as @p preTokenizer splits it.
 *
 * Text is read as UTF-8; each byte that does not start a valid sequence is one character of class other, its byte
 * kept, so the pieces always spell the text.
 *
 * IndividualDigitsThenByteLevel isolates each number, as Hugging Face's Digits with individual_digits does, then splits
 * the rest as its ByteLevel does, by GPT-2's pattern
 * `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, an isolated number ending the text that
 * pattern sees.
 *
 * @param preTokenizer How the text splits.
 * @param text         The bytes.
 * @param bytes        How many.
 * @param at           Where the piece starts: the end of the piece before it, or 0.
 * @return The piece's bytes, at least 1; 0 when @p at is not before @p bytes.
 */
[[nodiscard]] core::u32 preTokenizedPieceBytes(PreTokenizer preTokenizer, const char *text, core::u32 bytes,
                                               core::u32 at) noexcept;

/**
 * @brief The words every encoding since the start has cut, for want of room, into parts merged apart.
 * @return 0 while every word fits kBytePairWorkspaceSymbols; anything else says a word's tokens may differ from the
 *         model's own tokenizer at a cut.
 */
[[nodiscard]] core::u32 bytePairWordsCut() noexcept;

/**
 * @class Tokenizer
 * @brief Segments text against a vocabulary, without allocating.
 */
class Tokenizer {
public:
    Tokenizer() = default;

    /**
     * @brief Binds a vocabulary without merges, segmented by longest match over the whole text.
     * @param vocab The table; must outlive the tokenizer.
     */
    explicit Tokenizer(const Vocab &vocab) noexcept : _vocab(&vocab) {}

    /**
     * @brief Binds a vocabulary, its merges and its pre-tokenizer.
     *
     * Claims, for a table that holds merges, the token of each byte and the workspace of one word's merging:
     * kBytePairWorkspaceSymbols symbols of six words each. A table without merges claims nothing and segments each
     * piece by longest match.
     *
     * @param arena        Storage for the byte table and the workspace.
     * @param vocab        The table; must outlive the tokenizer.
     * @param merges       Its merges, indexed against @p vocab; must outlive the tokenizer.
     * @param preTokenizer How text splits into pieces.
     * @return false, the tokenizer unbound, when the arena cannot hold what it claims.
     */
    [[nodiscard]] bool bind(TensorArena &arena, const Vocab &vocab, const BytePairMerges &merges,
                            PreTokenizer preTokenizer);

    /**
     * @brief Segments @p text into tokens.
     *
     * A byte the vocabulary cannot spell is DROPPED, and it is worth being explicit
     * about why that is not a silent failure: the caller is told, through the
     * returned count and @p outSkipped, that its text did not survive intact. The
     * alternative — an unknown-token identifier — would put a symbol in the context
     * that the model has never seen and cannot have learnt anything about. Merging drops a byte that has no
     * single-byte token the same way, as the model's own tokenizer does when it has no unknown token.
     *
     * Not reentrant once merges are bound: a word is merged in the workspace the binding claimed, so one call at a
     * time per tokenizer.
     *
     * @param text       Bytes to segment.
     * @param bytes      How many.
     * @param out        Receives the identifiers.
     * @param capacity   Room in @p out; the tokens past it are not written.
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
    [[nodiscard]] core::u32 matchPiece(const char *piece, core::u32 bytes, core::u32 *out, core::u32 room,
                                       core::u32 &skipped) const noexcept;
    [[nodiscard]] core::u32 mergePiece(const char *piece, core::u32 bytes, core::u32 *out, core::u32 room,
                                       core::u32 &skipped) const noexcept;

    const Vocab *_vocab{nullptr};
    const BytePairMerges *_merges{nullptr};
    PreTokenizer _preTokenizer{PreTokenizer::WholeText};
    core::u32 *_byteTokens{nullptr};
    core::u32 *_workspace{nullptr};
};

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_TOKENIZER_HPP
