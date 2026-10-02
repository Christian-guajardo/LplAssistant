/**
 * @file Vocab.hpp
 * @brief The token table.
 *
 * Bounded, sorted, searched without allocating.
 *
 * Two tables, not one. The tokens are held in declaration order because a token's
 * identifier is its index and reordering the table would change every model that
 * ever referred to it; the search goes through a separate index sorted by bytes, so
 * lookup is a binary search and the identifiers stay put. Conflating the two would
 * force a choice between a stable vocabulary and a fast one.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_VOCAB_HPP
#    define LPL_LPL_INFER_VOCAB_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/TensorArena.hpp>

namespace lpl::infer {

/// Returned where a token identifier is expected and none applies.
inline constexpr core::u32 kNoToken = 0xFFFFFFFFu;

/// Longest byte string one token may stand for.
inline constexpr core::u32 kMaxTokenBytes = 16u;

/**
 * @struct VocabEntry
 * @brief Where one token's bytes live in the blob.
 */
struct VocabEntry {
    core::u32 offset{0u};
    core::u32 length{0u};
};

/**
 * @class Vocab
 * @brief An immutable token table over arena storage.
 */
class Vocab {
public:
    Vocab() = default;

    /**
     * @brief Builds a table from a blob and its entries.
     *
     * Copies both into @p arena, so the caller's tables may be temporaries — a
     * vocabulary that borrowed a stack buffer would be a dangling pointer the first
     * time it was used from another frame.
     *
     * @param arena   Storage.
     * @param blob    Concatenated token bytes.
     * @param entries One per token, in identifier order.
     * @param count   Tokens.
     * @return true when everything fit.
     */
    bool build(TensorArena &arena, const char *blob, core::u32 blobBytes, const VocabEntry *entries, core::u32 count);

    /**
     * @brief Tokens in the table.
     * @return The count.
     */
    [[nodiscard]] core::u32 size() const noexcept { return _count; }

    /**
     * @brief The bytes one token stands for.
     * @param token   Identifier.
     * @param outLength Receives the byte count.
     * @return Pointer into the blob, or nullptr for an unknown identifier.
     */
    [[nodiscard]] const char *text(core::u32 token, core::u32 &outLength) const noexcept;

    /**
     * @brief The token whose bytes are exactly @p bytes.
     * @param bytes  Candidate text.
     * @param length Its length.
     * @return The identifier, or @ref kNoToken.
     */
    [[nodiscard]] core::u32 find(const char *bytes, core::u32 length) const noexcept;

    /**
     * @brief The longest token that is a prefix of @p bytes.
     *
     * The primitive the tokenizer is built on. Returning the length as well as the
     * identifier saves the caller re-deriving it, which is where a greedy tokenizer
     * would otherwise be able to disagree with itself.
     *
     * @param bytes     Text to match against.
     * @param available Bytes readable from it.
     * @param outLength Receives the matched length.
     * @return The identifier, or @ref kNoToken when nothing matches.
     */
    [[nodiscard]] core::u32 longestPrefix(const char *bytes, core::u32 available, core::u32 &outLength) const noexcept;

    /**
     * @brief The concatenated token bytes.
     *
     * Exposed so a model can be written back out. The sorted index is deliberately
     * NOT exposed: it is derived from the other two, and a serialiser that wrote it
     * would let a file arrive whose index disagreed with its own table.
     *
     * @return The blob, or nullptr when unbuilt.
     */
    [[nodiscard]] const char *blob() const noexcept { return _blob; }

    /**
     * @brief Bytes in the blob.
     * @return The length.
     */
    [[nodiscard]] core::u32 blobBytes() const noexcept { return _blobBytes; }

    /**
     * @brief The per-token extents, in identifier order.
     * @return The entries, or nullptr when unbuilt.
     */
    [[nodiscard]] const VocabEntry *entries() const noexcept { return _entries; }

private:
    const char *_blob{nullptr};
    const VocabEntry *_entries{nullptr};
    const core::u32 *_sorted{nullptr};
    core::u32 _count{0u};
    core::u32 _blobBytes{0u};
};

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_VOCAB_HPP
