/**
 * @file Recall.hpp
 * @brief The interface to the knowledge side.
 *
 * Declared here, implemented over a knowledge pack. Structured filter first, similarity
 * second: the demon narrows by fact before it reaches for resemblance.
 *
 * That order is the whole design. Resemblance is cheap to compute and easy to be wrong
 * about — two notes can share every word and be about different reactors — so it is
 * asked LAST, of a set that a fact has already narrowed. Inverting the two would let a
 * confident-looking similarity score outvote something the demon actually knows.
 *
 * The similarity itself is integer arithmetic end to end: each text is reduced to a
 * 64-bit signature of hashed words, and the score is the overlap of two signatures over
 * their union, in `Fixed32`. No dot products, no square roots, no libm — this module is
 * compiled into a kernel that may not call any of them, and a score that used a
 * different code path on the host would make a recall gate meaningless.
 *
 * @warning A signature of 64 bits collides. Two unrelated texts can share bits, so a score is
 * a shortlist and never a verdict — which is exactly why the structured filter runs
 * first and why the caller gets the score rather than a boolean.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_MIND_RECALL_HPP
#    define LPL_LPL_MIND_RECALL_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/math/FixedPoint.hpp>
#        include <lpl/mind/Memory.hpp>

namespace lpl::mind {

/**
 * @struct TextSignature
 * @brief A text reduced to the set of words it contains.
 *
 * Two words rather than one 64-bit value because `core::u64` arithmetic is the kind of
 * thing that differs between a 64-bit host and i686, and the whole point of this type
 * is that both compute the same bits.
 */
struct TextSignature {
    core::u32 low{0u};  ///< Bits 0 to 31 of the word set.
    core::u32 high{0u}; ///< Bits 32 to 63.
};

/**
 * @struct RecallHit
 * @brief One note the lookup surfaced, and how well it matched.
 */
struct RecallHit {
    core::u32 index{0u};       ///< Position in the store.
    math::Fixed32 score{};     ///< Overlap over union, between zero and one.
};

/**
 * @brief Reduces a text to the set of words it contains.
 *
 * @param text  Bytes to reduce; may be null when @p count is zero.
 * @param count How many.
 * @return The signature.
 */
[[nodiscard]] TextSignature signatureOf(const char *text, core::u32 count) noexcept;

/**
 * @brief How much two texts have in common.
 *
 * @param left  One signature.
 * @param right The other.
 * @return Shared bits over total bits; zero when both are empty.
 */
[[nodiscard]] math::Fixed32 similarity(const TextSignature &left, const TextSignature &right) noexcept;

/**
 * @brief Finds the notes worth putting in front of the demon.
 *
 * Results come back ordered by score, highest first, ties broken by the lower store
 * index so the order is the same on both targets.
 *
 * @param store        Notes to search.
 * @param topic        Structured filter; zero matches every topic.
 * @param query        Text to resemble; may be null when @p queryBytes is zero.
 * @param queryBytes   How many bytes of query.
 * @param minimumScore Scores below this are not returned at all.
 * @param out          Receives the hits.
 * @param capacity     Room in @p out.
 * @return Hits written.
 */
core::u32 recall(const MemoryStore &store, core::u32 topic, const char *query, core::u32 queryBytes,
                 math::Fixed32 minimumScore, RecallHit *out, core::u32 capacity) noexcept;

/**
 * @brief Folds a set of hits into a signature.
 *
 * @param hits  Hits to fold.
 * @param count How many.
 * @param hash  Running value.
 * @return The updated hash.
 */
[[nodiscard]] core::u32 foldRecall(const RecallHit *hits, core::u32 count, core::u32 hash) noexcept;

} // namespace lpl::mind

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_MIND_RECALL_HPP
