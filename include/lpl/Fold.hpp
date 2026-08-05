/**
 * @file Fold.hpp
 * @brief The one FNV-1a used by every gate in this repository.
 *
 * Written four times before this header existed — twice in `infer/`, twice in
 * `satellite/` — with the same two constants and the same single line of arithmetic.
 * Four copies of a hash is not four chances to be slow, it is four chances for one of
 * them to be changed and the gate that depends on it to start disagreeing with a
 * sibling for a reason nobody can see. A signature's whole job is to be the same
 * number on two machines, so the function that produces it is the last thing that
 * should exist more than once.
 *
 * The constants are the project's, not a choice made here: offset basis 0x811C9DC5
 * and prime 0x01000193, the same pair every fold in LplKernel and LplPlugin uses, so
 * a signature computed on one side of the seam is comparable with one computed on the
 * other.
 *
 * Everything is `constexpr` and takes only fixed-width words, so it compiles into a
 * freestanding kernel exactly as it compiles on the host — which is the property the
 * gates are built on.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_FOLD_HPP
#    define LPL_FOLD_HPP

#    include <lpl/Foundation.hpp>

namespace lpl {

/// FNV-1a offset basis, and the value every signature in this project starts from.
inline constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;

/// FNV-1a prime.
inline constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Folds one 32-bit word into a running hash.
 *
 * @param hash Running value, updated in place.
 * @param word Word to absorb.
 */
constexpr void foldWord(core::u32 &hash, core::u32 word) noexcept { hash = (hash ^ word) * kFnv1aPrime; }

/**
 * @brief Folds a run of bytes into a running hash.
 *
 * Byte at a time rather than word at a time, because a word-wise fold of a byte
 * buffer would depend on the machine's endianness — which is exactly the kind of
 * silent disagreement between two targets that a signature exists to catch, and
 * would therefore hide it instead.
 *
 * @param hash  Running value, updated in place.
 * @param bytes Start of the run; may be null when @p count is zero.
 * @param count How many bytes.
 */
constexpr void foldBytes(core::u32 &hash, const core::u8 *bytes, core::u32 count) noexcept
{
    for (core::u32 i = 0u; i < count; ++i)
        foldWord(hash, static_cast<core::u32>(bytes[i]));
}

} // namespace lpl

#endif // LPL_FOLD_HPP
