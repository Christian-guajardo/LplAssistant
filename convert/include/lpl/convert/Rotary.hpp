/**
 * @file Rotary.hpp
 * @brief A model's rotary table, computed once on the host from its base.
 *
 * Pair i of a head of d channels turns by base^(-2i/d) radians per position, as llama.cpp b3775's ggml_rope_cache_init
 * computes it (ggml.c:14530-14544). The image holds it in units of 2^-32 turn: round(base^(-2i/d) / (2 pi) 2^32),
 * which ring 0 multiplies by the position modulo 2^32, exact at every position (lpl::infer::RotaryFrequencies).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CONVERT_ROTARY_HPP
#    define LPL_LPL_CONVERT_ROTARY_HPP

#    include <lpl/convert/Source.hpp>

#    include <cstdint>
#    include <optional>
#    include <vector>

namespace lpl::convert {

/** How close to a rounding tie a word may lie, in units: closer, a libm one ulp off could round it the other way. */
inline constexpr double kRotaryLeastDistanceToTie = 1e-6;

/**
 * @brief Computes how far each pair of a head of @p dimensions channels turns per position at @p base, in units of
 *        2^-32 turn, rounded half away from zero, in long double arithmetic.
 *
 * @param base       The model's rope_theta.
 * @param dimensions The channels the rotation turns, even.
 * @param out        Receives dimensions / 2 words.
 * @return Nothing when every word is written; `metadata_out_of_range` for a base that is not finite or not past 1, or
 *         for no channel or an odd count of them; `rotary_near_tie` naming the first pair whose word lies within
 *         kRotaryLeastDistanceToTie of a tie.
 */
[[nodiscard]] std::optional<Refusal> rotaryTurnsPerPosition(double base, std::uint32_t dimensions,
                                                            std::vector<std::uint32_t> &out);

} // namespace lpl::convert

#endif /* LPL_LPL_CONVERT_ROTARY_HPP */
