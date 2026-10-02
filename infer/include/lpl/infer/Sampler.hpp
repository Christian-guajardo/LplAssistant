/**
 * @file Sampler.hpp
 * @brief Choosing the next token, reproducibly.
 *
 * Seeded and deterministic. A demon whose answers cannot be replayed cannot be
 * audited, and the whole project is built on replay.
 *
 * The draw is integer end to end — cumulative raw Q16.16 weights and one bounded
 * draw from the project's own xorshift. A float cumulative sum would make the
 * chosen token depend on the order the compiler summed it in, which is the one
 * property a replay must not have.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_SAMPLER_HPP
#    define LPL_LPL_INFER_SAMPLER_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/Tensor.hpp>
#        include <lpl/math/Random.hpp>

namespace lpl::infer {

/**
 * @brief Candidates a sampled draw may consider.
 *
 * Bounded so the shortlist lives on the stack: the forward pass is allocation-free
 * after init, and a sampler that reached for the arena would be the one place per
 * token that could fail for a reason unrelated to the model.
 */
inline constexpr core::u32 kMaxTopK = 64u;

/**
 * @struct SamplerParams
 * @brief How the next token is chosen.
 */
struct SamplerParams {
    /**
     * Divides the scores before they become probabilities. Below one sharpens the
     * distribution, above one flattens it. Zero or negative is treated as greedy,
     * because dividing by it has no meaning and silently picking a nearby value
     * would hide a caller's mistake.
     */
    math::Fixed32 temperature{math::Fixed32::one()};

    /// Candidates kept; 0 or 1 means take the best one and draw nothing.
    core::u32 topK{0u};

    /// Anchors the stream. The same seed and the same scores give the same tokens.
    core::u32 seed{0u};
};

/**
 * @class Sampler
 * @brief Turns scores into a token, with a stream it owns.
 */
class Sampler {
public:
    /**
     * @brief Seeds the stream.
     * @param params How to choose.
     */
    explicit Sampler(const SamplerParams &params) noexcept : _params(params), _stream(params.seed) {}

    /**
     * @brief Picks the next token.
     *
     * A masked-out candidate is not merely made unlikely, it is skipped entirely —
     * which is what lets a grammar make a malformed call unrepresentable rather than
     * improbable. Ties go to the lower identifier, stated because a tie is common on
     * a small vocabulary and "whichever the loop saw last" is not a rule two targets
     * can be held to.
     *
     * @param logits  Scores, one per token; modified in place.
     * @param allowed Optional mask, one flag per token; nullptr allows everything.
     * @return The chosen identifier, or @ref kNoToken when the mask forbids all.
     */
    [[nodiscard]] core::u32 next(VectorView logits, const bool *allowed) noexcept;

    /**
     * @brief Draws taken from the stream so far.
     *
     * Folded by the gate: two targets that agreed on every token while consuming a
     * different number of random words would agree by luck, and the next prompt
     * would part them.
     *
     * @return The draw count.
     */
    [[nodiscard]] core::u32 draws() const noexcept { return _draws; }

private:
    SamplerParams _params;
    math::Random _stream;
    core::u32 _draws{0u};
};

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_SAMPLER_HPP
