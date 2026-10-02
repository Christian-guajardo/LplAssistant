/**
 * @file Inference.hpp
 * @brief The façade: prompt in, tokens out, within a budget.
 *
 * One entry point, so the ring-0 and hosted paths cannot drift.
 *
 * The budget is in TOKENS, not in milliseconds. A wall clock inside a replayable
 * path would make the answer depend on how fast the machine was that day, which is
 * the same reason @c engine::InferenceBudget counts turns: everything downstream of
 * a demon's answer is deterministic, so the answer has to be too.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_INFERENCE_HPP
#    define LPL_LPL_INFER_INFERENCE_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/GrammarSampler.hpp>
#        include <lpl/infer/Sampler.hpp>
#        include <lpl/infer/Transformer.hpp>

namespace lpl::infer {

/**
 * @struct GenerationParams
 * @brief What one generation is allowed to do.
 */
struct GenerationParams {
    SamplerParams sampler{}; ///< How each token is chosen.
    core::u32 maxTokens{0u}; ///< Ceiling on tokens produced, budget included.
};

/**
 * @struct GenerationReport
 * @brief What one generation did.
 */
struct GenerationReport {
    core::u32 promptTokens{0u};   ///< Positions consumed feeding the prompt.
    core::u32 generated{0u};      ///< Tokens produced.
    core::u32 evictions{0u};      ///< Times the window slid.
    core::u32 draws{0u};          ///< Random words the sampler consumed.
    core::u32 admittedLast{0u};   ///< Tokens the grammar allowed at the final step.
    bool windowExhausted{false};  ///< The cache refused a position and stopped it.
    bool grammarExhausted{false}; ///< The grammar admitted nothing and stopped it.
    bool grammarComplete{false};  ///< The output is a whole phrase of the grammar.
};

/**
 * @class Inference
 * @brief Weights, cache, scratch and sampler, wired into one call.
 */
class Inference {
public:
    Inference() = default;

    /**
     * @brief Claims the cache and every working buffer.
     * @param arena  Storage.
     * @param model  The weights.
     * @param policy What the cache does when full.
     * @return false when the model is unbuilt or the arena is exhausted.
     */
    bool initialise(TensorArena &arena, const Model &model, CachePolicy policy);

    /**
     * @brief Feeds a prompt and produces tokens.
     *
     * The prompt is fed one position at a time and its logits are discarded, which
     * is the cost of a cache that holds one token's key per slot. A batched prefill
     * would be faster and would have to sum the same products in a different order,
     * so it belongs behind its own parity gate rather than inside this one.
     *
     * @param prompt      Identifiers to feed.
     * @param promptCount How many.
     * @param params      Ceiling and sampling rule.
     * @param grammar     Optional constraint; nullptr leaves the vocabulary open.
     * @param out         Receives the generated identifiers.
     * @param capacity    Room in @p out.
     * @param report      Receives what happened.
     * @return false when the model is unbuilt or the prompt does not fit the window.
     */
    [[nodiscard]] bool generate(const core::u32 *prompt, core::u32 promptCount, const GenerationParams &params,
                                const Grammar *grammar, core::u32 *out, core::u32 capacity,
                                GenerationReport &report) noexcept;

    /**
     * @brief The scores of the last position evaluated.
     * @return The logits.
     */
    [[nodiscard]] ConstVectorView logits() const noexcept { return _logits; }

    /**
     * @brief The residual stream of the last position evaluated.
     * @return The stream.
     */
    [[nodiscard]] ConstVectorView residual() const noexcept { return _transformer.residual(); }

    /**
     * @brief The cache this façade owns.
     * @return The window.
     */
    [[nodiscard]] const KvCache &cache() const noexcept { return _cache; }

private:
    const Model *_model{nullptr};
    Transformer _transformer{};
    KvCache _cache{};
    VectorView _logits{};
    bool *_allowed{nullptr};
};

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_INFERENCE_HPP
