/**
 * @file Reasoning.hpp
 * @brief The reasoner that actually runs a model.
 *
 * `DeterministicReasoner` decides by rule; this one decides by generation, under a
 * grammar rebuilt from the world at every step. It is the point where `infer/` and
 * `mind/` meet, and where the demon in ring 0 stops following a policy and starts
 * thinking.
 *
 * SEPARATE HEADER, and the reason is a build one rather than a taste one. `ReAct.hpp`
 * carries the loop and the model-free policy, and is included by `Parity.hpp`,
 * `Dialogue.hpp` and every consumer of a transcript. Putting this class there would
 * drag the whole transformer into all of them — so the file that needs a model is the
 * only file that includes one. The seam itself is in neither: `agent::IDecider` lives
 * in LplPlugin, so the hosted demon and this one plug into the same thing.
 *
 * THE CLAIM THIS EXISTS TO MAKE, and it is measurable rather than rhetorical: a model
 * that has learned nothing still cannot name an action the world does not offer.
 * The guarantee comes from the language, not from the model's competence, so it does
 * not weaken as the model gets smaller — which is the whole argument for putting
 * inference behind a grammar instead of behind a validator. A validator rejects a bad
 * call after it was generated; a grammar means the bad call was never spellable.
 *
 * @warning An action ALREADY TAKEN is left out of the language rather than filtered out of
 * the result. The difference matters: filtering afterwards means the model spends its
 * whole budget re-proposing the move it just made and being told no, while a phrase
 * that is not in the grammar cannot be emitted in the first place. It also puts the
 * anti-loop guard in the same place as every other rule about what is possible now.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_MIND_REASONING_HPP
#    define LPL_LPL_MIND_REASONING_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/agent/Decision.hpp>
#        include <lpl/infer/GrammarSampler.hpp>
#        include <lpl/infer/Inference.hpp>
#        include <lpl/infer/Model.hpp>
#        include <lpl/infer/TensorArena.hpp>
#        include <lpl/infer/Tokenizer.hpp>
#        include <lpl/mind/ReAct.hpp>

namespace lpl::mind {

/// Actions one step's grammar may offer. The alphabet is regenerated, so it stays small.
inline constexpr core::u32 kMaxLegalPhrases = 16u;

/// Bytes of prompt fed before a decision.
inline constexpr core::u32 kReasoningPromptBytes = 12u;

/**
 * @class ModelReasoner
 * @brief Chooses the next act by constrained generation.
 */
class ModelReasoner final : public agent::IDecider {
  public:
    /**
     * @brief Wires the model, the vocabulary and the two arenas.
     *
     * TWO arenas, and they must not be the same one. @p persistent holds the cache and
     * the working buffers for the whole session; @p scratch holds one step's grammar
     * and is RESET at the top of every decision. Resetting the persistent arena
     * mid-session would hand a live matrix's storage to the next grammar, which is
     * exactly the failure its own `reset()` documentation warns about.
     *
     * @param persistent Storage for the inference façade; never reset here.
     * @param scratch    Storage for per-step grammars; reset on every decision.
     * @param model      The weights.
     * @param vocab      The token table.
     * @param seed       Sampler stream seed.
     * @return false when the model is unbuilt or an arena is exhausted.
     */
    bool initialise(infer::TensorArena &persistent, infer::TensorArena &scratch, const infer::Model &model,
                    const infer::Vocab &vocab, core::u32 seed) noexcept;

    /**
     * @brief Binds the identity the fallback policy decides as.
     * @param persona Must outlive the reasoner.
     */
    void bind(const Persona &persona) noexcept { _fallback.bind(persona); }

    [[nodiscard]] agent::Act decide(const agent::DecisionContext &context) noexcept override;

    [[nodiscard]] const char *name() const noexcept override { return "model"; }

    /// Decisions that went to the model rather than to a rule.
    [[nodiscard]] core::u32 generations() const noexcept { return _generations; }

    /**
     * Actions emitted that the world did not offer.
     *
     * The number this class exists to keep at zero. It is counted rather than asserted
     * because a count can be folded into a gate and an assertion cannot — and because
     * an illegal action that got out would otherwise only be visible as a refusal in
     * the transcript, which looks exactly like a demon making a reasonable mistake.
     */
    [[nodiscard]] core::u32 illegalActions() const noexcept { return _illegalActions; }

    /// Steps where the grammar admitted nothing and the rule had to take over.
    [[nodiscard]] core::u32 grammarExhausted() const noexcept { return _grammarExhausted; }

    /// Steps that produced a whole phrase.
    [[nodiscard]] core::u32 completions() const noexcept { return _completions; }

    /// Tokens the model produced across the session.
    [[nodiscard]] core::u32 tokensGenerated() const noexcept { return _tokensGenerated; }

  private:
    infer::TensorArena *_scratch{nullptr};
    const infer::Vocab *_vocab{nullptr};
    infer::Inference _inference{};
    DeterministicReasoner _fallback{};
    core::u32 _seed{0u};
    core::u32 _generations{0u};
    core::u32 _illegalActions{0u};
    core::u32 _grammarExhausted{0u};
    core::u32 _completions{0u};
    core::u32 _tokensGenerated{0u};
    bool _ready{false};
};

} // namespace lpl::mind

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_MIND_REASONING_HPP
