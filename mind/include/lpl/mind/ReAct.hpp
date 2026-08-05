/**
 * @file ReAct.hpp
 * @brief Reason, act, observe, repeat.
 *
 * Each step regenerates the alphabet from the current world state, so the set of legal
 * actions is a function of the situation rather than a fixed menu.
 *
 * The two seams this loop turns on are NOT declared here. `agent::IDecider` (what the
 * demon decides) and `agent::IWorldSurface` (what the world allows) live in LplPlugin,
 * because the hosted demon and the ring-0 one must plug into the same ones — they were
 * briefly declared twice, once on each side, which is the duplication this project
 * spends most of its time undoing. What is left here is the LOOP, which is the only
 * part that is genuinely this module's.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_MIND_REACT_HPP
#    define LPL_LPL_MIND_REACT_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/agent/Decision.hpp>
#        include <lpl/mind/Budget.hpp>
#        include <lpl/mind/Intent.hpp>
#        include <lpl/mind/Persona.hpp>

namespace lpl::mind {

/**
 * @class DeterministicReasoner
 * @brief A real policy that needs no model.
 *
 * Not a placeholder for inference — a policy in its own right, and the one the parity
 * gate replays. It exists so the loop can be exercised, ordered and folded without a
 * model in the room; the same seam is where a model goes when there is one.
 *
 * The rules it follows are readable in one sentence each: a satisfied world is
 * answered; an intent the surface offers no untried action for is put back to the
 * sovereign when caution is high and abandoned when it is not; otherwise the first
 * available action not yet taken is taken.
 *
 * The persona is held rather than passed, because identity is a property of WHO is
 * deciding and not of the situation being decided about — and because the shared
 * context has to be statable by a hosted planner that has no persona at all.
 */
class DeterministicReasoner final : public agent::IDecider {
  public:
    DeterministicReasoner() = default;

    /**
     * @brief Binds the identity this policy decides as.
     * @param persona Must outlive the reasoner.
     */
    void bind(const Persona &persona) noexcept { _persona = &persona; }

    [[nodiscard]] agent::Act decide(const agent::DecisionContext &context) noexcept override;

    [[nodiscard]] const char *name() const noexcept override { return "deterministic"; }

  private:
    const Persona *_persona{nullptr};
};

/**
 * @brief Runs the loop until the world is satisfied or the budget is gone.
 *
 * A Stop is honoured HERE and never handed to a decider, which is a safety decision
 * rather than an optimisation: a stop that went through a sampler could be missed by
 * bad luck, and a demon that ignores a stop once in a hundred turns is worse than one
 * that cannot think at all.
 *
 * @param intent     What was asked.
 * @param surface    The world.
 * @param decider    The policy.
 * @param budget     What the turn may cost; charged as it goes.
 * @param transcript Receives the lines.
 * @param capacity   Room in @p transcript.
 * @return Lines written.
 */
core::u32 runReAct(const Intent &intent, agent::IWorldSurface &surface, agent::IDecider &decider, Budget &budget,
                   agent::Act *transcript, core::u32 capacity) noexcept;

/**
 * @brief Folds a transcript into a signature.
 *
 * @param transcript Lines to fold.
 * @param count      How many.
 * @param hash       Running value.
 * @return The updated hash.
 */
[[nodiscard]] core::u32 foldTranscript(const agent::Act *transcript, core::u32 count, core::u32 hash) noexcept;

} // namespace lpl::mind

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_MIND_REACT_HPP
