/**
 * @file Parity.hpp
 * @brief The constexpr session both sides replay.
 *
 * Same intent, same world, same transcript.
 *
 * What is folded is a whole TURN and not a set of functions: a persona is loaded, an
 * utterance from the sovereign is parsed, notes are filed until the store has to start
 * choosing, a lookup narrows them, the loop runs against a small world, and the demon
 * says one thing at the end. Every decision the agency layer makes appears somewhere in
 * those signatures — which of two notes survived, which action was reached for first,
 * whether the turn ended in an answer or a question, and what it cost.
 *
 * The world is here rather than injected, and that is what makes this a gate. A demon
 * driven by a real model against a real engine cannot be replayed, so the parity case
 * uses @ref DeterministicReasoner against @ref ParityWorld — both real policies, neither
 * a stub. The seam that a model plugs into is exercised by the same run.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_MIND_PARITY_HPP
#    define LPL_LPL_MIND_PARITY_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/agent/Decision.hpp>
#        include <lpl/mind/Dialogue.hpp>
#        include <lpl/mind/Memory.hpp>
#        include <lpl/mind/Persona.hpp>
#        include <lpl/mind/ReAct.hpp>
#        include <lpl/mind/Recall.hpp>

namespace lpl::mind {

/// Lines the canonical transcript may reach.
inline constexpr core::u32 kParityTranscriptCapacity = 16u;

/// Hits the canonical lookup may return.
inline constexpr core::u32 kParityRecallCapacity = 4u;

/**
 * @brief Notes the canonical session files.
 *
 * Two more than the store holds, on purpose: a session that never fills its memory
 * never exercises the eviction rule, and the eviction rule is the only part of
 * @ref MemoryStore where two implementations could plausibly disagree.
 */
[[nodiscard]] constexpr core::u32 parityNoteCount() noexcept { return kMaxNotes + 2u; }

/// Tokens the canonical turn is allowed.
[[nodiscard]] constexpr core::u32 parityTokenBudget() noexcept { return 512u; }

/// Rounds the canonical turn is allowed.
[[nodiscard]] constexpr core::u32 parityStepBudget() noexcept { return 8u; }

/// Arena bytes the canonical turn is allowed.
[[nodiscard]] constexpr core::u32 parityArenaBudget() noexcept { return 4096u; }

/**
 * @class ParityWorld
 * @brief The small world the canonical turn acts on.
 *
 * Four verbs and one goal. Small enough to reason about by hand, and large enough that
 * the action alphabet CHANGES as the turn proceeds — which is the property being
 * tested. A world with a fixed alphabet would let a broken "regenerate the grammar
 * every step" pass, because a menu and a grammar look identical when nothing moves.
 */
class ParityWorld final : public agent::IWorldSurface {
  public:
    core::u32 available(char *out, core::u32 capacity) noexcept override;

    bool perform(const char *action, core::u32 actionBytes, char *report, core::u32 capacity,
                 core::u32 *reportBytes) noexcept override;

    [[nodiscard]] bool satisfied() const noexcept override { return _logged; }

    /// Has the valve been opened?
    [[nodiscard]] bool open() const noexcept { return _open; }

    /// Has the sensor been read?
    [[nodiscard]] bool read() const noexcept { return _read; }

    /// Actions the world refused.
    [[nodiscard]] core::u32 refusals() const noexcept { return _refusals; }

  private:
    bool _read{false};
    bool _open{false};
    bool _logged{false};
    core::u32 _refusals{0u};
};

/**
 * @brief Builds the persona the canonical session runs as.
 * @return The persona.
 */
[[nodiscard]] Persona parityPersona() noexcept;

/**
 * @struct AgencyFoldResult
 * @brief The signatures the kernel must reproduce.
 *
 * Plain words only, no Fixed32 and no bool, like every other fold result in the project:
 * every field is a word a test checks or records, and the kernel's records are compared with
 * the host's.
 */
struct AgencyFoldResult {
    core::u32 personaSignature{0u};    ///< Fold of who was thinking.
    core::u32 intentSignature{0u};     ///< Fold of the parsed utterance.
    core::u32 memorySignature{0u};     ///< Fold of the store after it had to choose.
    core::u32 recallSignature{0u};     ///< Fold of what the lookup surfaced.
    core::u32 transcriptSignature{0u}; ///< Fold of every line of the turn.
    core::u32 utteranceSignature{0u};  ///< Fold of what was said to the sovereign.
    core::u32 budgetSignature{0u};     ///< Fold of what the turn cost.
    core::u32 intentKind{0u};          ///< How the utterance was classified.
    core::u32 droppedBytes{0u};        ///< Bytes the parser refused.
    core::u32 notesHeld{0u};           ///< Notes left in the store.
    core::u32 evictions{0u};           ///< Notes displaced.
    core::u32 refusals{0u};            ///< Notes turned away as too trivial.
    core::u32 recallHits{0u};          ///< Notes the lookup returned.
    core::u32 transcriptLines{0u};     ///< Lines the turn produced.
    core::u32 stepsSpent{0u};          ///< Rounds it took.
    core::u32 tokensSpent{0u};         ///< Tokens it cost.
    core::u32 utteranceKind{0u};       ///< Answer, Ask or Report.
    core::u32 worldSatisfied{0u};      ///< 1 when the world's goal was met.
    core::u32 worldRefusals{0u};       ///< Actions the world turned down.
};

/**
 * @brief Runs the canonical turn and folds every stage of it.
 *
 * @param out Receives the signatures.
 */
void foldAgency(AgencyFoldResult &out) noexcept;

/**
 * @struct ReasoningFoldResult
 * @brief Gate P17 — the same turn, decided by a model instead of by a rule.
 *
 * Two numbers here carry the claim and they have to be read together.
 * @ref illegalActions is what the grammar makes impossible, and it must be zero.
 * @ref freeLegalNames is the control: the SAME model, the same prompts, generating
 * without the constraint, and how often it happens to name something the world would
 * accept. A zero in the first column means nothing on its own — a demon that never
 * acts also never acts illegally — so the second column is what turns it into a
 * measurement rather than a reassurance.
 */
struct ReasoningFoldResult {
    core::u32 transcriptSignature{0u}; ///< Fold of the model-driven turn.
    core::u32 actionSignature{0u};     ///< Fold of the chosen actions alone.
    core::u32 utteranceSignature{0u};  ///< Fold of what was said at the end.
    core::u32 generations{0u};         ///< Decisions that went to the model.
    core::u32 completions{0u};         ///< Decisions that spelled a whole phrase.
    core::u32 illegalActions{0u};      ///< Actions the world did not offer. MUST be zero.
    core::u32 grammarExhausted{0u};    ///< Steps where the language ran out.
    core::u32 tokensGenerated{0u};     ///< Tokens the model produced.
    core::u32 transcriptLines{0u};     ///< Lines the turn produced.
    core::u32 stepsSpent{0u};          ///< Rounds it took.
    core::u32 satisfied{0u};           ///< 1 when the world's goal was met.
    core::u32 freeAttempts{0u};        ///< Unconstrained generations run as a control.
    core::u32 freeLegalNames{0u};      ///< How many of those named a legal action.
    core::u32 arenaBytes{0u};          ///< Bytes the run carved out; a fit check, not an invariant.
};

/**
 * @brief Runs the canonical turn with a real model behind the decisions.
 *
 * @param out    Receives the signatures.
 * @param memory Arena block; when null, one is claimed from the host allocator.
 * @param bytes  Size of @p memory.
 */
void foldReasoning(ReasoningFoldResult &out, void *memory = nullptr, core::usize bytes = 0u) noexcept;

/**
 * @brief Bytes the reasoning turn wants.
 *
 * Larger than the mind gate's because this one holds the weights, the cache, and a
 * scratch region reset at every step — three lifetimes that must not share an arena.
 *
 * @return The capacity.
 */
[[nodiscard]] constexpr core::usize parityReasoningArenaBytes() noexcept { return 512u * 1024u; }

/// Bytes of that arena reserved for per-step grammars.
[[nodiscard]] constexpr core::usize parityReasoningScratchBytes() noexcept { return 16u * 1024u; }

} // namespace lpl::mind

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_MIND_PARITY_HPP
