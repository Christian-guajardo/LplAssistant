/**
 * @file Dialogue.hpp
 * @brief The channel to the master.
 *
 * Deliberately not a tool. Tools are how the demon acts on its world; dialogue is how
 * it addresses the one entity outside that world, whose next move is the only thing it
 * cannot compute.
 *
 * That distinction has a consequence this file exists to enforce: speaking is not in
 * the action alphabet, so a demon can never "use" the sovereign as a step in a plan.
 * It ends its turn and says something. A `say` verb sitting in @ref IActionSurface
 * would make the sovereign one more resource to spend, and a demon that has learned to
 * spend the sovereign is exactly the failure this project does not want to build.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_MIND_DIALOGUE_HPP
#    define LPL_LPL_MIND_DIALOGUE_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/mind/Budget.hpp>
#        include <lpl/mind/ReAct.hpp>

namespace lpl::mind {

/// Bytes one utterance to the sovereign may occupy.
inline constexpr core::u32 kUtteranceBytes = 160u;

/**
 * @enum Address
 * @brief How the demon is addressing the sovereign.
 */
enum class Address : core::u8 {
    Answer, ///< The thing that was asked for.
    Ask,    ///< It cannot proceed and needs a decision.
    Report, ///< It stopped for a reason of its own — usually the budget.
};

/**
 * @struct Utterance
 * @brief What the demon says when it stops working and starts speaking.
 */
struct Utterance {
    Address kind{Address::Report};  ///< Why it is speaking.
    char text[kUtteranceBytes]{};   ///< What it says.
    core::u32 bytes{0u};            ///< How many.
    core::u32 truncatedBytes{0u};   ///< Bytes the cap removed.
};

/**
 * @brief Turns a finished turn into one thing said to the sovereign.
 *
 * Reads the transcript rather than being told the outcome, so an utterance cannot
 * claim something the transcript does not show. A caller that could pass in the kind
 * directly would eventually pass in Answer for a turn that ran out of budget, and the
 * sovereign would be told a job was done that was not.
 *
 * @param transcript The turn.
 * @param count      Lines in it.
 * @param budget     What the turn cost.
 * @param persona    Who is speaking; brevity decides how much detail survives.
 * @return What to say.
 */
[[nodiscard]] Utterance concludeDialogue(const agent::Act *transcript, core::u32 count, const Budget &budget,
                                         const Persona &persona) noexcept;

/**
 * @brief Folds an utterance into a signature.
 *
 * @param utterance Utterance to fold.
 * @param hash      Running value.
 * @return The updated hash.
 */
[[nodiscard]] core::u32 foldUtterance(const Utterance &utterance, core::u32 hash) noexcept;

} // namespace lpl::mind

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_MIND_DIALOGUE_HPP
