/**
 * @file Budget.cpp
 * @brief Implementation of thinking within a deadline.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/mind/Budget.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/Fold.hpp>

namespace lpl::mind {

core::u32 foldBudget(const Budget &budget, core::u32 hash) noexcept
{
    foldWord(hash, budget.tokensSpent());
    foldWord(hash, budget.stepsSpent());
    foldWord(hash, budget.arenaSpent());
    /* The refusals are part of the signature, not a diagnostic beside it. A turn that
       finished and a turn that was cut short can produce the same transcript, and this
       is the only field that tells the two apart. */
    foldWord(hash, budget.denied());
    return hash;
}

} // namespace lpl::mind

#endif // LPL_HAS_FOUNDATION
