/**
 * @file Budget.hpp
 * @brief Thinking within a deadline.
 *
 * Tokens, steps and arena bytes, all bounded. The tick contract outranks the demon: it
 * may think as much as the frame allows and not one step more.
 *
 * Everything here is counted in WORK rather than in wall-clock time, and that is the
 * decision the file exists to make. A budget measured in milliseconds would make a
 * replay depend on how fast the machine was that day — reproducible on a quiet host and
 * not on a busy one, which is the same as not reproducible. Tokens and steps are the
 * same number everywhere.
 *
 * A refusal is COUNTED rather than silent. A demon that stopped early because it ran
 * out and one that stopped early because it was finished look identical from the
 * outside, and only the first is a reason to raise the budget.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_MIND_BUDGET_HPP
#    define LPL_LPL_MIND_BUDGET_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

namespace lpl::mind {

/**
 * @class Budget
 * @brief What one turn of thought is allowed to cost.
 */
class Budget {
  public:
    /**
     * @brief Opens a budget.
     *
     * @param tokens     Tokens the whole turn may generate.
     * @param steps      Reason-act-observe rounds it may take.
     * @param arenaBytes Bytes it may carve out of the arena.
     */
    constexpr Budget(core::u32 tokens, core::u32 steps, core::u32 arenaBytes) noexcept
        : _tokens(tokens), _steps(steps), _arenaBytes(arenaBytes)
    {
    }

    /**
     * @brief Claims tokens.
     * @param count How many.
     * @return true when they were within the budget.
     */
    constexpr bool claimTokens(core::u32 count) noexcept
    {
        if (_tokensSpent + count > _tokens)
        {
            ++_denied;
            return false;
        }
        _tokensSpent += count;
        return true;
    }

    /**
     * @brief Claims one reason-act-observe round.
     * @return true when it was within the budget.
     */
    constexpr bool claimStep() noexcept
    {
        if (_stepsSpent + 1u > _steps)
        {
            ++_denied;
            return false;
        }
        ++_stepsSpent;
        return true;
    }

    /**
     * @brief Claims arena bytes.
     * @param bytes How many.
     * @return true when they were within the budget.
     */
    constexpr bool claimArena(core::u32 bytes) noexcept
    {
        if (_arenaSpent + bytes > _arenaBytes)
        {
            ++_denied;
            return false;
        }
        _arenaSpent += bytes;
        return true;
    }

    /**
     * @brief Is the turn into its last tenth of tokens?
     *
     * Computed by multiplication and never by division, because a budget of nine
     * tokens divided by ten is zero — so a small budget would never report a final
     * stretch, and the demon would be cut off mid-sentence rather than winding up.
     *
     * @return true when nine tenths or more of the tokens are gone.
     */
    [[nodiscard]] constexpr bool finalStretch() const noexcept { return _tokensSpent * 10u >= _tokens * 9u; }

    /// Has anything run out?
    [[nodiscard]] constexpr bool exhausted() const noexcept
    {
        return _tokensSpent >= _tokens || _stepsSpent >= _steps || _arenaSpent >= _arenaBytes;
    }

    /// Tokens generated so far.
    [[nodiscard]] constexpr core::u32 tokensSpent() const noexcept { return _tokensSpent; }

    /// Rounds taken so far.
    [[nodiscard]] constexpr core::u32 stepsSpent() const noexcept { return _stepsSpent; }

    /// Arena bytes taken so far.
    [[nodiscard]] constexpr core::u32 arenaSpent() const noexcept { return _arenaSpent; }

    /// Claims refused because something had run out.
    [[nodiscard]] constexpr core::u32 denied() const noexcept { return _denied; }

  private:
    core::u32 _tokens;
    core::u32 _steps;
    core::u32 _arenaBytes;
    core::u32 _tokensSpent{0u};
    core::u32 _stepsSpent{0u};
    core::u32 _arenaSpent{0u};
    core::u32 _denied{0u};
};

/**
 * @brief Folds a budget's ledger into a signature.
 *
 * @param budget Budget to fold.
 * @param hash   Running value.
 * @return The updated hash.
 */
[[nodiscard]] core::u32 foldBudget(const Budget &budget, core::u32 hash) noexcept;

} // namespace lpl::mind

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_MIND_BUDGET_HPP
