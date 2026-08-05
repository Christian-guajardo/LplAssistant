/**
 * @file Dialogue.cpp
 * @brief Implementation of the channel to the master.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/mind/Dialogue.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/Fold.hpp>

namespace lpl::mind {

namespace {

/**
 * @brief Appends bytes to an utterance, counting whatever will not fit.
 *
 * @param utterance Utterance to extend.
 * @param text      Bytes to append.
 * @param count     How many.
 */
void appendUtterance(Utterance &utterance, const char *text, core::u32 count) noexcept
{
    for (core::u32 i = 0u; i < count; ++i)
    {
        if (utterance.bytes >= kUtteranceBytes)
        {
            ++utterance.truncatedBytes;
            continue;
        }
        utterance.text[utterance.bytes++] = text[i];
    }
}

} // namespace

Utterance concludeDialogue(const agent::Act *transcript, core::u32 count, const Budget &budget,
                           const Persona &persona) noexcept
{
    Utterance utterance;

    /* The last line the demon addressed to the sovereign decides the kind, and the
       transcript is read backwards to find it. Reading forwards would find the first
       thing said rather than the last, and in a turn that asked a question, tried
       something else and then answered, the first is no longer true. */
    core::u32 spoke = count;
    while (spoke > 0u)
    {
        --spoke;
        if (transcript[spoke].kind == agent::ActKind::Answer || transcript[spoke].kind == agent::ActKind::Question)
            break;
    }

    const bool addressed =
        count > 0u && (transcript[spoke].kind == agent::ActKind::Answer || transcript[spoke].kind == agent::ActKind::Question);

    if (!addressed)
    {
        /* The turn ended without the demon saying anything — it ran out. That is a
           Report and never an Answer: a budget that expired mid-plan has produced no
           conclusion, and dressing it as one is how a sovereign comes to believe a job
           is finished. */
        utterance.kind = Address::Report;
        appendUtterance(utterance, "out of budget", 13u);
    }
    else if (transcript[spoke].kind == agent::ActKind::Question)
    {
        utterance.kind = Address::Ask;
        appendUtterance(utterance, transcript[spoke].text, transcript[spoke].bytes);
    }
    else
    {
        utterance.kind = Address::Answer;
        appendUtterance(utterance, transcript[spoke].text, transcript[spoke].bytes);
    }

    /* Brevity decides whether the working is shown, not whether the conclusion is.
       A terse demon still answers; it just does not narrate how many steps it took. */
    if (!(persona.brevity > math::Fixed32::half()))
    {
        appendUtterance(utterance, " (", 2u);
        const core::u32 steps = budget.stepsSpent();
        char digits[10]{};
        core::u32 length = 0u;
        core::u32 value = steps;
        do
        {
            digits[length++] = static_cast<char>('0' + (value % 10u));
            value /= 10u;
        } while (value != 0u && length < 10u);
        for (core::u32 i = 0u; i < length; ++i)
            appendUtterance(utterance, &digits[length - 1u - i], 1u);
        appendUtterance(utterance, " steps)", 7u);
    }

    return utterance;
}

core::u32 foldUtterance(const Utterance &utterance, core::u32 hash) noexcept
{
    foldWord(hash, static_cast<core::u32>(utterance.kind));
    foldWord(hash, utterance.bytes);
    foldBytes(hash, reinterpret_cast<const core::u8 *>(utterance.text), utterance.bytes);
    foldWord(hash, utterance.truncatedBytes);
    return hash;
}

} // namespace lpl::mind

#endif // LPL_HAS_FOUNDATION
