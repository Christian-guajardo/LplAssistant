/**
 * @file ReAct.cpp
 * @brief Implementation of reason, act, observe, repeat.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/mind/ReAct.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/Fold.hpp>

namespace lpl::mind {

namespace {

/**
 * @brief Fills one act's text, truncating at its capacity.
 *
 * @param act   Act to fill.
 * @param text  Bytes to store; may be null when @p count is zero.
 * @param count How many.
 */
void setActText(agent::Act &act, const char *text, core::u32 count) noexcept
{
    act.bytes = count < agent::kActBytes ? count : agent::kActBytes;
    for (core::u32 i = 0u; i < act.bytes; ++i)
        act.text[i] = text[i];
}

/**
 * @brief Extracts one space-separated word from an alphabet.
 *
 * @param alphabet Space-separated names.
 * @param bytes    Its length.
 * @param wanted   Which word, counting from zero.
 * @param outStart Receives where it begins.
 * @param outEnd   Receives where it ends.
 * @return true when there is a word at that position.
 */
bool nthWord(const char *alphabet, core::u32 bytes, core::u32 wanted, core::u32 &outStart, core::u32 &outEnd) noexcept
{
    core::u32 index = 0u;
    core::u32 cursor = 0u;
    while (cursor < bytes)
    {
        while (cursor < bytes && alphabet[cursor] == ' ')
            ++cursor;
        const core::u32 start = cursor;
        while (cursor < bytes && alphabet[cursor] != ' ')
            ++cursor;
        if (cursor == start)
            break;
        if (index == wanted)
        {
            outStart = start;
            outEnd = cursor;
            return true;
        }
        ++index;
    }
    return false;
}

/**
 * @brief Has this act already been taken in the transcript?
 *
 * The guard that stops the loop repeating one move until the budget is gone. Without
 * it a surface that reports an action available but ineffective produces a transcript
 * of the same line over and over, which costs the whole budget and says nothing.
 *
 * @param transcript Lines so far.
 * @param count      How many.
 * @param text       Action bytes.
 * @param bytes      How many.
 * @return true when it appears as an Action already.
 */
bool alreadyTried(const agent::Act *transcript, core::u32 count, const char *text, core::u32 bytes) noexcept
{
    for (core::u32 i = 0u; i < count; ++i)
    {
        if (transcript[i].kind != agent::ActKind::Action || transcript[i].bytes != bytes)
            continue;
        bool match = true;
        for (core::u32 b = 0u; b < bytes && match; ++b)
            match = transcript[i].text[b] == text[b];
        if (match)
            return true;
    }
    return false;
}

} // namespace

agent::Act DeterministicReasoner::decide(const agent::DecisionContext &context) noexcept
{
    agent::Act act;
    act.step = context.turn;

    /* The job is done, so say so. Tested BEFORE the search for an untried action,
       because a satisfied world still offers actions — the demon could keep fiddling
       with the valve forever — and a loop that only stops when it runs out of moves is
       not finishing, it is exhausting itself. */
    if (context.satisfied)
    {
        act.kind = agent::ActKind::Answer;
        core::u32 lastObservation = context.transcriptLines;
        for (core::u32 i = context.transcriptLines; i > 0u; --i)
        {
            if (context.transcript[i - 1u].kind == agent::ActKind::Observation)
            {
                lastObservation = i - 1u;
                break;
            }
        }
        if (lastObservation < context.transcriptLines)
            setActText(act, context.transcript[lastObservation].text, context.transcript[lastObservation].bytes);
        else
            setActText(act, "already done", 12u);
        return act;
    }

    // The first available action that has not been tried yet.
    core::u32 candidate = 0u;
    core::u32 start = 0u;
    core::u32 end = 0u;
    while (nthWord(context.available, context.availableBytes, candidate, start, end))
    {
        if (!alreadyTried(context.transcript, context.transcriptLines, context.available + start, end - start))
        {
            act.kind = agent::ActKind::Action;
            setActText(act, context.available + start, end - start);
            return act;
        }
        ++candidate;
    }

    /* Nothing left to try. What happens now is the persona's to decide, and it is the
       one place a trait changes the transcript rather than its tone: a cautious demon
       hands the problem back, a bold one reports what it managed. Both are defensible,
       which is exactly why it is written as data the sovereign can change. */
    const bool cautious = _persona != nullptr && _persona->caution > math::Fixed32::half();
    if (cautious)
    {
        act.kind = agent::ActKind::Question;
        setActText(act, "no action left, how should I proceed", 36u);
    }
    else
    {
        act.kind = agent::ActKind::Answer;
        setActText(act, "done what was possible", 22u);
    }
    return act;
}

core::u32 runReAct(const Intent &intent, agent::IWorldSurface &surface, agent::IDecider &decider, Budget &budget,
                   agent::Act *transcript, core::u32 capacity) noexcept
{
    if (transcript == nullptr || capacity == 0u)
        return 0u;

    /* A Stop outranks everything, including a half-finished job, and it is honoured
       here rather than inside a decider. That is what makes it a stop rather than a
       suggestion: a model asked to respect one could fail to, and the failure would be
       intermittent. */
    if (intent.kind == IntentKind::Stop)
    {
        transcript[0] = agent::Act{};
        transcript[0].kind = agent::ActKind::Answer;
        setActText(transcript[0], "stopped", 7u);
        (void) budget.claimStep();
        (void) budget.claimTokens(7u);
        return 1u;
    }

    core::u32 written = 0u;
    char available[agent::kAvailableBytes]{};

    while (written < capacity && budget.claimStep())
    {
        // Regenerated every round, from the world as it is now. This call is the
        // difference between a grammar and a menu.
        agent::DecisionContext context;
        context.availableBytes = surface.available(available, agent::kAvailableBytes);
        context.available = available;
        context.transcript = transcript;
        context.transcriptLines = written;
        context.goal = intent.text;
        context.goalBytes = intent.bytes;
        context.turn = written;
        context.satisfied = surface.satisfied();

        const agent::Act decided = decider.decide(context);

        /* One token charged per byte of the decision. Crude on purpose: this module
           does not own a tokeniser, and inventing one here would put a second answer
           to "how long is this text" next to the real one in infer/. What matters for
           the budget is that thinking costs something monotonic in what it produced. */
        if (!budget.claimTokens(decided.bytes))
            break;

        transcript[written++] = decided;

        if (decided.kind != agent::ActKind::Action)
            break; // The demon addressed the sovereign; the turn is over.

        if (written >= capacity)
            break;

        agent::Act observation;
        observation.kind = agent::ActKind::Observation;
        observation.step = decided.step;
        core::u32 observed = 0u;
        if (!surface.perform(decided.text, decided.bytes, observation.text, agent::kActBytes, &observed))
        {
            /* A refused action is recorded as an observation rather than dropped. The
               demon has to be able to see that it tried something illegal, or it will
               keep trying it — and a transcript with a hole where a mistake was is
               worse than one that shows the mistake. */
            const char refused[] = "refused";
            for (core::u32 i = 0u; i < 7u; ++i)
                observation.text[i] = refused[i];
            observed = 7u;
        }
        observation.bytes = observed < agent::kActBytes ? observed : agent::kActBytes;
        transcript[written++] = observation;
    }
    return written;
}

core::u32 foldTranscript(const agent::Act *transcript, core::u32 count, core::u32 hash) noexcept
{
    foldWord(hash, count);
    for (core::u32 i = 0u; i < count; ++i)
    {
        foldWord(hash, static_cast<core::u32>(transcript[i].kind));
        foldWord(hash, transcript[i].step);
        foldWord(hash, transcript[i].bytes);
        foldBytes(hash, reinterpret_cast<const core::u8 *>(transcript[i].text), transcript[i].bytes);
    }
    return hash;
}

} // namespace lpl::mind

#endif // LPL_HAS_FOUNDATION
