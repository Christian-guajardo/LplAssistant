/**
 * @file Reasoning.cpp
 * @brief Implementation of the reasoner that actually runs a model.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/mind/Reasoning.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/math/Random.hpp>

namespace lpl::mind {

namespace {

/// Tokens one decision may produce. Bounded by the window, not by taste.
constexpr core::u32 kMaxDecisionTokens = 16u;

/**
 * @brief Do these bytes spell exactly this phrase?
 * @param left      Candidate.
 * @param leftBytes Its length.
 * @param right     The phrase.
 * @param rightBytes Its length.
 * @return true when they match byte for byte.
 */
constexpr bool same(const char *left, core::u32 leftBytes, const char *right, core::u32 rightBytes) noexcept
{
    if (leftBytes != rightBytes)
        return false;
    for (core::u32 i = 0u; i < leftBytes; ++i)
        if (left[i] != right[i])
            return false;
    return true;
}

/**
 * @brief Has this action already been taken?
 * @param transcript Lines so far.
 * @param count      How many.
 * @param text       Action bytes.
 * @param bytes      How many.
 * @return true when it appears as an Action already.
 */
bool alreadyTaken(const agent::Act *transcript, core::u32 count, const char *text, core::u32 bytes) noexcept
{
    for (core::u32 i = 0u; i < count; ++i)
        if (transcript[i].kind == agent::ActKind::Action &&
            same(transcript[i].text, transcript[i].bytes, text, bytes))
            return true;
    return false;
}

} // namespace

bool ModelReasoner::initialise(infer::TensorArena &persistent, infer::TensorArena &scratch, const infer::Model &model,
                               const infer::Vocab &vocab, core::u32 seed) noexcept
{
    _scratch = &scratch;
    _vocab = &vocab;
    _seed = seed;
    _generations = 0u;
    _illegalActions = 0u;
    _grammarExhausted = 0u;
    _completions = 0u;
    _tokensGenerated = 0u;

    /* Refuse rather than slide when the window is full. A demon whose context quietly
       dropped its oldest positions would make one decision from a prompt and the next
       from a truncation of it, and nothing in the transcript would say so. */
    _ready = _inference.initialise(persistent, model, infer::CachePolicy::Refuse);
    return _ready;
}

agent::Act ModelReasoner::decide(const agent::DecisionContext &context) noexcept
{
    /* A satisfied world is concluded by RULE and never by generation. Same reasoning
       as the Stop the loop handles above it: a conclusion that went through a sampler
       could be missed by bad luck, and a demon that failed to notice it had finished
       would carry on spending a budget it no longer needs. */
    if (!_ready || context.satisfied)
        return _fallback.decide(context);

    _scratch->reset();

    /* The step's language: every available action that has NOT been taken yet.
       Building it from the alphabet the world just produced is what makes the
       constraint a function of the situation; dropping what was already tried is what
       stops the model re-proposing its last move until the budget is gone. */
    const char *phrases[kMaxLegalPhrases]{};
    char *const blob = _scratch->claim<char>(kMaxLegalPhrases * (agent::kActBytes + 1u));
    if (blob == nullptr)
        return _fallback.decide(context);

    core::u32 phraseCount = 0u;
    core::u32 cursor = 0u;
    while (cursor < context.availableBytes && phraseCount < kMaxLegalPhrases)
    {
        while (cursor < context.availableBytes && context.available[cursor] == ' ')
            ++cursor;
        const core::u32 start = cursor;
        while (cursor < context.availableBytes && context.available[cursor] != ' ')
            ++cursor;
        const core::u32 length = cursor - start;
        if (length == 0u || length > agent::kActBytes)
            continue;
        if (alreadyTaken(context.transcript, context.transcriptLines, context.available + start, length))
            continue;

        char *const slot = blob + phraseCount * (agent::kActBytes + 1u);
        for (core::u32 i = 0u; i < length; ++i)
            slot[i] = context.available[start + i];
        slot[length] = '\0';
        phrases[phraseCount++] = slot;
    }

    // Nothing left the world will accept. That is the rule's question, not the model's.
    if (phraseCount == 0u)
        return _fallback.decide(context);

    infer::Grammar grammar;
    if (!grammar.build(*_scratch, phrases, phraseCount))
        return _fallback.decide(context);

    /* The prompt ties the run to what was asked. It cannot make an untrained model
       competent — nothing can — so what it buys is that two different goals do not
       decide from the same context. Short because the window is thirty-two positions
       and the answer has to fit in it beside the question. */
    core::u32 promptTokens[kReasoningPromptBytes]{};
    core::u32 skipped = 0u;
    const infer::Tokenizer tokenizer{*_vocab};
    const core::u32 promptBytes = context.goalBytes < kReasoningPromptBytes ? context.goalBytes : kReasoningPromptBytes;
    const core::u32 promptCount =
        tokenizer.encode(context.goal, promptBytes, promptTokens, kReasoningPromptBytes, skipped);

    infer::GenerationParams params;
    params.sampler.topK = 8u;
    /* A stream derived from the step, never the step added to a seed. Consecutive
       seeds do not avalanche through three shift-xors, which this project has already
       paid for once: a fountain encoder walked its seeds one by one and produced a
       degree distribution narrow enough to look like a tuning problem. */
    params.sampler.seed = math::deriveStream(_seed, context.turn).state();
    params.maxTokens = kMaxDecisionTokens;

    core::u32 produced[kMaxDecisionTokens]{};
    infer::GenerationReport report{};
    ++_generations;
    if (!_inference.generate(promptTokens, promptCount, params, &grammar, produced, kMaxDecisionTokens, report))
        return _fallback.decide(context);

    _tokensGenerated += report.generated;
    if (report.grammarExhausted)
        ++_grammarExhausted;
    if (!report.grammarComplete)
        return _fallback.decide(context);
    ++_completions;

    agent::Act act;
    act.kind = agent::ActKind::Action;
    act.step = context.turn;
    act.bytes = tokenizer.decode(produced, report.generated, act.text, agent::kActBytes);

    /* Checked against the language it was generated from. This should be impossible to
       fail, which is exactly why it is measured: a grammar that had stopped
       constraining would otherwise show up as a demon making plausible mistakes, and
       plausible mistakes are the hardest kind to notice. */
    bool legalChoice = false;
    for (core::u32 i = 0u; i < phraseCount && !legalChoice; ++i)
    {
        core::u32 length = 0u;
        while (phrases[i][length] != '\0')
            ++length;
        legalChoice = same(act.text, act.bytes, phrases[i], length);
    }
    if (!legalChoice)
    {
        ++_illegalActions;
        return _fallback.decide(context);
    }
    return act;
}

} // namespace lpl::mind

#endif // LPL_HAS_FOUNDATION
