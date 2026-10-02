/**
 * @file Parity.cpp
 * @brief Implementation of the constexpr session both sides replay.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/mind/Parity.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/Fold.hpp>
#    include <lpl/infer/Parity.hpp>
#    include <lpl/mind/Intent.hpp>
#    include <lpl/mind/Reasoning.hpp>

namespace lpl::mind {

namespace {

/**
 * @brief Length of a null-terminated literal.
 * @param text Literal.
 * @return Bytes before the terminator.
 */
constexpr core::u32 literalBytes(const char *text) noexcept
{
    core::u32 count = 0u;
    while (text[count] != '\0')
        ++count;
    return count;
}

/**
 * @brief Do these bytes spell this literal?
 * @param bytes Candidate.
 * @param count Its length.
 * @param text  Null-terminated literal.
 * @return true when they match exactly.
 */
constexpr bool spells(const char *bytes, core::u32 count, const char *text) noexcept
{
    if (count != literalBytes(text))
        return false;
    for (core::u32 i = 0u; i < count; ++i)
        if (bytes[i] != text[i])
            return false;
    return true;
}

/**
 * @brief Copies a literal into a caller's buffer.
 * @param out      Destination.
 * @param capacity Room in it.
 * @param text     Null-terminated literal.
 * @return Bytes written.
 */
core::u32 emit(char *out, core::u32 capacity, const char *text) noexcept
{
    core::u32 written = 0u;
    while (text[written] != '\0' && written < capacity)
    {
        out[written] = text[written];
        ++written;
    }
    return written;
}

/// The canonical utterance from the sovereign, with two bytes that must not survive.
constexpr char kParityUtterance[] = "what is the reactor pressure\x01\x02";

} // namespace

core::u32 ParityWorld::available(char *out, core::u32 capacity) noexcept
{
    /* The alphabet shrinks and grows as the turn proceeds, which is the whole reason
       this world exists: opening a valve that is already open stops being spellable,
       and logging only becomes spellable once there is a reading to log.

       Note which state gates which verb. Logging depends on having READ, not on the
       valve — the first version tied it to the valve being open, so closing the valve
       withdrew the goal and the turn deadlocked between opening and closing something
       it had already done both of. A verb has to be gated on the fact it actually
       needs, or the world is unwinnable for a reason that looks like a broken loop. */
    if (!_read)
        return emit(out, capacity, "read_sensor");
    if (!_open)
        return emit(out, capacity, "open_valve log_reading");
    return emit(out, capacity, "close_valve log_reading");
}

bool ParityWorld::perform(const char *action, core::u32 actionBytes, char *observation, core::u32 capacity,
                          core::u32 *observationBytes) noexcept
{
    if (observationBytes != nullptr)
        *observationBytes = 0u;

    if (spells(action, actionBytes, "read_sensor") && !_read)
    {
        _read = true;
        const core::u32 written = emit(observation, capacity, "pressure 41 kilopascal");
        if (observationBytes != nullptr)
            *observationBytes = written;
        return true;
    }
    if (spells(action, actionBytes, "open_valve") && _read && !_open)
    {
        _open = true;
        const core::u32 written = emit(observation, capacity, "valve open");
        if (observationBytes != nullptr)
            *observationBytes = written;
        return true;
    }
    if (spells(action, actionBytes, "close_valve") && _open)
    {
        _open = false;
        const core::u32 written = emit(observation, capacity, "valve closed");
        if (observationBytes != nullptr)
            *observationBytes = written;
        return true;
    }
    if (spells(action, actionBytes, "log_reading") && _read)
    {
        _logged = true;
        const core::u32 written = emit(observation, capacity, "logged");
        if (observationBytes != nullptr)
            *observationBytes = written;
        return true;
    }

    ++_refusals;
    return false;
}

Persona parityPersona() noexcept
{
    Persona persona;
    personaSetName(persona, "Caine", 5u);
    personaAddDirective(persona, "answer from what was measured", 29u);
    personaAddDirective(persona, "never guess a number", 20u);

    /* Caution above a half, so a turn with nothing left to try ends in a QUESTION
       rather than a claim. That choice is what the gate pins: a persona edited to be
       bolder produces a different transcript from the same world, which is the point
       of keeping identity as data. */
    persona.caution = math::Fixed32::fromFloat(0.75f);
    persona.brevity = math::Fixed32::fromFloat(0.25f);
    persona.initiative = math::Fixed32::fromFloat(0.50f);
    return persona;
}

void foldAgency(AgencyFoldResult &out) noexcept
{
    out = AgencyFoldResult{};

    const Persona persona = parityPersona();
    out.personaSignature = foldPersona(persona, kFnv1aOffsetBasis);

    const Intent intent =
        parseIntent(reinterpret_cast<const core::u8 *>(kParityUtterance), literalBytes(kParityUtterance));
    out.intentSignature = foldIntent(intent, kFnv1aOffsetBasis);
    out.intentKind = static_cast<core::u32>(intent.kind);
    out.droppedBytes = intent.droppedBytes;

    /* File more notes than the store holds, with salience that rises and then falls,
       so both halves of the eviction rule are exercised: notes that displace weaker
       ones, and notes that are refused because everything held matters more. */
    static MemoryStore store;
    store = MemoryStore{};
    for (core::u32 i = 0u; i < parityNoteCount(); ++i)
    {
        MemoryNote note;
        const char stem[] = "reactor pressure sample ";
        note.bytes = 0u;
        for (core::u32 b = 0u; b < literalBytes(stem); ++b)
            note.text[note.bytes++] = stem[b];
        note.text[note.bytes++] = static_cast<char>('a' + static_cast<char>(i % 26u));
        note.topic = topicOf("reactor", 7u);
        note.tick = i;

        /* The store fills with rising salience, then takes exactly two more notes: one
           above everything held, and one below everything held. That is the smallest
           sequence that exercises BOTH halves of the eviction rule — a displacement
           and a refusal.

           A ramp that merely rose and fell does not do it, which is what the first
           version got wrong: the least salient notes are the ones filed first, so they
           are also the ones evicted, and every later note beats whatever is left. The
           refusal path never ran, and the gate reported a rule it had tested half of. */
        const core::u32 rung = i < kMaxNotes ? (i + 1u) : (i == kMaxNotes ? parityNoteCount() : 0u);
        note.salience = math::Fixed32::fromRaw(static_cast<core::i32>((rung << 16) / parityNoteCount()));
        store.remember(note);
    }
    out.memorySignature = foldMemory(store, kFnv1aOffsetBasis);
    out.notesHeld = store.count();
    out.evictions = store.evictions();
    out.refusals = store.refusals();

    RecallHit hits[kParityRecallCapacity]{};
    out.recallHits = recall(store, intent.topic, intent.text, intent.bytes, math::Fixed32::zero(), hits,
                            kParityRecallCapacity);
    out.recallSignature = foldRecall(hits, out.recallHits, kFnv1aOffsetBasis);

    ParityWorld world;
    DeterministicReasoner reasoner;
    Budget budget{parityTokenBudget(), parityStepBudget(), parityArenaBudget()};
    static agent::Act transcript[kParityTranscriptCapacity];
    for (core::u32 i = 0u; i < kParityTranscriptCapacity; ++i)
        transcript[i] = agent::Act{};

    reasoner.bind(persona);
    out.transcriptLines = runReAct(intent, world, reasoner, budget, transcript, kParityTranscriptCapacity);
    out.transcriptSignature = foldTranscript(transcript, out.transcriptLines, kFnv1aOffsetBasis);

    const Utterance utterance = concludeDialogue(transcript, out.transcriptLines, budget, persona);
    out.utteranceSignature = foldUtterance(utterance, kFnv1aOffsetBasis);
    out.utteranceKind = static_cast<core::u32>(utterance.kind);

    out.budgetSignature = foldBudget(budget, kFnv1aOffsetBasis);
    out.stepsSpent = budget.stepsSpent();
    out.tokensSpent = budget.tokensSpent();
    out.worldSatisfied = world.satisfied() ? 1u : 0u;
    out.worldRefusals = world.refusals();
}


void foldReasoning(ReasoningFoldResult &out, void *memory, core::usize bytes) noexcept
{
    out = ReasoningFoldResult{};

    infer::TensorArena arena = memory != nullptr ? infer::TensorArena{memory, bytes}
                                                 : infer::TensorArena{parityReasoningArenaBytes()};

    infer::Vocab vocab;
    if (!infer::buildParityVocab(arena, vocab))
        return;

    infer::ModelConfig shape = infer::parityModelConfig();
    shape.vocabSize = vocab.size();

    infer::Model model;
    if (!model.synthesise(arena, shape, vocab, infer::parityModelSeed()))
        return;

    /* The scratch region is carved out of the same block but handed over as its OWN
       arena, so resetting it every step cannot reach anything the inference façade is
       still using. One allocator with two lifetimes in it would be the bug its own
       reset() documentation warns about. */
    core::u8 *const scratchBlock = arena.claim<core::u8>(parityReasoningScratchBytes());
    if (scratchBlock == nullptr)
        return;
    infer::TensorArena scratch{scratchBlock, parityReasoningScratchBytes()};

    ModelReasoner reasoner;
    if (!reasoner.initialise(arena, scratch, model, vocab, infer::paritySamplerSeed()))
        return;

    const Persona persona = parityPersona();
    const Intent intent =
        parseIntent(reinterpret_cast<const core::u8 *>(kParityUtterance), literalBytes(kParityUtterance));

    ParityWorld world;
    Budget budget{parityTokenBudget(), parityStepBudget(), parityArenaBudget()};
    static agent::Act transcript[kParityTranscriptCapacity];
    for (core::u32 i = 0u; i < kParityTranscriptCapacity; ++i)
        transcript[i] = agent::Act{};

    reasoner.bind(persona);
    out.transcriptLines = runReAct(intent, world, reasoner, budget, transcript, kParityTranscriptCapacity);
    out.transcriptSignature = foldTranscript(transcript, out.transcriptLines, kFnv1aOffsetBasis);

    /* The actions alone, folded separately from the whole transcript. An observation
       is the world talking; only these lines are the model's decisions, and a change
       in what the demon CHOSE should be visible without reading it out of a signature
       that also moves when the world's wording changes. */
    core::u32 actionHash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < out.transcriptLines; ++i)
    {
        if (transcript[i].kind != agent::ActKind::Action)
            continue;
        foldWord(actionHash, transcript[i].bytes);
        foldBytes(actionHash, reinterpret_cast<const core::u8 *>(transcript[i].text), transcript[i].bytes);
    }
    out.actionSignature = actionHash;

    const Utterance utterance = concludeDialogue(transcript, out.transcriptLines, budget, persona);
    out.utteranceSignature = foldUtterance(utterance, kFnv1aOffsetBasis);

    out.generations = reasoner.generations();
    out.completions = reasoner.completions();
    out.illegalActions = reasoner.illegalActions();
    out.grammarExhausted = reasoner.grammarExhausted();
    out.tokensGenerated = reasoner.tokensGenerated();
    out.stepsSpent = budget.stepsSpent();
    out.satisfied = world.satisfied() ? 1u : 0u;

    /* The control, and the reason the column above means anything. The SAME model, the
       same prompt, the same seeds — generating with no grammar at all — and how often
       it lands on something the world would have accepted. Without this, "zero illegal
       actions" is a claim a demon that never acts would also satisfy. */
    {
        infer::Inference free;
        if (free.initialise(arena, model, infer::CachePolicy::Refuse))
        {
            const infer::Tokenizer tokenizer{vocab};
            char alphabet[agent::kAvailableBytes]{};
            ParityWorld probe;
            const core::u32 alphabetBytes = probe.available(alphabet, agent::kAvailableBytes);

            for (core::u32 attempt = 0u; attempt < 8u; ++attempt)
            {
                core::u32 promptTokens[kReasoningPromptBytes]{};
                core::u32 skipped = 0u;
                const core::u32 promptBytes =
                    intent.bytes < kReasoningPromptBytes ? intent.bytes : kReasoningPromptBytes;
                const core::u32 promptCount =
                    tokenizer.encode(intent.text, promptBytes, promptTokens, kReasoningPromptBytes, skipped);

                infer::GenerationParams params;
                params.sampler.topK = 8u;
                params.sampler.seed = math::deriveStream(infer::paritySamplerSeed(), attempt).state();
                params.maxTokens = 16u;

                core::u32 produced[16]{};
                infer::GenerationReport report{};
                ++out.freeAttempts;
                if (!free.generate(promptTokens, promptCount, params, nullptr, produced, 16u, report))
                    continue;

                char text[agent::kActBytes]{};
                const core::u32 textBytes = tokenizer.decode(produced, report.generated, text, agent::kActBytes);

                // Does the free run's output CONTAIN a legal action name anywhere in it?
                // Deliberately generous: an exact match would be a stricter bar than the
                // constrained run has to clear, and a control has to be easy to pass or
                // it proves nothing when it fails.
                core::u32 cursor = 0u;
                bool named = false;
                while (cursor < alphabetBytes && !named)
                {
                    while (cursor < alphabetBytes && alphabet[cursor] == ' ')
                        ++cursor;
                    const core::u32 start = cursor;
                    while (cursor < alphabetBytes && alphabet[cursor] != ' ')
                        ++cursor;
                    const core::u32 length = cursor - start;
                    if (length == 0u || length > textBytes)
                        continue;
                    for (core::u32 offset = 0u; offset + length <= textBytes && !named; ++offset)
                    {
                        bool match = true;
                        for (core::u32 b = 0u; b < length && match; ++b)
                            match = text[offset + b] == alphabet[start + b];
                        named = match;
                    }
                }
                if (named)
                    ++out.freeLegalNames;
            }
        }
    }

    out.arenaBytes = static_cast<core::u32>(arena.used());
}

} // namespace lpl::mind

#endif // LPL_HAS_FOUNDATION
