/**
 * @file test_agency_parity.cpp
 * @brief One turn of thought, two machines, one transcript.
 *
 * The gate for the agency floor. What it guards is not an algorithm but a SEQUENCE OF
 * DECISIONS: which note survived a full memory, which action was reached for first,
 * whether the turn ended in an answer or a question, and what it cost. A demon that
 * decided any of those differently in ring 0 than on the host would be a different
 * demon wearing the same persona.
 *
 * Several checks below assert the OPPOSITE of what the fold asserts — that a bolder
 * persona reaches a different conclusion, that a flood of trivia is turned away, that
 * an unreachable action is refused rather than pretended. A gate that only ever
 * confirms the happy path is satisfied by a policy that does nothing.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/agent/Decision.hpp>
#include <lpl/mind/Budget.hpp>
#include <lpl/mind/Dialogue.hpp>
#include <lpl/mind/Intent.hpp>
#include <lpl/mind/Memory.hpp>
#include <lpl/mind/Parity.hpp>
#include <lpl/mind/Persona.hpp>
#include <lpl/mind/ReAct.hpp>
#include <lpl/mind/Recall.hpp>

#include <cstdio>

namespace {

int gFailures = 0;
int gChecks = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++gFailures;
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== agency: one turn of thought, two machines ==\n");

    // ── The intent is treated as an untrusted packet ─────────────────────────
    std::printf("\n-- what the sovereign said, made safe --\n");
    {
        const char raw[] = "what is the reactor pressure\x01\x02";
        const mind::Intent intent =
            mind::parseIntent(reinterpret_cast<const core::u8 *>(raw), sizeof(raw) - 1u);

        check(intent.kind == mind::IntentKind::Question, "a leading interrogative makes it a question");
        check(intent.droppedBytes == 2u, "control bytes are dropped and counted, not carried into a prompt");
        check(intent.bytes == 28u, "every printable byte survives");
        check(intent.topic == mind::topicOf("reactor", 7u),
              "a question is filed under its SUBJECT — not the interrogative, not the verb");

        const mind::Intent stop =
            mind::parseIntent(reinterpret_cast<const core::u8 *>("stop that"), 9u);
        check(stop.kind == mind::IntentKind::Stop, "stop is recognised");

        const mind::Intent watch =
            mind::parseIntent(reinterpret_cast<const core::u8 *>("stopwatch reading"), 17u);
        check(watch.kind != mind::IntentKind::Stop,
              "a keyword needs a word break — stopwatch is not a stop");

        char oversized[mind::kIntentBytes + 40u];
        for (core::u32 i = 0u; i < sizeof(oversized); ++i)
            oversized[i] = 'a';
        const mind::Intent capped =
            mind::parseIntent(reinterpret_cast<const core::u8 *>(oversized), sizeof(oversized));
        check(capped.bytes == mind::kIntentBytes && capped.truncatedBytes == 40u,
              "the cap is this module's and the overflow is counted");
    }

    // ── Memory keeps what matters, not what arrived last ─────────────────────
    std::printf("\n-- which note dies when the store is full --\n");
    {
        mind::MemoryStore store;
        for (core::u32 i = 0u; i < mind::kMaxNotes; ++i)
        {
            mind::MemoryNote note;
            note.bytes = 1u;
            note.text[0] = static_cast<char>('a' + static_cast<char>(i % 26u));
            note.tick = i;
            note.salience = math::Fixed32::fromFloat(0.5f);
            store.remember(note);
        }
        check(store.count() == mind::kMaxNotes, "the store fills");

        mind::MemoryNote trivia;
        trivia.bytes = 1u;
        trivia.text[0] = 'z';
        trivia.tick = 999u;
        trivia.salience = math::Fixed32::fromFloat(0.1f);
        const bool kept = store.remember(trivia);
        check(!kept && store.refusals() == 1u,
              "a trivial note is REFUSED rather than allowed to displace something better");

        mind::MemoryNote important;
        important.bytes = 1u;
        important.text[0] = 'Q';
        important.tick = 1000u;
        important.salience = math::Fixed32::fromFloat(0.9f);
        check(store.remember(important) && store.evictions() == 1u,
              "a salient note does displace the weakest one");
    }

    // ── Recall narrows by fact before it reaches for resemblance ─────────────
    std::printf("\n-- structured filter first, similarity second --\n");
    {
        mind::MemoryStore store;
        mind::MemoryNote reactor;
        reactor.bytes = 24u;
        for (core::u32 i = 0u; i < 24u; ++i)
            reactor.text[i] = "reactor pressure nominal"[i];
        reactor.topic = mind::topicOf("reactor", 7u);
        reactor.salience = math::Fixed32::fromFloat(0.8f);
        store.remember(reactor);

        mind::MemoryNote kitchen;
        kitchen.bytes = 24u;
        for (core::u32 i = 0u; i < 24u; ++i)
            kitchen.text[i] = "reactor pressure nominal"[i];
        kitchen.topic = mind::topicOf("kitchen", 7u);
        kitchen.salience = math::Fixed32::fromFloat(0.8f);
        store.remember(kitchen);

        mind::RecallHit hits[4]{};
        const core::u32 found = mind::recall(store, mind::topicOf("reactor", 7u), "reactor pressure", 16u,
                                             math::Fixed32::zero(), hits, 4u);
        check(found == 1u,
              "two notes with IDENTICAL text but different topics are not both returned");
        check(hits[0].index == 0u, "the one returned is the one the fact selected");

        const core::u32 anyTopic =
            mind::recall(store, 0u, "reactor pressure", 16u, math::Fixed32::zero(), hits, 4u);
        check(anyTopic == 2u, "topic zero means every topic");

        const mind::TextSignature same = mind::signatureOf("reactor pressure", 16u);
        check(mind::similarity(same, same) == math::Fixed32::one(), "a text resembles itself exactly");
        const mind::TextSignature other = mind::signatureOf("kitchen lighting", 16u);
        check(mind::similarity(same, other) < math::Fixed32::half(),
              "unrelated texts do not resemble each other");
    }

    // ── The loop, against a world whose alphabet moves ───────────────────────
    std::printf("\n-- reason, act, observe, repeat --\n");
    {
        mind::ParityWorld world;
        char alphabet[agent::kAvailableBytes]{};
        const core::u32 before = world.available(alphabet, agent::kAvailableBytes);

        core::u32 observed = 0u;
        char observation[agent::kActBytes]{};
        check(!world.perform("open_valve", 10u, observation, agent::kActBytes, &observed),
              "an action the situation does not allow is refused, not pretended");

        check(world.perform("read_sensor", 11u, observation, agent::kActBytes, &observed),
              "the one action that is legal succeeds");
        const core::u32 after = world.available(alphabet, agent::kAvailableBytes);
        check(before != after, "the legal alphabet CHANGED — this is a grammar, not a menu");
    }

    // ── Identity is data, and editing it changes the outcome ─────────────────
    std::printf("\n-- a bolder persona reaches a different conclusion --\n");
    {
        /* A world with nothing to offer and no goal to reach. It has to be built for
           this check rather than reused: ParityWorld ignores the intent entirely, so an
           "unanswerable" question still walks its four verbs to completion — which made
           the first version of this check pass both personas through the SAME branch
           and compare two identical outcomes. */
        class StuckWorld final : public agent::IWorldSurface {
          public:
            core::u32 available(char *, core::u32) noexcept override { return 0u; }
            bool perform(const char *, core::u32, char *, core::u32, core::u32 *) noexcept override
            {
                return false;
            }
            [[nodiscard]] bool satisfied() const noexcept override { return false; }
        };

        const mind::Intent unanswerable =
            mind::parseIntent(reinterpret_cast<const core::u8 *>("recalibrate everything"), 22u);

        agent::Act cautiousTranscript[mind::kParityTranscriptCapacity]{};
        StuckWorld cautiousWorld;
        mind::DeterministicReasoner reasoner;
        mind::Budget cautiousBudget{mind::parityTokenBudget(), mind::parityStepBudget(),
                                    mind::parityArenaBudget()};
        mind::Persona cautious = mind::parityPersona();
        reasoner.bind(cautious);
        const core::u32 cautiousLines = mind::runReAct(unanswerable, cautiousWorld, reasoner, cautiousBudget,
                                                       cautiousTranscript, mind::kParityTranscriptCapacity);

        agent::Act boldTranscript[mind::kParityTranscriptCapacity]{};
        StuckWorld boldWorld;
        mind::Budget boldBudget{mind::parityTokenBudget(), mind::parityStepBudget(), mind::parityArenaBudget()};
        mind::Persona bold = mind::parityPersona();
        bold.caution = math::Fixed32::fromFloat(0.10f);
        mind::DeterministicReasoner boldReasoner;
        boldReasoner.bind(bold);
        const core::u32 boldLines = mind::runReAct(unanswerable, boldWorld, boldReasoner, boldBudget,
                                                   boldTranscript, mind::kParityTranscriptCapacity);

        const mind::Utterance cautiousSaid =
            mind::concludeDialogue(cautiousTranscript, cautiousLines, cautiousBudget, cautious);
        const mind::Utterance boldSaid = mind::concludeDialogue(boldTranscript, boldLines, boldBudget, bold);

        check(cautiousSaid.kind == mind::Address::Ask,
              "with nothing left to try, a cautious demon hands the problem back");
        check(boldSaid.kind == mind::Address::Answer,
              "a bold one reports what it managed — same code, different persona");
        check(mind::foldUtterance(cautiousSaid, 0u) != mind::foldUtterance(boldSaid, 0u),
              "editing identity changes the transcript and not only its tone");
    }

    // ── A turn that runs out never claims to have finished ───────────────────
    std::printf("\n-- an exhausted turn reports, it does not answer --\n");
    {
        mind::ParityWorld world;
        mind::DeterministicReasoner reasoner;
        mind::Persona persona = mind::parityPersona();
        const mind::Intent intent =
            mind::parseIntent(reinterpret_cast<const core::u8 *>("what is the reactor pressure"), 28u);

        mind::Budget starved{4u, 1u, mind::parityArenaBudget()};
        agent::Act transcript[mind::kParityTranscriptCapacity]{};
        reasoner.bind(persona);
        const core::u32 lines = mind::runReAct(intent, world, reasoner, starved, transcript,
                                               mind::kParityTranscriptCapacity);
        const mind::Utterance said = mind::concludeDialogue(transcript, lines, starved, persona);

        check(said.kind == mind::Address::Report,
              "a budget that expired mid-plan produces a report, never an answer");
        check(starved.denied() > 0u, "and the refusal is counted rather than silent");
        check(!world.satisfied(), "the world agrees the job was not done");
    }

    // ── The budget's final stretch survives a small budget ───────────────────
    std::printf("\n-- a budget of nine still has a last tenth --\n");
    {
        mind::Budget small{9u, 4u, 128u};
        check(!small.finalStretch(), "a fresh budget is not in its final stretch");
        small.claimTokens(9u);
        check(small.finalStretch(),
              "computed by multiplication: a division by ten would make this unreachable");
    }

    // ── The fold, and that it is a function of nothing but its inputs ────────
    mind::AgencyFoldResult folded{};
    mind::foldAgency(folded);

    std::printf("\n-- the canonical turn --\n");
    check(folded.worldSatisfied == 1u, "the canonical turn actually finishes the job");
    check(folded.transcriptLines > 2u, "and it took more than one move to do it");
    check(folded.refusals > 0u && folded.evictions > 0u,
          "the canonical session exercises BOTH halves of the eviction rule");
    check(folded.recallHits > 0u, "the lookup surfaced something to think with");
    check(folded.droppedBytes == 2u, "the canonical utterance's control bytes did not survive");

    mind::AgencyFoldResult again{};
    mind::foldAgency(again);
    check(again.transcriptSignature == folded.transcriptSignature &&
              again.memorySignature == folded.memorySignature &&
              again.utteranceSignature == folded.utteranceSignature,
          "running the turn twice decides the same things");

    std::printf("\n-- signatures the kernel must reproduce --\n");
    std::printf("  persona_sig    = 0x%08X\n", folded.personaSignature);
    std::printf("  intent_sig     = 0x%08X\n", folded.intentSignature);
    std::printf("  memory_sig     = 0x%08X\n", folded.memorySignature);
    std::printf("  recall_sig     = 0x%08X\n", folded.recallSignature);
    std::printf("  transcript_sig = 0x%08X\n", folded.transcriptSignature);
    std::printf("  utterance_sig  = 0x%08X\n", folded.utteranceSignature);
    std::printf("  budget_sig     = 0x%08X\n", folded.budgetSignature);
    std::printf("  intent_kind    = %u\n", folded.intentKind);
    std::printf("  dropped        = %u\n", folded.droppedBytes);
    std::printf("  notes_held     = %u\n", folded.notesHeld);
    std::printf("  evictions      = %u\n", folded.evictions);
    std::printf("  refusals       = %u\n", folded.refusals);
    std::printf("  recall_hits    = %u\n", folded.recallHits);
    std::printf("  lines          = %u\n", folded.transcriptLines);
    std::printf("  steps          = %u\n", folded.stepsSpent);
    std::printf("  tokens         = %u\n", folded.tokensSpent);
    std::printf("  utterance_kind = %u\n", folded.utteranceKind);
    std::printf("  satisfied      = %u\n", folded.worldSatisfied);
    std::printf("  world_refusals = %u\n", folded.worldRefusals);

    // ── The same turn, decided by a model instead of by a rule ───────────────
    std::printf("\n-- a model that learned nothing, inside a grammar --\n");
    mind::ReasoningFoldResult reasoning{};
    mind::foldReasoning(reasoning);

    check(reasoning.generations > 0u, "the model really was consulted");
    check(reasoning.illegalActions == 0u,
          "not one action the world had not offered — the grammar makes them unspellable");
    check(reasoning.freeAttempts > 0u, "the unconstrained control actually ran");
    check(reasoning.freeLegalNames == 0u,
          "and the SAME model, ungoverned, never once named a legal action");
    check(reasoning.satisfied == 1u, "constrained, it finishes the job");
    check(reasoning.arenaBytes < mind::parityReasoningArenaBytes(), "the run fits the arena it declared");

    mind::ReasoningFoldResult reasoningAgain{};
    mind::foldReasoning(reasoningAgain);
    check(reasoningAgain.transcriptSignature == reasoning.transcriptSignature &&
              reasoningAgain.actionSignature == reasoning.actionSignature,
          "two runs of the model-driven turn decide the same things");

    std::printf("\n-- P17 signatures the kernel must reproduce --\n");
    std::printf("  reason_transcript_sig = 0x%08X\n", reasoning.transcriptSignature);
    std::printf("  reason_action_sig     = 0x%08X\n", reasoning.actionSignature);
    std::printf("  reason_utterance_sig  = 0x%08X\n", reasoning.utteranceSignature);
    std::printf("  reason_generations    = %u\n", reasoning.generations);
    std::printf("  reason_completions    = %u\n", reasoning.completions);
    std::printf("  reason_illegal        = %u\n", reasoning.illegalActions);
    std::printf("  reason_exhausted      = %u\n", reasoning.grammarExhausted);
    std::printf("  reason_tokens         = %u\n", reasoning.tokensGenerated);
    std::printf("  reason_lines          = %u\n", reasoning.transcriptLines);
    std::printf("  reason_steps          = %u\n", reasoning.stepsSpent);
    std::printf("  reason_satisfied      = %u\n", reasoning.satisfied);
    std::printf("  reason_free_attempts  = %u\n", reasoning.freeAttempts);
    std::printf("  reason_free_legal     = %u\n", reasoning.freeLegalNames);
    std::printf("  reason_arena          = %u\n", reasoning.arenaBytes);

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
