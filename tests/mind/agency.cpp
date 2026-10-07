#include <lpl/agent/Decision.hpp>
#include <lpl/mind/Budget.hpp>
#include <lpl/mind/Dialogue.hpp>
#include <lpl/mind/Intent.hpp>
#include <lpl/mind/Memory.hpp>
#include <lpl/mind/Parity.hpp>
#include <lpl/mind/Persona.hpp>
#include <lpl/mind/ReAct.hpp>
#include <lpl/mind/Recall.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(agency);

namespace {

/**
 * @brief A world with nothing to offer and no goal to reach.
 *
 * @details ParityWorld ignores the intent, so an unanswerable question still walks its four verbs to
 *          completion, and two personas would leave through the same branch: comparing their
 *          outcomes would compare two identical things.
 */
class StuckWorld final : public lpl::agent::IWorldSurface {
public:
    lpl::core::u32 available(char *, lpl::core::u32) noexcept override { return 0u; }

    bool perform(const char *, lpl::core::u32, char *, lpl::core::u32, lpl::core::u32 *) noexcept override
    {
        return false;
    }

    [[nodiscard]] bool satisfied() const noexcept override { return false; }
};

[[nodiscard]] lpl::mind::Intent intentOf(const char *text, lpl::core::u32 bytes)
{
    return lpl::mind::parseIntent(reinterpret_cast<const lpl::core::u8 *>(text), bytes);
}

[[nodiscard]] lpl::mind::MemoryNote noteOf(const char *text, lpl::core::u32 bytes, lpl::core::u32 topic,
                                           lpl::math::Fixed32 salience)
{
    lpl::mind::MemoryNote note;

    note.bytes = bytes;
    for (lpl::core::u32 index = 0u; index < bytes; ++index)
        note.text[index] = text[index];
    note.topic = topic;
    note.salience = salience;
    return note;
}

/**
 * @brief Runs one turn of @p persona against a StuckWorld and says how it concludes.
 */
[[nodiscard]] lpl::mind::Utterance concludeStuckTurn(const lpl::mind::Persona &persona, const lpl::mind::Intent &intent)
{
    lpl::agent::Act transcript[lpl::mind::kParityTranscriptCapacity]{};
    StuckWorld world;
    lpl::mind::DeterministicReasoner reasoner;
    lpl::mind::Budget budget{lpl::mind::parityTokenBudget(), lpl::mind::parityStepBudget(),
                             lpl::mind::parityArenaBudget()};

    reasoner.bind(persona);

    const lpl::core::u32 lines =
        lpl::mind::runReAct(intent, world, reasoner, budget, transcript, lpl::mind::kParityTranscriptCapacity);

    return lpl::mind::concludeDialogue(transcript, lines, budget, persona);
}

} // namespace

/**
 * @brief What the sovereign said is made safe: control bytes are dropped and counted, a question
 *        is filed under its subject, a keyword needs a word break, and the cap is this module's.
 */
LPL_TEST(what_the_sovereign_said_is_made_safe)
{
    const char raw[] = "what is the reactor pressure\x01\x02";
    const lpl::mind::Intent intent = intentOf(raw, sizeof(raw) - 1u);

    test.check(intent.kind == lpl::mind::IntentKind::Question, "a leading interrogative makes it a question");
    test.check(intent.droppedBytes == 2u, "control bytes are dropped and counted, not carried into a prompt");
    test.check(intent.bytes == 28u, "every printable byte survives");
    test.check(intent.topic == lpl::mind::topicOf("reactor", 7u),
               "a question is filed under its subject, not the interrogative and not the verb");
    test.check(intentOf("stop that", 9u).kind == lpl::mind::IntentKind::Stop, "stop is recognised");
    test.check(intentOf("stopwatch reading", 17u).kind != lpl::mind::IntentKind::Stop,
               "a keyword needs a word break: stopwatch is not a stop");

    char oversized[lpl::mind::kIntentBytes + 40u];

    for (char &byte : oversized)
        byte = 'a';

    const lpl::mind::Intent capped = intentOf(oversized, sizeof(oversized));

    test.check(capped.bytes == lpl::mind::kIntentBytes && capped.truncatedBytes == 40u,
               "the cap is this module's and the overflow is counted");
}

/**
 * @brief When the store is full, a trivial note is refused rather than allowed to displace a better
 *        one, and a salient note displaces the weakest.
 */
LPL_TEST(a_trivial_note_never_displaces_a_better_one)
{
    lpl::mind::MemoryStore store;

    for (lpl::core::u32 index = 0u; index < lpl::mind::kMaxNotes; ++index)
    {
        const char letter = static_cast<char>('a' + static_cast<char>(index % 26u));
        lpl::mind::MemoryNote note = noteOf(&letter, 1u, 0u, lpl::math::Fixed32::half());

        note.tick = index;
        store.remember(note);
    }
    test.check(store.count() == lpl::mind::kMaxNotes, "the store fills");

    lpl::mind::MemoryNote trivia = noteOf("z", 1u, 0u, lpl::math::Fixed32::fromRaw(lpl::math::Fixed32::kOne / 10));

    trivia.tick = 999u;
    test.check(!store.remember(trivia) && store.refusals() == 1u,
               "a trivial note is refused rather than allowed to displace something better");

    lpl::mind::MemoryNote important =
        noteOf("Q", 1u, 0u, lpl::math::Fixed32::fromRaw(lpl::math::Fixed32::kOne * 9 / 10));

    important.tick = 1000u;
    test.check(store.remember(important) && store.evictions() == 1u, "a salient note does displace the weakest one");
}

/**
 * @brief Recall filters on the topic first and ranks by similarity second, so two notes with the
 *        same text under different topics are not both returned.
 */
LPL_TEST(the_topic_filters_before_similarity_ranks)
{
    const lpl::math::Fixed32 salience = lpl::math::Fixed32::fromRaw(lpl::math::Fixed32::kOne * 4 / 5);
    lpl::mind::MemoryStore store;

    store.remember(noteOf("reactor pressure nominal", 24u, lpl::mind::topicOf("reactor", 7u), salience));
    store.remember(noteOf("reactor pressure nominal", 24u, lpl::mind::topicOf("kitchen", 7u), salience));

    lpl::mind::RecallHit hits[4]{};
    const lpl::core::u32 found = lpl::mind::recall(store, lpl::mind::topicOf("reactor", 7u), "reactor pressure", 16u,
                                                   lpl::math::Fixed32::zero(), hits, 4u);

    test.check(found == 1u, "two notes with identical text but different topics are not both returned");
    test.check(hits[0].index == 0u, "the one returned is the one the topic selected");
    test.check(lpl::mind::recall(store, 0u, "reactor pressure", 16u, lpl::math::Fixed32::zero(), hits, 4u) == 2u,
               "topic zero means every topic");

    const lpl::mind::TextSignature same = lpl::mind::signatureOf("reactor pressure", 16u);
    const lpl::mind::TextSignature other = lpl::mind::signatureOf("kitchen lighting", 16u);

    test.check(lpl::mind::similarity(same, same) == lpl::math::Fixed32::one(), "a text resembles itself exactly");
    test.check(lpl::mind::similarity(same, other) < lpl::math::Fixed32::half(),
               "unrelated texts do not resemble each other");
}

/**
 * @brief The world offers a grammar, not a menu: an action the situation does not allow is refused,
 *        the legal one succeeds, and the legal alphabet changes after it.
 */
LPL_TEST(an_action_the_situation_forbids_is_refused)
{
    lpl::mind::ParityWorld world;
    char alphabet[lpl::agent::kAvailableBytes]{};
    char observation[lpl::agent::kActBytes]{};
    lpl::core::u32 observed = 0u;
    const lpl::core::u32 before = world.available(alphabet, lpl::agent::kAvailableBytes);

    test.check(!world.perform("open_valve", 10u, observation, lpl::agent::kActBytes, &observed),
               "an action the situation does not allow is refused, not pretended");
    test.check(world.perform("read_sensor", 11u, observation, lpl::agent::kActBytes, &observed),
               "the one action that is legal succeeds");
    test.check(world.available(alphabet, lpl::agent::kAvailableBytes) != before, "the legal alphabet changed");
}

/**
 * @brief With nothing left to try, a cautious persona hands the problem back and a bold one reports
 *        what it managed: the same code, a different identity, a different transcript.
 */
LPL_TEST(a_bolder_persona_reaches_another_conclusion)
{
    const lpl::mind::Intent unanswerable = intentOf("recalibrate everything", 22u);
    const lpl::mind::Persona cautious = lpl::mind::parityPersona();
    lpl::mind::Persona bold = lpl::mind::parityPersona();

    bold.caution = lpl::math::Fixed32::fromRaw(lpl::math::Fixed32::kOne / 10);

    const lpl::mind::Utterance cautiousSaid = concludeStuckTurn(cautious, unanswerable);
    const lpl::mind::Utterance boldSaid = concludeStuckTurn(bold, unanswerable);

    test.check(cautiousSaid.kind == lpl::mind::Address::Ask,
               "with nothing left to try, a cautious demon hands the problem back");
    test.check(boldSaid.kind == lpl::mind::Address::Answer,
               "a bold one reports what it managed: same code, different persona");
    test.check(lpl::mind::foldUtterance(cautiousSaid, 0u) != lpl::mind::foldUtterance(boldSaid, 0u),
               "editing identity changes the transcript and not only its tone");
}

/**
 * @brief A budget that expires mid-plan produces a report, never an answer, and the refusal is
 *        counted.
 */
LPL_TEST(an_exhausted_turn_reports_and_does_not_answer)
{
    lpl::mind::ParityWorld world;
    lpl::mind::DeterministicReasoner reasoner;
    const lpl::mind::Persona persona = lpl::mind::parityPersona();
    const lpl::mind::Intent intent = intentOf("what is the reactor pressure", 28u);
    lpl::mind::Budget starved{4u, 1u, lpl::mind::parityArenaBudget()};
    lpl::agent::Act transcript[lpl::mind::kParityTranscriptCapacity]{};

    reasoner.bind(persona);

    const lpl::core::u32 lines =
        lpl::mind::runReAct(intent, world, reasoner, starved, transcript, lpl::mind::kParityTranscriptCapacity);
    const lpl::mind::Utterance said = lpl::mind::concludeDialogue(transcript, lines, starved, persona);

    test.check(said.kind == lpl::mind::Address::Report,
               "a budget that expired mid-plan produces a report, never an answer");
    test.check(starved.denied() > 0u, "and the refusal is counted rather than silent");
    test.check(!world.satisfied(), "the world agrees the job was not done");
}

/**
 * @brief The final stretch is computed by multiplication: a division by ten would leave a budget of
 *        nine without one.
 */
LPL_TEST(a_budget_of_nine_still_has_a_last_tenth)
{
    lpl::mind::Budget small{9u, 4u, 128u};

    test.check(!small.finalStretch(), "a fresh budget is not in its final stretch");
    small.claimTokens(9u);
    test.check(small.finalStretch(), "a spent one is");
}

/**
 * @brief Gate P16 agency: one turn of thought decides the same things on both targets: which note
 *        survived a full memory, which action came first, whether the turn ended in an answer, and
 *        what it cost.
 */
LPL_TEST(one_turn_of_thought_decides_the_same_things)
{
    lpl::mind::AgencyFoldResult folded{};

    lpl::mind::foldAgency(folded);
    test.check(folded.worldSatisfied == 1u, "the canonical turn actually finishes the job");
    test.check(folded.transcriptLines > 2u, "and it took more than one move to do it");
    test.check(folded.refusals > 0u && folded.evictions > 0u,
               "the canonical session exercises both halves of the eviction rule");
    test.check(folded.recallHits > 0u, "the lookup surfaced something to think with");
    test.check(folded.droppedBytes == 2u, "the canonical utterance's control bytes did not survive");
    test.check(folded.utteranceKind == static_cast<lpl::core::u32>(lpl::mind::Address::Answer),
               "the turn ends in an answer");

    lpl::mind::AgencyFoldResult again{};

    lpl::mind::foldAgency(again);
    test.check(again.transcriptSignature == folded.transcriptSignature &&
                   again.memorySignature == folded.memorySignature &&
                   again.utteranceSignature == folded.utteranceSignature,
               "running the turn twice decides the same things");

    test.measureHexadecimal("persona_signature", folded.personaSignature);
    test.measureHexadecimal("intent_signature", folded.intentSignature);
    test.measureHexadecimal("memory_signature", folded.memorySignature);
    test.measureHexadecimal("recall_signature", folded.recallSignature);
    test.measureHexadecimal("transcript_signature", folded.transcriptSignature);
    test.measureHexadecimal("utterance_signature", folded.utteranceSignature);
    test.measureHexadecimal("budget_signature", folded.budgetSignature);
    test.measure("intent_kind", folded.intentKind);
    test.measure("notes_held", folded.notesHeld);
    test.measure("evictions", folded.evictions);
    test.measure("refusals", folded.refusals);
    test.measure("recall_hits", folded.recallHits);
    test.measure("transcript_lines", folded.transcriptLines);
    test.measure("steps_spent", folded.stepsSpent);
    test.measure("tokens_spent", folded.tokensSpent);
    test.measure("world_refusals", folded.worldRefusals);
}
