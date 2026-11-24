#include "../infer/chat_vocabulary.hpp"
#include "../infer/fixtures/smollm2_byte_pairs.hpp"

#include <lpl/agent/Decision.hpp>
#include <lpl/infer/Model.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/infer/TensorArena.hpp>
#include <lpl/infer/Tokenizer.hpp>
#include <lpl/mind/Budget.hpp>
#include <lpl/mind/Intent.hpp>
#include <lpl/mind/Parity.hpp>
#include <lpl/mind/ReAct.hpp>
#include <lpl/mind/Reasoning.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(reasoning);

namespace {

constexpr lpl::core::usize kMindArenaBytes = 1024u * 1024u;
constexpr lpl::core::usize kMindScratchBytes = 32u * 1024u;
constexpr lpl::core::u32 kChatContext = 128u; ///< Positions that hold a chat turn and its answer.
constexpr lpl::core::u32 kTurnTokenCount = 3u;
constexpr lpl::core::u32 kWordPastTheWorkspace = lpl::infer::kBytePairWorkspaceSymbols + 1u;
constexpr lpl::core::u32 kWordsUnderTheWorkspace = 4000u; ///< Letters of the first word of a goal of two words.

/** The utterance the turn answers. */
constexpr char kUtterance[] = "what is the reactor pressure";

/** The three tokens of SmolLM2's turns, appended to the parity vocabulary. */
constexpr const char *kTurnTokens[kTurnTokenCount] = {"<|endoftext|>", "<|im_start|>", "<|im_end|>"};

/** An action the parity world offers first. */
constexpr char kFirstAction[] = "read_sensor";

/**
 * @brief What the canonical turn, decided by a model, did.
 */
struct TurnRun {
    lpl::core::u32 generations{0u};    ///< Decisions that went to the model.
    lpl::core::u32 completions{0u};    ///< Decisions that spelled a whole action.
    lpl::core::u32 chatTurns{0u};      ///< Decisions whose goal went in the model's chat turn.
    lpl::core::u32 illegalActions{0u}; ///< Actions the world did not offer.
    bool satisfied{false};             ///< Whether the world's goal was met.
    bool ran{false};                   ///< Whether the reasoner was set up and the turn ran.
};

/**
 * @brief Runs the canonical turn over the parity world with a reasoner over @p model, into @p out.
 */
void runTheTurn(lpl::infer::TensorArena &arena, const lpl::infer::Model &model, TurnRun &out)
{
    static lpl::agent::Act transcript[lpl::mind::kParityTranscriptCapacity];
    lpl::core::u8 *const block = arena.claim<lpl::core::u8>(kMindScratchBytes);
    if (block == nullptr)
        return;
    lpl::infer::TensorArena scratch{block, kMindScratchBytes};
    lpl::mind::ModelReasoner reasoner;
    if (!reasoner.initialise(arena, scratch, model, model.vocab(), lpl::infer::paritySamplerSeed()))
        return;

    const lpl::mind::Persona persona = lpl::mind::parityPersona();
    const lpl::mind::Intent intent =
        lpl::mind::parseIntent(reinterpret_cast<const lpl::core::u8 *>(kUtterance), sizeof(kUtterance) - 1u);
    lpl::mind::ParityWorld world;
    lpl::mind::Budget budget{lpl::mind::parityTokenBudget(), lpl::mind::parityStepBudget(),
                             lpl::mind::parityArenaBudget()};
    for (lpl::agent::Act &line : transcript)
        line = lpl::agent::Act{};
    reasoner.bind(persona);
    (void) lpl::mind::runReAct(intent, world, reasoner, budget, transcript, lpl::mind::kParityTranscriptCapacity);
    out.generations = reasoner.generations();
    out.completions = reasoner.completions();
    out.chatTurns = reasoner.chatTurns();
    out.illegalActions = reasoner.illegalActions();
    out.satisfied = world.satisfied();
    out.ran = true;
}

/**
 * @brief The words the merging of every encoding since the start has cut, after one decision of a reasoner over
 *        @p model whose goal is the @p goalBytes bytes at @p goal.
 */
[[nodiscard]] lpl::core::u32 wordsCutDeciding(lpl::infer::TensorArena &arena, const lpl::infer::Model &model,
                                              const char *goal, lpl::core::u32 goalBytes, lpl::core::u32 &generations)
{
    lpl::core::u8 *const block = arena.claim<lpl::core::u8>(kMindScratchBytes);
    if (block == nullptr)
        return 0u;
    lpl::infer::TensorArena scratch{block, kMindScratchBytes};
    lpl::mind::ModelReasoner reasoner;
    if (!reasoner.initialise(arena, scratch, model, model.vocab(), lpl::infer::paritySamplerSeed()))
        return 0u;

    const lpl::mind::Persona persona = lpl::mind::parityPersona();
    lpl::agent::DecisionContext context;
    context.available = kFirstAction;
    context.availableBytes = sizeof(kFirstAction) - 1u;
    context.goal = goal;
    context.goalBytes = goalBytes;
    reasoner.bind(persona);
    const lpl::core::u32 cutBefore = lpl::infer::bytePairWordsCut();
    (void) reasoner.decide(context);
    generations = reasoner.generations();
    return lpl::infer::bytePairWordsCut() - cutBefore;
}

} // namespace

/**
 * @brief Gate P17 reasoning: the canonical turn of gate P16 agency, every move chosen by the
 *        transformer. Inside a grammar it never names an action the world did not offer and
 *        finishes the job; the same model, ungoverned, never names a legal one, which is what makes
 *        the first zero a measurement rather than a reassurance.
 *
 * @details The arena's occupancy is checked against its capacity, not recorded: it differs between
 *          a 64-bit host and i686.
 */
LPL_TEST(the_grammar_makes_an_illegal_action_unspellable)
{
    lpl::mind::ReasoningFoldResult folded{};

    lpl::mind::foldReasoning(folded);
    test.check(folded.generations > 0u, "the model really was consulted");
    test.check(folded.illegalActions == 0u, "not one action the world had not offered");
    test.check(folded.freeAttempts > 0u, "the unconstrained control actually ran");
    test.check(folded.freeLegalNames == 0u, "and the same model, ungoverned, never once named a legal action");
    test.check(folded.satisfied == 1u, "constrained, it finishes the job");
    test.check(folded.arenaBytes > 0u && folded.arenaBytes < lpl::mind::parityReasoningArenaBytes(),
               "the run fits the arena it declared");

    lpl::mind::ReasoningFoldResult again{};

    lpl::mind::foldReasoning(again);
    test.check(again.transcriptSignature == folded.transcriptSignature &&
                   again.actionSignature == folded.actionSignature,
               "two runs of the model-driven turn decide the same things");

    test.measureHexadecimal("transcript_signature", folded.transcriptSignature);
    test.measureHexadecimal("action_signature", folded.actionSignature);
    test.measureHexadecimal("utterance_signature", folded.utteranceSignature);
    test.measure("generations", folded.generations);
    test.measure("completions", folded.completions);
    test.measure("grammar_exhausted", folded.grammarExhausted);
    test.measure("tokens_generated", folded.tokensGenerated);
    test.measure("transcript_lines", folded.transcriptLines);
    test.measure("steps_spent", folded.stepsSpent);
    test.measure("free_attempts", folded.freeAttempts);
}

/**
 * @brief A model with turns decides in its chat turn: every decision of a ChatML model puts the goal in its turn, and
 *        still names only what the world offers and finishes the job; no decision of the Plain model does. In the
 *        parity window of 32 positions, which the turn does not fit beside its answer, no decision of the ChatML model
 *        completes, where the Plain model's do: the turn is what the model is fed.
 */
LPL_TEST(a_model_with_turns_decides_in_its_chat_turn)
{
    lpl::infer::TensorArena arena{kMindArenaBytes};
    lpl::infer::Vocab vocab;
    lpl::infer::Model chat;
    lpl::infer::Model narrow;
    lpl::infer::Model plain;
    const lpl::core::u32 window = lpl::infer::parityModelConfig().contextLength;

    if (!test.check(lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens, kTurnTokenCount, vocab) &&
                        lpl::infer::tests::synthesiseParityModelOver(arena, vocab, lpl::infer::ChatFormat::ChatMl,
                                                                     kChatContext, chat) &&
                        lpl::infer::tests::synthesiseParityModelOver(arena, vocab, lpl::infer::ChatFormat::ChatMl,
                                                                     window, narrow) &&
                        lpl::infer::tests::synthesiseParityModelOver(arena, vocab, lpl::infer::ChatFormat::Plain,
                                                                     window, plain),
                    "a ChatML model of 128 positions, one of 32 and a Plain one of 32 build over the same vocabulary"))
        return;

    TurnRun chatRun{};
    TurnRun narrowRun{};
    TurnRun plainRun{};
    runTheTurn(arena, chat, chatRun);
    runTheTurn(arena, narrow, narrowRun);
    runTheTurn(arena, plain, plainRun);

    test.check(chatRun.ran && chatRun.generations > 0u && chatRun.chatTurns == chatRun.generations,
               "every decision of the ChatML model puts the goal in its chat turn");
    test.check(chatRun.completions > 0u && chatRun.illegalActions == 0u && chatRun.satisfied,
               "and it names only what the world offers, and finishes the job");
    test.check(plainRun.ran && plainRun.generations > 0u && plainRun.chatTurns == 0u && plainRun.completions > 0u,
               "no decision of the Plain model goes in a turn, and they complete");
    test.check(narrowRun.ran && narrowRun.generations > 0u && narrowRun.chatTurns == narrowRun.generations &&
                   narrowRun.completions == 0u,
               "in 32 positions, the ChatML model's turn leaves no room to answer: none of its decisions completes");
    test.measure("chat_generations", chatRun.generations);
    test.measure("chat_turns", chatRun.chatTurns);
    test.measure("chat_completions", chatRun.completions);
    test.measure("plain_generations", plainRun.generations);
}

/**
 * @brief The mind encodes a goal by the model's merges and its pre-tokenizer: in a ChatML turn, a goal of one word
 * longer than the merging workspace is cut once and counted, which longest match never does; a goal of two words whose
 * turn's text passes the workspace is not cut, which merging the text whole would. Either turn is longer than its room,
 * and the rule decides.
 */
LPL_TEST(the_mind_encodes_a_goal_by_the_model_s_merges)
{
    static char oneWord[kWordPastTheWorkspace];
    static char twoWords[kWordPastTheWorkspace + 100u];
    lpl::infer::TensorArena arena{kMindArenaBytes};
    lpl::infer::Vocab vocab;
    lpl::infer::BytePairMerges merges;
    lpl::infer::Model model;

    for (char &byte : oneWord)
        byte = 'a';
    for (char &byte : twoWords)
        byte = 'a';
    twoWords[kWordsUnderTheWorkspace] = ' ';

    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();
    shape.vocabSize = smollm2_byte_pairs::kTokenCount;
    if (!test.check(vocab.borrow(arena, smollm2_byte_pairs::kTokenText, smollm2_byte_pairs::kTokenTextBytes,
                                 smollm2_byte_pairs::kTokenEntries, smollm2_byte_pairs::kTokenCount) &&
                        merges.borrow(arena, smollm2_byte_pairs::kMerges, smollm2_byte_pairs::kMergeCount, vocab) ==
                            lpl::infer::BytePairMergesVerdict::Indexed &&
                        model.synthesise(arena, shape, vocab, merges,
                                         lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel,
                                         lpl::infer::ChatFormat::ChatMl, lpl::infer::parityModelSeed()),
                    "a ChatML model over SmolLM2's merges builds"))
        return;

    lpl::core::u32 oneWordGenerations = 0u;
    lpl::core::u32 twoWordsGenerations = 0u;
    const lpl::core::u32 oneWordCut = wordsCutDeciding(arena, model, oneWord, sizeof(oneWord), oneWordGenerations);
    const lpl::core::u32 twoWordsCut = wordsCutDeciding(arena, model, twoWords, sizeof(twoWords), twoWordsGenerations);

    test.check(oneWordCut == 1u, "a goal of one word of 4097 letters is merged, and cut once");
    test.check(twoWordsCut == 0u, "a goal of two words is split before it is merged, and nothing is cut");
    test.check(oneWordGenerations == 0u && twoWordsGenerations == 0u,
               "and either turn is too long for its room, so the rule decides");
}
