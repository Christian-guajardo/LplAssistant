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
constexpr lpl::core::u32 kChatContext = 128u;   ///< Positions that hold a chat turn and its answer.
constexpr lpl::core::u32 kDecisionTokens = 16u; ///< Tokens one decision may produce, as the reasoner bounds them.
constexpr lpl::core::u32 kTurnTokenCount = 3u;
constexpr lpl::core::u32 kWordPastTheWorkspace = lpl::infer::kBytePairWorkspaceSymbols + 1u;
constexpr lpl::core::u32 kWordsUnderTheWorkspace = 4000u; ///< Letters of the first word of a goal of two words.

/** A goal of twelve bytes that SmolLM2's merges and longest match over the same tokens encode to different counts. */
constexpr char kMergedApartGoal[] = "sistant name";

/** A goal holding a byte the parity vocabulary does not spell. */
constexpr char kUnspelledGoal[] = "read\001";

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
    lpl::core::u32 generations{0u};        ///< Decisions that went to the model.
    lpl::core::u32 completions{0u};        ///< Decisions that spelled a whole action.
    lpl::core::u32 chatTurns{0u};          ///< Decisions whose goal went in the model's chat turn.
    lpl::core::u32 chatTurnsRefused{0u};   ///< Decisions whose turn could not be had, left to the rule.
    lpl::core::u32 promptTokensFed{0u};    ///< Tokens fed to the model as prompts.
    lpl::core::u32 promptBytesSkipped{0u}; ///< Bytes of the prompts no token spelled.
    lpl::core::u32 illegalActions{0u};     ///< Actions the world did not offer.
    lpl::core::u32 wordsCut{0u};           ///< Words the merging cut while the reasoner decided.
    bool satisfied{false};                 ///< Whether the world's goal was met.
    bool ran{false};                       ///< Whether the reasoner was set up and the turn ran.
};

/**
 * @brief Copies what @p reasoner counted into @p out.
 */
void readCounts(const lpl::mind::ModelReasoner &reasoner, TurnRun &out)
{
    out.generations = reasoner.generations();
    out.completions = reasoner.completions();
    out.chatTurns = reasoner.chatTurns();
    out.chatTurnsRefused = reasoner.chatTurnsRefused();
    out.promptTokensFed = reasoner.promptTokensFed();
    out.promptBytesSkipped = reasoner.promptBytesSkipped();
    out.illegalActions = reasoner.illegalActions();
    out.ran = true;
}

/**
 * @brief The tokens of the chat turn the mind puts the canonical goal in, over @p vocab, or 0 when it cannot be had.
 */
[[nodiscard]] lpl::core::u32 canonicalTurnTokens(lpl::infer::TensorArena &arena, const lpl::infer::Vocab &vocab)
{
    lpl::core::u8 *const block = arena.claim<lpl::core::u8>(kMindScratchBytes);
    lpl::core::u32 *const ids = arena.claim<lpl::core::u32>(lpl::mind::kReasoningChatTurnTokens);
    lpl::infer::ChatTokens tokens{};
    if (block == nullptr || ids == nullptr ||
        !lpl::infer::findChatTokens(lpl::infer::ChatFormat::ChatMl, vocab, tokens))
        return 0u;

    lpl::infer::TensorArena scratch{block, kMindScratchBytes};
    const lpl::mind::Intent intent =
        lpl::mind::parseIntent(reinterpret_cast<const lpl::core::u8 *>(kUtterance), sizeof(kUtterance) - 1u);
    const lpl::infer::ChatTurnReport turn =
        lpl::infer::encodeChatTurn(scratch, lpl::infer::Tokenizer{vocab}, tokens, lpl::mind::kReasoningSystemMessage,
                                   sizeof(lpl::mind::kReasoningSystemMessage) - 1u, intent.text, intent.bytes, ids,
                                   lpl::mind::kReasoningChatTurnTokens);
    return turn.verdict == lpl::infer::ChatTurnVerdict::Encoded ? turn.tokens : 0u;
}

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
    readCounts(reasoner, out);
    out.satisfied = world.satisfied();
}

/**
 * @brief The tokens the canonical goal's first kReasoningPromptBytes bytes encode to by @p tokenizer.
 */
[[nodiscard]] lpl::core::u32 canonicalPromptTokens(const lpl::infer::Tokenizer &tokenizer)
{
    const lpl::mind::Intent intent =
        lpl::mind::parseIntent(reinterpret_cast<const lpl::core::u8 *>(kUtterance), sizeof(kUtterance) - 1u);
    lpl::core::u32 ids[lpl::mind::kReasoningPromptBytes]{};
    lpl::core::u32 skipped = 0u;
    const lpl::core::u32 bytes =
        intent.bytes < lpl::mind::kReasoningPromptBytes ? intent.bytes : lpl::mind::kReasoningPromptBytes;

    return tokenizer.encode(intent.text, bytes, ids, lpl::mind::kReasoningPromptBytes, skipped);
}

/**
 * @brief What a reasoner over @p model counted after one decision whose goal is the @p goalBytes bytes at @p goal, and
 *        the words the merging cut meanwhile.
 */
[[nodiscard]] TurnRun decideOnce(lpl::infer::TensorArena &arena, const lpl::infer::Model &model, const char *goal,
                                 lpl::core::u32 goalBytes)
{
    TurnRun run{};
    lpl::core::u8 *const block = arena.claim<lpl::core::u8>(kMindScratchBytes);
    if (block == nullptr)
        return run;
    lpl::infer::TensorArena scratch{block, kMindScratchBytes};
    lpl::mind::ModelReasoner reasoner;
    if (!reasoner.initialise(arena, scratch, model, model.vocab(), lpl::infer::paritySamplerSeed()))
        return run;

    const lpl::mind::Persona persona = lpl::mind::parityPersona();
    lpl::agent::DecisionContext context;
    context.available = kFirstAction;
    context.availableBytes = sizeof(kFirstAction) - 1u;
    context.goal = goal;
    context.goalBytes = goalBytes;
    reasoner.bind(persona);
    const lpl::core::u32 cutBefore = lpl::infer::bytePairWordsCut();
    (void) reasoner.decide(context);
    readCounts(reasoner, run);
    run.wordsCut = lpl::infer::bytePairWordsCut() - cutBefore;
    return run;
}

/**
 * @brief Builds the fixture's vocabulary and merges in @p arena, and a model over them in @p format.
 */
[[nodiscard]] bool synthesiseOverTheMerges(lpl::infer::TensorArena &arena, lpl::infer::Vocab &vocab,
                                           lpl::infer::BytePairMerges &merges, lpl::infer::ChatFormat format,
                                           lpl::infer::Model &model)
{
    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();

    shape.vocabSize = smollm2_byte_pairs::kTokenCount;
    return vocab.borrow(arena, smollm2_byte_pairs::kTokenText, smollm2_byte_pairs::kTokenTextBytes,
                        smollm2_byte_pairs::kTokenEntries, smollm2_byte_pairs::kTokenCount) &&
           merges.borrow(arena, smollm2_byte_pairs::kMerges, smollm2_byte_pairs::kMergeCount, vocab) ==
               lpl::infer::BytePairMergesVerdict::Indexed &&
           model.synthesise(arena, shape, vocab, merges, lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel,
                            format, lpl::infer::parityModelSeed());
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
 * @brief A model with turns decides in its chat turn: every decision of a ChatML model is fed the very turn
 *        encodeChatTurn puts the mind's system message and the whole goal in, and still names only what the world
 *        offers and finishes the job. The Plain model's decisions go in no turn, and are each fed the goal's first
 *        twelve bytes as they are.
 */
LPL_TEST(a_model_with_turns_decides_in_its_chat_turn)
{
    lpl::infer::TensorArena arena{kMindArenaBytes};
    lpl::infer::Vocab vocab;
    lpl::infer::Model chat;
    lpl::infer::Model plain;
    const lpl::core::u32 window = lpl::infer::parityModelConfig().contextLength;

    const bool built = lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens, kTurnTokenCount, vocab);
    const lpl::core::u32 turn = built ? canonicalTurnTokens(arena, vocab) : 0u;
    const lpl::core::u32 prompt = built ? canonicalPromptTokens(lpl::infer::Tokenizer{vocab}) : 0u;
    if (!test.check(turn > window && turn + kDecisionTokens <= kChatContext && prompt > 0u &&
                        lpl::infer::tests::synthesiseParityModelOver(arena, vocab, lpl::infer::ChatFormat::ChatMl,
                                                                     kChatContext, chat) &&
                        lpl::infer::tests::synthesiseParityModelOver(arena, vocab, lpl::infer::ChatFormat::Plain,
                                                                     window, plain),
                    "the canonical turn passes the parity window, and a ChatML and a Plain model build"))
        return;

    TurnRun chatRun{};
    TurnRun plainRun{};
    runTheTurn(arena, chat, chatRun);
    runTheTurn(arena, plain, plainRun);

    test.check(chatRun.ran && chatRun.generations > 0u && chatRun.chatTurns == chatRun.generations &&
                   chatRun.chatTurnsRefused == 0u,
               "every decision of the ChatML model puts the goal in its chat turn");
    test.check(chatRun.promptTokensFed == turn * chatRun.generations,
               "and is fed that turn, the mind's system message and the whole goal, to its last token");
    test.check(chatRun.completions > 0u && chatRun.illegalActions == 0u && chatRun.satisfied,
               "and it names only what the world offers, and finishes the job");
    test.check(plainRun.ran && plainRun.generations > 0u && plainRun.chatTurns == 0u && plainRun.completions > 0u &&
                   plainRun.promptTokensFed == prompt * plainRun.generations,
               "no decision of the Plain model goes in a turn: each is fed the goal's first twelve bytes");
    test.measure("turn_tokens", turn);
    test.measure("chat_generations", chatRun.generations);
    test.measure("chat_completions", chatRun.completions);
    test.measure("plain_generations", plainRun.generations);
}

/**
 * @brief The mind encodes a goal by the model's merges and its pre-tokenizer: in a ChatML turn, a goal of one word
 * longer than the merging workspace is cut once and counted, which longest match never does; a goal of two words whose
 * turn's text passes the workspace is not cut, which merging the text whole would. Either turn is longer than its room:
 * the rule decides, and the refusal is counted.
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
    if (!test.check(synthesiseOverTheMerges(arena, vocab, merges, lpl::infer::ChatFormat::ChatMl, model),
                    "a ChatML model over SmolLM2's merges builds"))
        return;

    const TurnRun one = decideOnce(arena, model, oneWord, sizeof(oneWord));
    const TurnRun two = decideOnce(arena, model, twoWords, sizeof(twoWords));

    test.check(one.ran && one.wordsCut == 1u, "a goal of one word of 4097 letters is merged, and cut once");
    test.check(two.ran && two.wordsCut == 0u, "a goal of two words is split before it is merged, and nothing is cut");
    test.check(one.generations == 0u && two.generations == 0u && one.chatTurns == 0u && two.chatTurns == 0u,
               "either turn is too long for its room, so the rule decides");
    test.check(one.chatTurnsRefused == 1u && two.chatTurnsRefused == 1u, "and each refusal is counted");
}

/**
 * @brief A Plain model with merges is fed its goal as its merges and pre-tokenizer encode it: twelve bytes that
 * SmolLM2's merges and longest match over the same tokens encode to different counts are fed as the merges count them.
 */
LPL_TEST(a_plain_model_with_merges_is_fed_its_goal_by_its_merges)
{
    lpl::infer::TensorArena arena{kMindArenaBytes};
    lpl::infer::Vocab vocab;
    lpl::infer::BytePairMerges merges;
    lpl::infer::Model model;
    lpl::infer::Tokenizer merged;
    lpl::core::u32 ids[lpl::mind::kReasoningPromptBytes]{};
    lpl::core::u32 skipped = 0u;

    if (!test.check(synthesiseOverTheMerges(arena, vocab, merges, lpl::infer::ChatFormat::Plain, model) &&
                        merged.bind(arena, vocab, merges, lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel),
                    "a Plain model over SmolLM2's merges builds, and a tokenizer is bound to them"))
        return;

    const lpl::core::u32 byMerges =
        merged.encode(kMergedApartGoal, sizeof(kMergedApartGoal) - 1u, ids, lpl::mind::kReasoningPromptBytes, skipped);
    const lpl::core::u32 byLongestMatch = lpl::infer::Tokenizer{vocab}.encode(
        kMergedApartGoal, sizeof(kMergedApartGoal) - 1u, ids, lpl::mind::kReasoningPromptBytes, skipped);
    const TurnRun run = decideOnce(arena, model, kMergedApartGoal, sizeof(kMergedApartGoal) - 1u);

    test.check(byMerges != byLongestMatch,
               "the goal's merges and longest match give different counts: a fixture check");
    test.check(run.ran && run.generations == 1u && run.chatTurns == 0u && run.promptTokensFed == byMerges,
               "the model is fed the goal as its merges encode it");
    test.measure("tokens_by_merges", byMerges);
    test.measure("tokens_by_longest_match", byLongestMatch);
}

/**
 * @brief A goal's byte no token spells is dropped and counted, whether the goal is fed as it is or in a chat turn.
 */
LPL_TEST(a_goal_s_byte_no_token_spells_is_counted)
{
    lpl::infer::TensorArena arena{kMindArenaBytes};
    lpl::infer::Vocab vocab;
    lpl::infer::Model chat;
    lpl::infer::Model plain;

    if (!test.check(lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens, kTurnTokenCount, vocab) &&
                        lpl::infer::tests::synthesiseParityModelOver(arena, vocab, lpl::infer::ChatFormat::ChatMl,
                                                                     kChatContext, chat) &&
                        lpl::infer::tests::synthesiseParityModelOver(arena, vocab, lpl::infer::ChatFormat::Plain,
                                                                     kChatContext, plain),
                    "a ChatML and a Plain model build over the parity vocabulary and the three turn tokens"))
        return;

    const TurnRun inTurn = decideOnce(arena, chat, kUnspelledGoal, sizeof(kUnspelledGoal) - 1u);
    const TurnRun asItIs = decideOnce(arena, plain, kUnspelledGoal, sizeof(kUnspelledGoal) - 1u);

    test.check(inTurn.ran && inTurn.chatTurns == 1u && inTurn.promptBytesSkipped == 1u,
               "the byte 0x01 of a goal put in a chat turn is counted");
    test.check(asItIs.ran && asItIs.chatTurns == 0u && asItIs.promptBytesSkipped == 1u,
               "and so is that of a goal fed as it is");
}

/**
 * @brief A reasoner refuses a vocabulary that is not its model's own, whose tokens the model's merges and turn tokens
 *        are not numbered against.
 */
LPL_TEST(a_reasoner_refuses_a_vocabulary_that_is_not_its_model_s)
{
    lpl::infer::TensorArena arena{kMindArenaBytes};
    lpl::infer::Vocab vocab;
    lpl::infer::Vocab other;
    lpl::infer::Model model;

    if (!test.check(lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens, kTurnTokenCount, vocab) &&
                        lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens, kTurnTokenCount, other) &&
                        lpl::infer::tests::synthesiseParityModelOver(arena, vocab, lpl::infer::ChatFormat::ChatMl,
                                                                     kChatContext, model),
                    "two equal vocabularies and a ChatML model over the first build"))
        return;

    lpl::core::u8 *const block = arena.claim<lpl::core::u8>(kMindScratchBytes);
    if (!test.check(block != nullptr, "the scratch fits"))
        return;
    lpl::infer::TensorArena scratch{block, kMindScratchBytes};
    lpl::mind::ModelReasoner refused;
    lpl::mind::ModelReasoner accepted;

    test.check(!refused.initialise(arena, scratch, model, other, lpl::infer::paritySamplerSeed()),
               "a table of the same tokens held elsewhere is refused");
    test.check(accepted.initialise(arena, scratch, model, model.vocab(), lpl::infer::paritySamplerSeed()),
               "and the model's own is taken");
}
