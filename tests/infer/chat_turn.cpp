#include "chat_vocabulary.hpp"
#include "fixtures/smollm2_byte_pairs.hpp"
#include "model_image_writer.hpp"

#include <lpl/infer/Inference.hpp>
#include <lpl/infer/Model.hpp>
#include <lpl/infer/ModelImage.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/infer/TensorArena.hpp>
#include <lpl/infer/Tokenizer.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(chat_turn);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;
constexpr lpl::core::usize kTurnArenaBytes = 256u * 1024u;
constexpr lpl::core::usize kScratchBytes = 4u * 1024u;
constexpr lpl::core::usize kStopArenaBytes = 1024u * 1024u;
constexpr lpl::core::u32 kMostTurnIds = 128u;
constexpr lpl::core::u32 kTurnTokensOfTwoMessages = 5u; ///< Three tokens that open a turn and two that close one.
constexpr lpl::core::u32 kNoIndex = 0xFFFFFFFFu;

/**
 * @name The run a stop is judged on
 * @brief The parity weights over the parity vocabulary and three tokens more, fed the parity prompt, sampled from 64
 *        candidates at temperature one from seed 549: a seed found by trying seeds, under which the 28 tokens the
 *        Plain model generates hold the 122nd token at the third and the 124th at the tenth. The control of each test
 *        checks that it still does.
 * @{
 */
constexpr lpl::core::u32 kStopSamplerSeed = 549u;
constexpr lpl::core::u32 kStopCandidates = 64u;
constexpr lpl::core::u32 kStopRoom = 32u;
constexpr lpl::core::u32 kAppendedTokens = 3u;
/** @} */

/** Three tokens after the parity vocabulary: a filler where SmolLM2 has <|endoftext|>, then a turn's two tokens. */
constexpr const char *kTurnTokensWithoutEndOfText[kAppendedTokens] = {"<|filler|>", "<|im_start|>", "<|im_end|>"};

/** The three tokens of SmolLM2's turns after the parity vocabulary, in the order of their ids in SmolLM2. */
constexpr const char *kTurnTokens[kAppendedTokens] = {"<|endoftext|>", "<|im_start|>", "<|im_end|>"};

constexpr lpl::core::u32 kFirstAppended = lpl::infer::tests::kParityVocabularyTokens;       ///< 121.
constexpr lpl::core::u32 kSecondAppended = lpl::infer::tests::kParityVocabularyTokens + 1u; ///< 122.
constexpr lpl::core::u32 kThirdAppended = lpl::infer::tests::kParityVocabularyTokens + 2u;  ///< 123.

/**
 * @brief The fixture's tables read as a model reads them, a tokenizer bound to them with SmolLM2's pre-tokenizer, and
 *        the tokens of its turns.
 */
struct Fixture {
    lpl::infer::Vocab vocab{};           ///< The subset's tokens.
    lpl::infer::BytePairMerges merges{}; ///< The subset's merges.
    lpl::infer::Tokenizer tokenizer{};   ///< Bound to both.
    lpl::infer::ChatTokens tokens{};     ///< The tokens of a ChatML turn, found in the vocabulary.
    bool bound{false};                   ///< Whether every step fit and the tokens were found.
};

/**
 * @brief What generating from the parity prompt did.
 */
struct StopRun {
    lpl::core::u32 tokens[kStopRoom]{};    ///< The tokens generated.
    lpl::infer::GenerationReport report{}; ///< What the generation reported.
    bool ran{false};                       ///< Whether it ran.
};

/**
 * @brief Folds one word into an FNV-1a hash.
 */
void foldWord(lpl::core::u32 &hash, lpl::core::u32 word) { hash = (hash ^ word) * kFnv1aPrime; }

/**
 * @brief Reads the fixture's tables into @p arena, binds a tokenizer to them and finds the tokens of a ChatML turn.
 */
void readFixture(lpl::infer::TensorArena &arena, Fixture &fixture)
{
    fixture.bound = fixture.vocab.borrow(arena, smollm2_byte_pairs::kTokenText, smollm2_byte_pairs::kTokenTextBytes,
                                         smollm2_byte_pairs::kTokenEntries, smollm2_byte_pairs::kTokenCount) &&
                    fixture.merges.borrow(arena, smollm2_byte_pairs::kMerges, smollm2_byte_pairs::kMergeCount,
                                          fixture.vocab) == lpl::infer::BytePairMergesVerdict::Indexed &&
                    fixture.tokenizer.bind(arena, fixture.vocab, fixture.merges,
                                           lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel) &&
                    lpl::infer::findChatTokens(lpl::infer::ChatFormat::ChatMl, fixture.vocab, fixture.tokens);
}

/**
 * @brief Puts the @p index -th pair of the fixture in a turn, @p capacity ids at most, into @p ids.
 */
[[nodiscard]] lpl::infer::ChatTurnReport encodePair(lpl::infer::TensorArena &scratch, const Fixture &fixture,
                                                    lpl::core::u32 index, lpl::core::u32 *ids, lpl::core::u32 capacity)
{
    const smollm2_byte_pairs::ChatPair &pair = smollm2_byte_pairs::kChatPairs[index];

    scratch.reset();
    return lpl::infer::encodeChatTurn(scratch, fixture.tokenizer, fixture.tokens,
                                      smollm2_byte_pairs::kChatText + pair.systemOffset, pair.systemBytes,
                                      smollm2_byte_pairs::kChatText + pair.userOffset, pair.userBytes, ids, capacity);
}

/**
 * @brief The id in the model of the subset's token @p token, or kNoToken for a token past the subset.
 */
[[nodiscard]] lpl::core::u32 originalIdOf(lpl::core::u32 token)
{
    return token < smollm2_byte_pairs::kTokenCount ? smollm2_byte_pairs::kOriginalIds[token] : lpl::infer::kNoToken;
}

/**
 * @brief Whether the @p count ids at @p ids, numbered as the subset numbers them, are the @p count ids at @p expected,
 *        numbered as the model numbers them.
 */
[[nodiscard]] bool sameModelIds(const lpl::core::u32 *ids, lpl::core::u32 count, const lpl::core::u32 *expected)
{
    bool same = true;

    for (lpl::core::u32 at = 0u; same && at < count; ++at)
        same = originalIdOf(ids[at]) == expected[at];
    return same;
}

/**
 * @brief Whether the chat template, rendered and encoded by Hugging Face, gives the @p index -th pair the ids its turns
 *        give: a property of the fixture.
 */
[[nodiscard]] bool templateAgrees(lpl::core::u32 index)
{
    const smollm2_byte_pairs::ChatPair &pair = smollm2_byte_pairs::kChatPairs[index];
    bool same = pair.templateIdCount == pair.turnIdCount;

    for (lpl::core::u32 at = 0u; same && at < pair.turnIdCount; ++at)
        same = smollm2_byte_pairs::kTemplateIds[pair.firstTemplateId + at] ==
               smollm2_byte_pairs::kChatTurnIds[pair.firstTurnId + at];
    return same;
}

/**
 * @brief The ids among @p count at @p ids that are @p first, @p second or @p third.
 */
[[nodiscard]] lpl::core::u32 countAmong(const lpl::core::u32 *ids, lpl::core::u32 count, lpl::core::u32 first,
                                        lpl::core::u32 second, lpl::core::u32 third)
{
    lpl::core::u32 found = 0u;

    for (lpl::core::u32 at = 0u; at < count; ++at)
        found += ids[at] == first || ids[at] == second || ids[at] == third ? 1u : 0u;
    return found;
}

/**
 * @brief The first place of @p token among the tokens @p run generated, or kNoIndex.
 */
[[nodiscard]] lpl::core::u32 firstPlaceOf(const StopRun &run, lpl::core::u32 token)
{
    for (lpl::core::u32 at = 0u; at < run.report.generated; ++at)
        if (run.tokens[at] == token)
            return at;
    return kNoIndex;
}

/**
 * @brief Whether @p chat generated the first @p count tokens @p plain generated, and nothing else.
 */
[[nodiscard]] bool generatedThePrefix(const StopRun &chat, const StopRun &plain, lpl::core::u32 count)
{
    bool same = chat.ran && plain.ran && chat.report.generated == count && count <= plain.report.generated;

    for (lpl::core::u32 at = 0u; same && at < count; ++at)
        same = chat.tokens[at] == plain.tokens[at];
    return same;
}

/**
 * @brief Generates from the parity prompt with @p model, as the stop tests sample, into @p run.
 */
void generateFromTheParityPrompt(lpl::infer::TensorArena &arena, const lpl::infer::Model &model, StopRun &run)
{
    lpl::infer::Inference inference;
    if (!inference.initialise(arena, model, lpl::infer::CachePolicy::Refuse))
        return;

    const lpl::infer::Tokenizer tokenizer{model.vocab()};
    const char *const prompt = lpl::infer::parityPrompt();
    lpl::core::u32 promptBytes = 0u;
    while (prompt[promptBytes] != '\0')
        ++promptBytes;
    lpl::core::u32 promptIds[kStopRoom]{};
    lpl::core::u32 skipped = 0u;
    const lpl::core::u32 promptCount = tokenizer.encode(prompt, promptBytes, promptIds, kStopRoom, skipped);

    lpl::infer::GenerationParams params{};
    params.sampler.topK = kStopCandidates;
    params.sampler.seed = kStopSamplerSeed;
    params.maxTokens = model.config().contextLength - promptCount;
    run.ran = inference.generate(promptIds, promptCount, params, nullptr, run.tokens, kStopRoom, run.report);
}

/**
 * @brief Writes a ChatML model over the parity vocabulary and @p texts, reads it in place as @p chat, then, the image's
 *        chat format put at Plain, as @p plain, and generates from the parity prompt with each.
 */
void runChatAndPlain(lpl::infer::TensorArena &arena, const char *const *texts, StopRun &chat, StopRun &plain)
{
    lpl::infer::Vocab vocab;
    lpl::infer::Model written;
    lpl::core::u32 size = 0u;
    if (!lpl::infer::tests::buildParityVocabularyWith(arena, texts, kAppendedTokens, vocab) ||
        !lpl::infer::tests::synthesiseParityModelOver(arena, vocab, lpl::infer::ChatFormat::ChatMl,
                                                      lpl::infer::parityModelConfig().contextLength, written))
        return;
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, written, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    if (image == nullptr)
        return;

    lpl::infer::Model chatModel;
    lpl::infer::Model plainModel;
    const bool chatRead = chatModel.view(arena, image, size).verdict == lpl::infer::ModelReadVerdict::Read &&
                          chatModel.chatTokens().format == lpl::infer::ChatFormat::ChatMl;
    lpl::infer::tests::putChatFormatWord(image, static_cast<lpl::core::u32>(lpl::infer::ChatFormat::Plain));
    const bool plainRead = plainModel.view(arena, image, size).verdict == lpl::infer::ModelReadVerdict::Read &&
                           plainModel.chatTokens().format == lpl::infer::ChatFormat::Plain;
    if (!chatRead || !plainRead)
        return;
    generateFromTheParityPrompt(arena, chatModel, chat);
    generateFromTheParityPrompt(arena, plainModel, plain);
}

} // namespace

/**
 * @brief Each pair of messages SmolLM2's chat template agrees with is put in the ids the template gives it, as Hugging
 *        Face encodes the text it renders: <|im_start|> and <|im_end|> found by their text in the vocabulary, the
 *        second the token SmolLM2's generation ends on, and each turn's text, its role, newline and content, encoded
 *        whole by the model's merges.
 */
LPL_TEST(each_turn_is_put_in_the_ids_of_smollm2_s_chat_template)
{
    lpl::infer::TensorArena arena{kTurnArenaBytes};
    lpl::infer::TensorArena scratch{kScratchBytes};
    Fixture fixture{};
    lpl::core::u32 ids[kMostTurnIds]{};

    readFixture(arena, fixture);
    if (!test.check(fixture.bound, "the fixture's tables are read and the tokens of a ChatML turn found in them"))
        return;
    test.check(fixture.tokens.turnStart == smollm2_byte_pairs::kTurnStartToken &&
                   fixture.tokens.turnEnd == smollm2_byte_pairs::kTurnEndToken &&
                   fixture.tokens.endOfText == smollm2_byte_pairs::kEndOfTextToken,
               "<|im_start|>, <|im_end|> and <|endoftext|> are the vocabulary's own");
    test.check(originalIdOf(fixture.tokens.turnEnd) == smollm2_byte_pairs::kEndOfSequenceId,
               "the token that closes a turn is the one SmolLM2's generation ends on");

    lpl::core::u32 templatePairs = 0u;
    lpl::core::u32 agreeingWithTheTemplate = 0u;
    lpl::core::u32 agreeingWithTheTurns = 0u;
    lpl::core::u32 signature = kFnv1aOffsetBasis;
    for (lpl::core::u32 index = 0u; index < smollm2_byte_pairs::kChatPairCount; ++index)
    {
        const smollm2_byte_pairs::ChatPair &pair = smollm2_byte_pairs::kChatPairs[index];
        const lpl::infer::ChatTurnReport report = encodePair(scratch, fixture, index, ids, kMostTurnIds);
        const bool whole = report.verdict == lpl::infer::ChatTurnVerdict::Encoded && report.skipped == 0u;

        agreeingWithTheTurns +=
            whole && report.tokens == pair.turnIdCount &&
                    sameModelIds(ids, report.tokens, smollm2_byte_pairs::kChatTurnIds + pair.firstTurnId) ?
                1u :
                0u;
        if (templateAgrees(index))
        {
            ++templatePairs;
            agreeingWithTheTemplate +=
                whole && report.tokens == pair.templateIdCount &&
                        sameModelIds(ids, report.tokens, smollm2_byte_pairs::kTemplateIds + pair.firstTemplateId) ?
                    1u :
                    0u;
        }
        for (lpl::core::u32 at = 0u; at < report.tokens; ++at)
            foldWord(signature, originalIdOf(ids[at]));
    }
    test.check(smollm2_byte_pairs::kChatPairCount > 1u && templatePairs == smollm2_byte_pairs::kChatPairCount - 1u,
               "the fixture holds pairs, every one but one a pair the chat template agrees with");
    test.check(agreeingWithTheTemplate == templatePairs, "each of them is put in the ids the chat template gives it");
    test.check(agreeingWithTheTurns == smollm2_byte_pairs::kChatPairCount,
               "and every pair in the ids of its turns, each turn's text encoded whole");
    test.measure("pairs", smollm2_byte_pairs::kChatPairCount);
    test.measure("agreeing_with_the_template", agreeingWithTheTemplate);
    test.measureHexadecimal("turn_signature", signature);
}

/**
 * @brief A user message that holds the text of <|im_end|> and of <|im_start|> keeps it as text: the turn holds the
 *        three tokens that open a turn and the two that close one, no more, where Hugging Face, taking the text for the
 *        tokens, closes the user's turn inside the message and opens another.
 */
LPL_TEST(a_message_holding_the_text_of_a_turn_token_keeps_it_as_text)
{
    lpl::infer::TensorArena arena{kTurnArenaBytes};
    lpl::infer::TensorArena scratch{kScratchBytes};
    Fixture fixture{};
    lpl::core::u32 ids[kMostTurnIds]{};

    readFixture(arena, fixture);
    if (!test.check(fixture.bound, "the fixture's tables are read and the tokens of a ChatML turn found in them"))
        return;

    lpl::core::u32 hostile = kNoIndex;
    for (lpl::core::u32 index = 0u; index < smollm2_byte_pairs::kChatPairCount; ++index)
        hostile = !templateAgrees(index) ? index : hostile;
    if (!test.check(hostile != kNoIndex, "the fixture holds a pair the chat template disagrees with"))
        return;

    const smollm2_byte_pairs::ChatPair &pair = smollm2_byte_pairs::kChatPairs[hostile];
    const lpl::infer::ChatTurnReport report = encodePair(scratch, fixture, hostile, ids, kMostTurnIds);
    const lpl::core::u32 ours =
        countAmong(ids, report.tokens, fixture.tokens.turnStart, fixture.tokens.turnEnd, fixture.tokens.endOfText);
    const lpl::core::u32 theirs =
        countAmong(smollm2_byte_pairs::kTemplateIds + pair.firstTemplateId, pair.templateIdCount,
                   originalIdOf(smollm2_byte_pairs::kTurnStartToken), originalIdOf(smollm2_byte_pairs::kTurnEndToken),
                   originalIdOf(smollm2_byte_pairs::kEndOfTextToken));

    test.check(theirs > kTurnTokensOfTwoMessages,
               "Hugging Face takes the text in the message for turn tokens: more than five of them");
    test.check(report.verdict == lpl::infer::ChatTurnVerdict::Encoded && ours == kTurnTokensOfTwoMessages,
               "here the turn holds the five its format gives, none inside a message");
    test.check(report.tokens == pair.turnIdCount &&
                   sameModelIds(ids, report.tokens, smollm2_byte_pairs::kChatTurnIds + pair.firstTurnId),
               "and the message's text is encoded by the merges like any other");
    test.measure("turn_tokens_here", ours);
    test.measure("turn_tokens_of_the_template", theirs);
}

/**
 * @brief The tokens of a ChatML turn are found by their text: <|im_start|> and <|im_end|> are needed, <|endoftext|>
 *        is taken when the vocabulary has it; a Plain model has none, and no token of its vocabulary ends a turn.
 */
LPL_TEST(the_tokens_of_a_turn_are_found_by_their_text)
{
    lpl::infer::TensorArena arena{kTurnArenaBytes};
    lpl::infer::Vocab every;
    lpl::infer::Vocab withoutEndOfText;
    lpl::infer::Vocab withoutTurnEnd;
    lpl::infer::Vocab withoutTurnStart;
    const char *const turnStartOnly[1] = {"<|im_start|>"};
    const char *const turnEndOnly[1] = {"<|im_end|>"};
    lpl::infer::ChatTokens found{};
    lpl::infer::ChatTokens noEndOfText{};
    lpl::infer::ChatTokens plain{};
    lpl::infer::ChatTokens refused{};

    if (!test.check(lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens, kAppendedTokens, every) &&
                        lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens + 1u, 2u, withoutEndOfText) &&
                        lpl::infer::tests::buildParityVocabularyWith(arena, turnStartOnly, 1u, withoutTurnEnd) &&
                        lpl::infer::tests::buildParityVocabularyWith(arena, turnEndOnly, 1u, withoutTurnStart),
                    "four vocabularies build"))
        return;

    test.check(lpl::infer::findChatTokens(lpl::infer::ChatFormat::ChatMl, every, found) &&
                   found.format == lpl::infer::ChatFormat::ChatMl && found.endOfText == kFirstAppended &&
                   found.turnStart == kSecondAppended && found.turnEnd == kThirdAppended,
               "all three are found where they were appended: 121, 122 and 123");
    test.check(lpl::infer::findChatTokens(lpl::infer::ChatFormat::ChatMl, withoutEndOfText, noEndOfText) &&
                   noEndOfText.turnStart == kFirstAppended && noEndOfText.turnEnd == kSecondAppended &&
                   noEndOfText.endOfText == lpl::infer::kNoToken,
               "without <|endoftext|>, <|im_start|> and <|im_end|> are found at 121 and 122, and no end of text");
    test.check(!lpl::infer::findChatTokens(lpl::infer::ChatFormat::ChatMl, withoutTurnEnd, refused) &&
                   !lpl::infer::findChatTokens(lpl::infer::ChatFormat::ChatMl, withoutTurnStart, refused),
               "without <|im_end|>, or without <|im_start|>, a ChatML turn cannot be had");
    test.check(!lpl::infer::findChatTokens(static_cast<lpl::infer::ChatFormat>(2u), every, refused),
               "nor can a format this reader does not know");
    test.check(lpl::infer::findChatTokens(lpl::infer::ChatFormat::Plain, every, plain) &&
                   plain.format == lpl::infer::ChatFormat::Plain && plain.turnStart == lpl::infer::kNoToken &&
                   plain.turnEnd == lpl::infer::kNoToken && plain.endOfText == lpl::infer::kNoToken,
               "a Plain model has no turn tokens, though its vocabulary holds them");

    test.check(found.endsTurn(kThirdAppended) && found.endsTurn(kFirstAppended) && !found.endsTurn(kSecondAppended) &&
                   !found.endsTurn(kFirstAppended - 1u) && !found.endsTurn(0u),
               "<|im_end|> and <|endoftext|> end a turn, <|im_start|> and the others do not");
    test.check(noEndOfText.endsTurn(kSecondAppended) && !noEndOfText.endsTurn(kFirstAppended) &&
                   !noEndOfText.endsTurn(kThirdAppended),
               "without <|endoftext|>, <|im_end|> alone ends it");

    lpl::core::u32 endingPlain = 0u;
    for (lpl::core::u32 token = 0u; token < every.size(); ++token)
        endingPlain += plain.endsTurn(token) ? 1u : 0u;
    test.check(every.size() == kThirdAppended + 1u && endingPlain == 0u,
               "and no token of the vocabulary ends the turn of a Plain model");
}

/**
 * @brief A turn that holds more tokens than the room is refused as out of room, whether the room ends inside a text or
 *        before a turn token, and what is written is the turn's beginning; a turn that just fits is encoded. An arena
 *        that cannot hold a turn's text is refused as such, and a Plain model's tokens put nothing in a turn.
 */
LPL_TEST(a_turn_that_does_not_fit_is_refused_by_name)
{
    lpl::infer::TensorArena arena{kTurnArenaBytes};
    lpl::infer::TensorArena scratch{kScratchBytes};
    lpl::infer::TensorArena tiny{8u};
    Fixture fixture{};
    lpl::core::u32 whole[kMostTurnIds]{};
    lpl::core::u32 cut[kMostTurnIds]{};

    readFixture(arena, fixture);
    if (!test.check(fixture.bound, "the fixture's tables are read and the tokens of a ChatML turn found in them"))
        return;

    const lpl::infer::ChatTurnReport fits = encodePair(scratch, fixture, 0u, whole, kMostTurnIds);
    if (!test.check(fits.verdict == lpl::infer::ChatTurnVerdict::Encoded && fits.tokens > 4u,
                    "the first pair is put in a turn"))
        return;

    const lpl::infer::ChatTurnReport exact = encodePair(scratch, fixture, 0u, cut, fits.tokens);
    test.check(exact.verdict == lpl::infer::ChatTurnVerdict::Encoded && exact.tokens == fits.tokens,
               "in room for exactly its tokens, it is encoded");

    const lpl::infer::ChatTurnReport insideTheLastText = encodePair(scratch, fixture, 0u, cut, fits.tokens - 1u);
    bool prefix = insideTheLastText.tokens == fits.tokens - 1u;
    for (lpl::core::u32 at = 0u; prefix && at < insideTheLastText.tokens; ++at)
        prefix = cut[at] == whole[at];
    test.check(insideTheLastText.verdict == lpl::infer::ChatTurnVerdict::OutOfRoom && prefix,
               "one token short, inside the assistant's opening, it is out of room, its beginning written");

    lpl::core::u32 lastTurnStart = 0u;
    for (lpl::core::u32 at = 0u; at < fits.tokens; ++at)
        lastTurnStart = whole[at] == fixture.tokens.turnStart ? at : lastTurnStart;
    const lpl::infer::ChatTurnReport beforeATurnToken = encodePair(scratch, fixture, 0u, cut, lastTurnStart);
    test.check(lastTurnStart > 0u && beforeATurnToken.verdict == lpl::infer::ChatTurnVerdict::OutOfRoom &&
                   beforeATurnToken.tokens == lastTurnStart,
               "room that ends before the last <|im_start|> is out of room too");

    const smollm2_byte_pairs::ChatPair &pair = smollm2_byte_pairs::kChatPairs[0];
    const lpl::infer::ChatTurnReport cramped = lpl::infer::encodeChatTurn(
        tiny, fixture.tokenizer, fixture.tokens, smollm2_byte_pairs::kChatText + pair.systemOffset, pair.systemBytes,
        smollm2_byte_pairs::kChatText + pair.userOffset, pair.userBytes, cut, kMostTurnIds);
    test.check(cramped.verdict == lpl::infer::ChatTurnVerdict::ArenaExhausted,
               "8 bytes of arena cannot hold a turn's text");

    lpl::infer::ChatTokens plain{};
    (void) lpl::infer::findChatTokens(lpl::infer::ChatFormat::Plain, fixture.vocab, plain);
    scratch.reset();
    const lpl::infer::ChatTurnReport none = lpl::infer::encodeChatTurn(
        scratch, fixture.tokenizer, plain, smollm2_byte_pairs::kChatText + pair.systemOffset, pair.systemBytes,
        smollm2_byte_pairs::kChatText + pair.userOffset, pair.userBytes, cut, kMostTurnIds);
    test.check(none.verdict == lpl::infer::ChatTurnVerdict::NoTurns && none.tokens == 0u,
               "and a Plain model's tokens put nothing in a turn");
    test.measure("turn_tokens", fits.tokens);
}

/**
 * @brief A ChatML model generates the tokens the Plain model read from the same image generates, up to its first
 *        <|im_end|>, and stops there without keeping it, its end of turn reported; a token that is no turn token, where
 *        SmolLM2 has <|endoftext|>, does not stop it. The control: the Plain model goes past both.
 */
LPL_TEST(a_generation_stops_on_the_model_s_end_of_turn_without_keeping_it)
{
    lpl::infer::TensorArena arena{kStopArenaBytes};
    StopRun chat{};
    StopRun plain{};

    runChatAndPlain(arena, kTurnTokensWithoutEndOfText, chat, plain);
    const lpl::core::u32 filler = firstPlaceOf(plain, kFirstAppended);
    const lpl::core::u32 turnEnd = firstPlaceOf(plain, kThirdAppended);

    if (!test.check(plain.ran && filler != kNoIndex && turnEnd != kNoIndex && filler < turnEnd &&
                        turnEnd + 1u < plain.report.generated && !plain.report.endOfTurn,
                    "the control: the Plain model generates the filler, then <|im_end|>, and goes past both"))
        return;
    test.check(chat.report.endOfTurn && generatedThePrefix(chat, plain, turnEnd),
               "the ChatML model generates the same tokens up to <|im_end|>, and stops there without keeping it");
    test.check(chat.report.draws == turnEnd + 1u, "the token that ends the turn was drawn, then left out");
    test.measure("plain_generated", plain.report.generated);
    test.measure("chat_generated", chat.report.generated);
}

/**
 * @brief A ChatML model whose vocabulary has <|endoftext|> stops on it too: the same weights generate it before
 *        <|im_end|>, and the generation stops on it, without keeping it.
 */
LPL_TEST(a_generation_stops_on_the_model_s_end_of_text_too)
{
    lpl::infer::TensorArena arena{kStopArenaBytes};
    StopRun chat{};
    StopRun plain{};

    runChatAndPlain(arena, kTurnTokens, chat, plain);
    const lpl::core::u32 endOfText = firstPlaceOf(plain, kFirstAppended);
    const lpl::core::u32 turnEnd = firstPlaceOf(plain, kThirdAppended);

    if (!test.check(plain.ran && endOfText != kNoIndex && turnEnd != kNoIndex && endOfText < turnEnd &&
                        !plain.report.endOfTurn,
                    "the control: the Plain model generates <|endoftext|> before <|im_end|>, and goes past both"))
        return;
    test.check(chat.report.endOfTurn && generatedThePrefix(chat, plain, endOfText),
               "the ChatML model stops on <|endoftext|>, without keeping it");
    test.measure("chat_generated", chat.report.generated);
}
