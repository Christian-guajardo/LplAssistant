#include "fixtures/smollm2_byte_pairs.hpp"

#include <lpl/infer/Model.hpp>
#include <lpl/infer/ModelImage.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/infer/TensorArena.hpp>
#include <lpl/infer/Tokenizer.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(byte_pair_encoding);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;
constexpr lpl::core::usize kTokenizerArenaBytes = 160u * 1024u;
constexpr lpl::core::usize kImageArenaBytes = 1024u * 1024u;
constexpr lpl::core::u32 kMostIdsOfAPrompt = 256u;
constexpr lpl::core::u32 kShortTokenBytes = 16u; ///< The longest token LplAssistant 0.6.2 held.
constexpr lpl::core::u32 kCodePoints = 0x110000u;
constexpr lpl::core::u32 kTruncatedCapacity = 3u;
constexpr lpl::core::u32 kMadeUpTexts = 64u;
constexpr lpl::core::u32 kMadeUpTextBytes = 200u;
constexpr lpl::core::u32 kMadeUpSeed = 0x0105B9E5u;
constexpr lpl::core::u32 kLinearCongruentialMultiplier = 1664525u;
constexpr lpl::core::u32 kLinearCongruentialIncrement = 1013904223u;
constexpr lpl::core::u32 kDrawShift = 24u;
constexpr lpl::core::u32 kHalfShift = 16u;
constexpr lpl::core::u32 kPreTokenizerWordIndex = 68u; ///< Byte of the header's pre-tokenizer word.
constexpr lpl::core::u32 kMergeEntryBytes = 12u;
constexpr lpl::core::u32 kSegmentSignatureAt = 32u; ///< Byte of a table entry's signature.
constexpr lpl::core::u32 kMadeUpAlphabet = 3u;      ///< Single-byte tokens of the made-up merges: a to c.
constexpr lpl::core::u32 kMadeUpMergeCount = 200u;
constexpr lpl::core::u32 kMadeUpTokenCap = kMadeUpAlphabet + kMadeUpMergeCount;
constexpr lpl::core::u32 kMadeUpTokenMostBytes = 12u;
constexpr lpl::core::u32 kMadeUpMergeAttempts = 20000u;
constexpr lpl::core::u32 kMadeUpWords = 1000u;
constexpr lpl::core::u32 kMadeUpWordMostBytes = 64u;
constexpr lpl::core::usize kMadeUpArenaBytes = 512u * 1024u;

/** The made-up vocabulary of the refusals: a, b, ab, c and abc. */
constexpr char kMadeUpText[] = "ababcabc";
constexpr lpl::infer::VocabEntry kMadeUpEntries[] = {
    {0u, 1u},
    {1u, 1u},
    {0u, 2u},
    {4u, 1u},
    {5u, 3u}
};
constexpr lpl::core::u32 kMadeUpTokens = sizeof(kMadeUpEntries) / sizeof(kMadeUpEntries[0]);
constexpr lpl::core::u32 kA = 0u;
constexpr lpl::core::u32 kB = 1u;
constexpr lpl::core::u32 kAB = 2u;
constexpr lpl::core::u32 kC = 3u;
constexpr lpl::core::u32 kABC = 4u;

/**
 * @brief The fixture's tables, read as a model would read them: the vocabulary, its merges and a tokenizer bound to
 *        both with SmolLM2's pre-tokenizer.
 */
struct Fixture {
    lpl::infer::Vocab vocab{};           ///< The subset's tokens.
    lpl::infer::BytePairMerges merges{}; ///< The subset's merges.
    lpl::infer::Tokenizer tokenizer{};   ///< Bound to both.
    lpl::infer::BytePairMergesVerdict indexed{lpl::infer::BytePairMergesVerdict::ArenaExhausted}; ///< The indexing.
    bool bound{false}; ///< Whether every step fit.
};

/**
 * @brief What encoding every prompt of the fixture found.
 */
struct PromptRun {
    lpl::core::u32 agreeing{0u};                 ///< Prompts whose ids, as the model numbers them, are the reference's.
    lpl::core::u32 skippedAsCounted{0u};         ///< Prompts whose skipped bytes are the fixture's unspellable bytes.
    lpl::core::u32 ids{0u};                      ///< Ids written over every prompt.
    lpl::core::u32 longestTokenBytes{0u};        ///< The longest token written.
    lpl::core::u32 signature{kFnv1aOffsetBasis}; ///< FNV-1a fold of every id written, as the model numbers them.
};

/**
 * @brief Folds one word into an FNV-1a hash.
 */
void foldWord(lpl::core::u32 &hash, lpl::core::u32 word) { hash = (hash ^ word) * kFnv1aPrime; }

/**
 * @brief Reads the fixture's tables into @p arena and binds a tokenizer to them.
 */
void readFixture(lpl::infer::TensorArena &arena, Fixture &fixture)
{
    if (!fixture.vocab.borrow(arena, smollm2_byte_pairs::kTokenText, smollm2_byte_pairs::kTokenTextBytes,
                              smollm2_byte_pairs::kTokenEntries, smollm2_byte_pairs::kTokenCount))
        return;
    fixture.indexed =
        fixture.merges.borrow(arena, smollm2_byte_pairs::kMerges, smollm2_byte_pairs::kMergeCount, fixture.vocab);
    fixture.bound = fixture.indexed == lpl::infer::BytePairMergesVerdict::Indexed &&
                    fixture.tokenizer.bind(arena, fixture.vocab, fixture.merges,
                                           lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel);
}

/**
 * @brief Encodes every prompt with @p tokenizer and compares the ids, as the model numbers them, with the reference.
 */
[[nodiscard]] PromptRun encodeEveryPrompt(const lpl::infer::Tokenizer &tokenizer, const lpl::infer::Vocab &vocab,
                                          lpl::core::u32 *ids)
{
    PromptRun run{};

    for (lpl::core::u32 index = 0u; index < smollm2_byte_pairs::kPromptCount; ++index)
    {
        const smollm2_byte_pairs::Prompt &prompt = smollm2_byte_pairs::kPrompts[index];
        lpl::core::u32 skipped = 0u;
        const lpl::core::u32 count = tokenizer.encode(smollm2_byte_pairs::kPromptText + prompt.textOffset,
                                                      prompt.textBytes, ids, kMostIdsOfAPrompt, skipped);
        bool agrees = count == prompt.idCount;

        for (lpl::core::u32 at = 0u; at < count; ++at)
        {
            const lpl::core::u32 original = ids[at] < smollm2_byte_pairs::kTokenCount ?
                                                smollm2_byte_pairs::kOriginalIds[ids[at]] :
                                                lpl::infer::kNoToken;
            lpl::core::u32 length = 0u;

            (void) vocab.text(ids[at], length);
            run.longestTokenBytes = length > run.longestTokenBytes ? length : run.longestTokenBytes;
            agrees = agrees && original == smollm2_byte_pairs::kReferenceIds[prompt.firstId + at];
            foldWord(run.signature, original);
        }
        run.ids += count;
        run.agreeing += agrees ? 1u : 0u;
        run.skippedAsCounted += skipped == prompt.unspellable ? 1u : 0u;
    }
    return run;
}

/**
 * @brief The bytes of the vocabulary @p tokens stand for are @p text with the bytes @p unspellable says no single-byte
 *        token spells removed.
 */
[[nodiscard]] bool decodesToTheSpellableBytes(const lpl::infer::Tokenizer &tokenizer, const lpl::core::u32 *tokens,
                                              lpl::core::u32 count, const char *text, lpl::core::u32 bytes,
                                              const bool *unspellable, char *decoded)
{
    const lpl::core::u32 written = tokenizer.decode(tokens, count, decoded, bytes);
    lpl::core::u32 at = 0u;

    for (lpl::core::u32 i = 0u; i < bytes; ++i)
    {
        if (unspellable[static_cast<lpl::core::u8>(text[i])])
            continue;
        if (at >= written || decoded[at] != text[i])
            return false;
        ++at;
    }
    return at == written;
}

/**
 * @brief Whether no single-byte token of @p vocab spells each byte value, into @p out.
 */
void markUnspellableBytes(const lpl::infer::Vocab &vocab, bool *out)
{
    for (lpl::core::u32 value = 0u; value < 256u; ++value)
    {
        const char byte = static_cast<char>(value);
        out[value] = vocab.find(&byte, 1u) == lpl::infer::kNoToken;
    }
}

/**
 * @brief The next byte of a made-up text: any of the 256 values, so invalid UTF-8 and bytes without a token come up.
 */
[[nodiscard]] char madeUpByte(lpl::core::u32 &state)
{
    state = state * kLinearCongruentialMultiplier + kLinearCongruentialIncrement;
    return static_cast<char>(state >> kDrawShift);
}

/**
 * @brief Writes @p model's image into @p arena.
 * @return The image, or nullptr when the writer refused or the arena is short; @p size receives its bytes.
 */
[[nodiscard]] lpl::core::u8 *writeImage(lpl::infer::TensorArena &arena, const lpl::infer::Model &model,
                                        lpl::core::u32 &size)
{
    size = model.blobBytes();
    lpl::core::u32 *const words = size != 0u ? arena.claim<lpl::core::u32>(size / 4u + 1u) : nullptr;
    lpl::core::u8 *const image = reinterpret_cast<lpl::core::u8 *>(words);

    return image != nullptr && model.writeBlob(image, size) == size ? image : nullptr;
}

/**
 * @brief Whether @p merges hold the fixture's merges, in order.
 */
[[nodiscard]] bool holdTheFixtureMerges(const lpl::infer::BytePairMerges &merges)
{
    bool same = merges.size() == smollm2_byte_pairs::kMergeCount && merges.merges() != nullptr;

    for (lpl::core::u32 rank = 0u; same && rank < merges.size(); ++rank)
        same = merges.merges()[rank].left == smollm2_byte_pairs::kMerges[rank].left &&
               merges.merges()[rank].right == smollm2_byte_pairs::kMerges[rank].right &&
               merges.merges()[rank].result == smollm2_byte_pairs::kMerges[rank].result;
    return same;
}

/**
 * @brief Whether @p lhs and @p rhs are the same null-terminated text.
 */
[[nodiscard]] bool sameText(const char *lhs, const char *rhs)
{
    while (*lhs != '\0' && *lhs == *rhs)
    {
        ++lhs;
        ++rhs;
    }
    return *lhs == *rhs;
}

/**
 * @brief Writes @p value little-endian at @p offset of @p bytes.
 */
void putWordAt(lpl::core::u8 *bytes, lpl::core::u64 offset, lpl::core::u32 value)
{
    for (lpl::core::u32 i = 0u; i < 4u; ++i)
        bytes[offset + i] = static_cast<lpl::core::u8>((value >> (8u * i)) & 0xFFu);
}

/**
 * @brief The verdict of indexing @p merges over the made-up vocabulary, and whether a refused table holds nothing.
 */
[[nodiscard]] lpl::infer::BytePairMergesVerdict indexMadeUp(const lpl::infer::BytePairMerge *merges,
                                                            lpl::core::u32 count, lpl::core::usize arenaBytes,
                                                            bool &emptyWhenRefused)
{
    lpl::infer::TensorArena arena{arenaBytes};
    lpl::infer::TensorArena vocabularyArena{1024u};
    lpl::infer::Vocab vocab;
    lpl::infer::BytePairMerges table;

    emptyWhenRefused =
        vocab.borrow(vocabularyArena, kMadeUpText, sizeof(kMadeUpText) - 1u, kMadeUpEntries, kMadeUpTokens);
    const lpl::infer::BytePairMergesVerdict verdict = table.borrow(arena, merges, count, vocab);
    if (verdict != lpl::infer::BytePairMergesVerdict::Indexed)
        emptyWhenRefused = emptyWhenRefused && table.size() == 0u && table.rank(kA, kB) == lpl::infer::kNoMerge;
    return verdict;
}

/**
 * @struct MadeUpMerges
 * @brief Merges the test made up over the letters a to c, each joining two tokens it made before into a new one, and a
 *        table of every pair's rank worked apart from the index under test.
 */
struct MadeUpMerges {
    char *text{nullptr};                        ///< The tokens' bytes, one after the other.
    lpl::infer::VocabEntry *entries{nullptr};   ///< Where each token's bytes lie.
    lpl::infer::BytePairMerge *merges{nullptr}; ///< The merges, in the order the test made them: their ranks.
    lpl::core::u32 *rankOf{nullptr};            ///< kMadeUpTokenCap squared: the rank of each pair, or kNoMerge.
    lpl::core::u32 tokens{0u};                  ///< Tokens made.
    lpl::core::u32 mergeCount{0u};              ///< Merges made.
    lpl::core::u32 textBytes{0u};               ///< Bytes of text.
};

/**
 * @brief The next draw of a linear congruential generator, its high half.
 */
[[nodiscard]] lpl::core::u32 nextDraw(lpl::core::u32 &state)
{
    state = state * kLinearCongruentialMultiplier + kLinearCongruentialIncrement;
    return state >> kHalfShift;
}

/**
 * @brief Whether a token of @p table already spells the @p length bytes at @p bytes.
 */
[[nodiscard]] bool spelt(const MadeUpMerges &table, const char *bytes, lpl::core::u32 length)
{
    for (lpl::core::u32 token = 0u; token < table.tokens; ++token)
    {
        bool same = table.entries[token].length == length;
        for (lpl::core::u32 i = 0u; same && i < length; ++i)
            same = table.text[table.entries[token].offset + i] == bytes[i];
        if (same)
            return true;
    }
    return false;
}

/**
 * @brief Makes up to kMadeUpMergeCount merges in @p arena: the three letters, then pairs of tokens drawn at random
 * whose bytes no token spells yet, at most kMadeUpTokenMostBytes long.
 * @return Whether the arena held them.
 */
[[nodiscard]] bool makeUpMerges(lpl::infer::TensorArena &arena, MadeUpMerges &table)
{
    table.text = arena.claim<char>(kMadeUpTokenCap * kMadeUpTokenMostBytes);
    table.entries = arena.claim<lpl::infer::VocabEntry>(kMadeUpTokenCap);
    table.merges = arena.claim<lpl::infer::BytePairMerge>(kMadeUpMergeCount);
    table.rankOf = arena.claim<lpl::core::u32>(kMadeUpTokenCap * kMadeUpTokenCap);
    if (table.text == nullptr || table.entries == nullptr || table.merges == nullptr || table.rankOf == nullptr)
        return false;

    for (lpl::core::u32 pair = 0u; pair < kMadeUpTokenCap * kMadeUpTokenCap; ++pair)
        table.rankOf[pair] = lpl::infer::kNoMerge;
    for (; table.tokens < kMadeUpAlphabet; ++table.tokens)
    {
        table.text[table.tokens] = static_cast<char>('a' + table.tokens);
        table.entries[table.tokens] = lpl::infer::VocabEntry{table.tokens, 1u};
    }
    table.textBytes = kMadeUpAlphabet;

    lpl::core::u32 state = kMadeUpSeed;
    for (lpl::core::u32 attempt = 0u; attempt < kMadeUpMergeAttempts && table.mergeCount < kMadeUpMergeCount; ++attempt)
    {
        const lpl::core::u32 left = nextDraw(state) % table.tokens;
        const lpl::core::u32 right = nextDraw(state) % table.tokens;
        const lpl::infer::VocabEntry leftEntry = table.entries[left];
        const lpl::infer::VocabEntry rightEntry = table.entries[right];
        const lpl::core::u32 length = leftEntry.length + rightEntry.length;
        if (length > kMadeUpTokenMostBytes || table.rankOf[left * kMadeUpTokenCap + right] != lpl::infer::kNoMerge)
            continue;

        char *const bytes = table.text + table.textBytes;
        for (lpl::core::u32 i = 0u; i < leftEntry.length; ++i)
            bytes[i] = table.text[leftEntry.offset + i];
        for (lpl::core::u32 i = 0u; i < rightEntry.length; ++i)
            bytes[leftEntry.length + i] = table.text[rightEntry.offset + i];
        if (spelt(table, bytes, length))
            continue;

        table.entries[table.tokens] = lpl::infer::VocabEntry{table.textBytes, length};
        table.merges[table.mergeCount] = lpl::infer::BytePairMerge{left, right, table.tokens};
        table.rankOf[left * kMadeUpTokenCap + right] = table.mergeCount;
        table.textBytes += length;
        ++table.tokens;
        ++table.mergeCount;
    }
    return true;
}

/**
 * @brief Merges @p count symbols the plain way, worked apart from the encoder: over and over, the adjacent pair of the
 *        lowest rank, the leftmost among equals, until no adjacent pair is a merge.
 * @return Merges applied; @p count receives the symbols left.
 */
lpl::core::u32 mergeLowestRankFirst(const MadeUpMerges &table, lpl::core::u32 *symbols, lpl::core::u32 &count)
{
    lpl::core::u32 applied = 0u;
    for (;;)
    {
        lpl::core::u32 lowest = lpl::infer::kNoMerge;
        lpl::core::u32 at = 0u;
        for (lpl::core::u32 i = 0u; i + 1u < count; ++i)
        {
            const lpl::core::u32 rank = table.rankOf[symbols[i] * kMadeUpTokenCap + symbols[i + 1u]];
            if (rank < lowest)
            {
                lowest = rank;
                at = i;
            }
        }
        if (lowest == lpl::infer::kNoMerge)
            return applied;

        symbols[at] = table.merges[lowest].result;
        for (lpl::core::u32 i = at + 1u; i + 1u < count; ++i)
            symbols[i] = symbols[i + 1u];
        --count;
        ++applied;
    }
}

/**
 * @struct InvalidCase
 * @brief A sequence of UTF-8 put between "ab" and "cd", and the pieces the five or more bytes make.
 */
struct InvalidCase {
    lpl::core::u8 bytes[4]; ///< The sequence.
    lpl::core::u32 length;  ///< Its bytes.
    lpl::core::u32 pieces;  ///< 3 when it breaks the letters around it, 1 when it is a letter.
};

/**
 * @brief Whether SmolLM2's pre-tokenizer splits "ab", then @p sample, then "cd" as @p sample says.
 */
[[nodiscard]] bool splitsAsExpected(const InvalidCase &sample)
{
    char text[8] = {'a', 'b'};
    for (lpl::core::u32 i = 0u; i < sample.length; ++i)
        text[2u + i] = static_cast<char>(sample.bytes[i]);
    text[2u + sample.length] = 'c';
    text[3u + sample.length] = 'd';
    const lpl::core::u32 bytes = 4u + sample.length;

    lpl::core::u32 pieces[4] = {0u, 0u, 0u, 0u};
    lpl::core::u32 count = 0u;
    for (lpl::core::u32 at = 0u; at < bytes && count < 4u; ++count)
    {
        pieces[count] = lpl::infer::preTokenizedPieceBytes(lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel,
                                                           text, bytes, at);
        at += pieces[count] != 0u ? pieces[count] : bytes;
    }
    if (sample.pieces == 1u)
        return count == 1u && pieces[0] == bytes;
    return count == 3u && pieces[0] == 2u && pieces[1] == sample.length && pieces[2] == 2u;
}

} // namespace

/**
 * @brief Every prompt of the fixture encodes to the ids SmolLM2's own tokenizer gives it, its bytes without a token
 *        counted as skipped; a token longer than LplAssistant 0.6.2's sixteen bytes comes out; and longest match over
 *        the same vocabulary gives other ids, so the merging is what agrees.
 */
LPL_TEST(the_prompts_encode_to_the_ids_of_smollm2_s_own_tokenizer)
{
    lpl::infer::TensorArena arena{kTokenizerArenaBytes};
    Fixture fixture{};
    lpl::core::u32 *const ids = arena.claim<lpl::core::u32>(kMostIdsOfAPrompt);
    const lpl::core::u32 cutBefore = lpl::infer::bytePairWordsCut();

    readFixture(arena, fixture);
    if (!test.check(fixture.bound && ids != nullptr, "the fixture's tables are read and a tokenizer is bound to them"))
        return;

    const PromptRun merged = encodeEveryPrompt(fixture.tokenizer, fixture.vocab, ids);
    const PromptRun matched = encodeEveryPrompt(lpl::infer::Tokenizer{fixture.vocab}, fixture.vocab, ids);

    test.check(smollm2_byte_pairs::kPromptCount > 0u && merged.ids > 0u, "the prompts are not empty");
    test.check(merged.agreeing == smollm2_byte_pairs::kPromptCount, "every prompt encodes to the reference's ids");
    test.check(merged.skippedAsCounted == smollm2_byte_pairs::kPromptCount,
               "and its bytes without a token are skipped and counted");
    test.check(merged.longestTokenBytes == smollm2_byte_pairs::kLongestEncodedTokenBytes &&
                   merged.longestTokenBytes > kShortTokenBytes,
               "a token longer than sixteen bytes comes out, the longest the reference gives");
    test.check(matched.agreeing < smollm2_byte_pairs::kPromptCount,
               "longest match over the same tokens disagrees with the reference on some prompt");
    test.check(lpl::infer::bytePairWordsCut() == cutBefore, "no word is cut");
    test.measure("prompts", smollm2_byte_pairs::kPromptCount);
    test.measure("agreeing", merged.agreeing);
    test.measure("ids", merged.ids);
    test.measure("longest_token_bytes", merged.longestTokenBytes);
    test.measure("agreeing_by_longest_match", matched.agreeing);
    test.measureHexadecimal("id_signature", merged.signature);
}

/**
 * @brief SmolLM2's pre-tokenizer splits every prompt into the words Hugging Face's splits it into.
 */
LPL_TEST(the_pre_tokenizer_splits_each_prompt_into_smollm2_s_words)
{
    lpl::core::u32 agreeing = 0u;
    lpl::core::u32 words = 0u;
    lpl::core::u32 signature = kFnv1aOffsetBasis;

    for (lpl::core::u32 index = 0u; index < smollm2_byte_pairs::kPromptCount; ++index)
    {
        const smollm2_byte_pairs::Prompt &prompt = smollm2_byte_pairs::kPrompts[index];
        const char *const text = smollm2_byte_pairs::kPromptText + prompt.textOffset;
        lpl::core::u32 at = 0u;
        lpl::core::u32 word = 0u;
        bool agrees = true;

        while (at < prompt.textBytes)
        {
            const lpl::core::u32 piece = lpl::infer::preTokenizedPieceBytes(
                lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel, text, prompt.textBytes, at);
            agrees = agrees && piece != 0u && word < prompt.wordCount &&
                     piece == smollm2_byte_pairs::kReferenceWordBytes[prompt.firstWord + word];
            foldWord(signature, piece);
            if (piece == 0u)
                break;
            at += piece;
            ++word;
        }
        words += word;
        agreeing += agrees && word == prompt.wordCount ? 1u : 0u;
    }

    test.check(words > 0u, "the prompts are not empty");
    test.check(agreeing == smollm2_byte_pairs::kPromptCount, "every prompt splits into the reference's words");
    test.measure("words", words);
    test.measureHexadecimal("word_signature", signature);
}

/**
 * @brief The class of every code point, folded, is the Unicode Character Database 16.0.0's, worked apart; a letter and
 *        a number new in 16.0 are classed so, a letter new in 17.0 is not, U+001C is no space though Python's
 *        str.isspace says it is, and a value past U+10FFFF is other.
 */
LPL_TEST(the_class_of_every_code_point_is_the_unicode_database_s)
{
    lpl::core::u32 signature = kFnv1aOffsetBasis;
    lpl::core::u32 counts[4] = {0u, 0u, 0u, 0u};

    for (lpl::core::u32 codePoint = 0u; codePoint < kCodePoints; ++codePoint)
    {
        const lpl::core::u32 value = static_cast<lpl::core::u32>(lpl::infer::characterClassOf(codePoint));
        foldWord(signature, value);
        ++counts[value & 3u];
    }

    test.check(signature == smollm2_byte_pairs::kCharacterClassSignature,
               "every code point has the class the database gives it");
    test.check(counts[1] == smollm2_byte_pairs::kLetterCodePoints &&
                   counts[2] == smollm2_byte_pairs::kNumberCodePoints &&
                   counts[3] == smollm2_byte_pairs::kSpaceCodePoints,
               "as many letters, numbers and spaces as it holds");
    test.check(lpl::infer::characterClassOf(0x1C89u) == lpl::infer::CharacterClass::Letter &&
                   lpl::infer::characterClassOf(0x16D70u) == lpl::infer::CharacterClass::Number &&
                   lpl::infer::characterClassOf(0x10940u) == lpl::infer::CharacterClass::Other,
               "a letter and a number of Unicode 16.0 are one, a letter of 17.0 is not yet");
    test.check(lpl::infer::characterClassOf(0x1Cu) == lpl::infer::CharacterClass::Other &&
                   lpl::infer::characterClassOf(0x3000u) == lpl::infer::CharacterClass::Space &&
                   lpl::infer::characterClassOf(0x110000u) == lpl::infer::CharacterClass::Other,
               "U+001C is no space, U+3000 is one, and past U+10FFFF is other");
    test.measure("letters", counts[1]);
    test.measure("numbers", counts[2]);
    test.measure("spaces", counts[3]);
    test.measureHexadecimal("class_signature", signature);
}

/**
 * @brief The index finds every merge of the fixture at its rank and no other pair of its tokens.
 */
LPL_TEST(the_index_finds_every_merge_at_its_rank_and_no_other_pair)
{
    lpl::infer::TensorArena arena{kTokenizerArenaBytes};
    Fixture fixture{};

    readFixture(arena, fixture);
    if (!test.check(fixture.indexed == lpl::infer::BytePairMergesVerdict::Indexed, "the fixture's merges are indexed"))
        return;

    lpl::core::u32 found = 0u;
    for (lpl::core::u32 rank = 0u; rank < smollm2_byte_pairs::kMergeCount; ++rank)
        found += fixture.merges.rank(smollm2_byte_pairs::kMerges[rank].left, smollm2_byte_pairs::kMerges[rank].right) ==
                         rank ?
                     1u :
                     0u;

    lpl::core::u32 pairs = 0u;
    lpl::core::u32 ranked = 0u;
    for (lpl::core::u32 left = 0u; left < smollm2_byte_pairs::kTokenCount; ++left)
    {
        for (lpl::core::u32 right = 0u; right < smollm2_byte_pairs::kTokenCount; ++right)
        {
            ranked += fixture.merges.rank(left, right) != lpl::infer::kNoMerge ? 1u : 0u;
            ++pairs;
        }
    }

    test.check(found == smollm2_byte_pairs::kMergeCount, "every merge is found at its rank");
    test.check(ranked == smollm2_byte_pairs::kMergeCount, "and of every pair of tokens, only the merges have a rank");
    test.check(fixture.merges.rank(smollm2_byte_pairs::kTokenCount, 0u) == lpl::infer::kNoMerge,
               "a token past the vocabulary has none");
    test.measure("pairs", pairs);
    test.measure("ranked", ranked);
}

/**
 * @brief A merge that names a token past the vocabulary, one whose result is not its pair's bytes, a pair merged twice
 *        and an arena too small are each refused by name, the table then holding no merge; no merge is indexed too.
 */
LPL_TEST(a_merge_list_that_does_not_hold_is_refused_by_name)
{
    const lpl::infer::BytePairMerge valid[] = {
        {kA,  kB, kAB },
        {kAB, kC, kABC}
    };
    const lpl::infer::BytePairMerge unknown[] = {
        {kA,  kB,            kAB },
        {kAB, kMadeUpTokens, kABC}
    };
    const lpl::infer::BytePairMerge notTheConcatenation[] = {
        {kA,  kB, kAB},
        {kAB, kC, kC }
    };
    const lpl::infer::BytePairMerge repeated[] = {
        {kA, kB, kAB},
        {kA, kB, kAB}
    };
    const lpl::infer::BytePairMerge swapped[] = {
        {kB, kA, kAB}
    };
    const lpl::infer::BytePairMerge otherLeft[] = {
        {kC, kB, kAB}
    };
    const lpl::infer::BytePairMerge otherRight[] = {
        {kA, kC, kAB}
    };
    bool emptyValid = false;
    bool emptyUnknown = false;
    bool emptyConcatenation = false;
    bool emptyRepeated = false;
    bool emptyExhausted = false;
    bool emptyNone = false;

    test.check(indexMadeUp(valid, 2u, 1024u, emptyValid) == lpl::infer::BytePairMergesVerdict::Indexed && emptyValid,
               "two merges that hold are indexed");
    test.check(indexMadeUp(unknown, 2u, 1024u, emptyUnknown) == lpl::infer::BytePairMergesVerdict::UnknownToken &&
                   emptyUnknown,
               "a token past the vocabulary is refused by name, the table empty");
    test.check(indexMadeUp(notTheConcatenation, 2u, 1024u, emptyConcatenation) ==
                       lpl::infer::BytePairMergesVerdict::NotTheConcatenation &&
                   emptyConcatenation,
               "a result that is not its pair's bytes is refused by name, the table empty");
    test.check(indexMadeUp(swapped, 1u, 1024u, emptyConcatenation) ==
                       lpl::infer::BytePairMergesVerdict::NotTheConcatenation &&
                   emptyConcatenation,
               "and so is a result of the right length whose bytes are the pair's the other way round");
    test.check(indexMadeUp(otherLeft, 1u, 1024u, emptyConcatenation) ==
                       lpl::infer::BytePairMergesVerdict::NotTheConcatenation &&
                   indexMadeUp(otherRight, 1u, 1024u, emptyConcatenation) ==
                       lpl::infer::BytePairMergesVerdict::NotTheConcatenation &&
                   emptyConcatenation,
               "or whose bytes differ from the left token's alone, or from the right token's alone");
    test.check(indexMadeUp(repeated, 2u, 1024u, emptyRepeated) == lpl::infer::BytePairMergesVerdict::RepeatedPair &&
                   emptyRepeated,
               "a pair merged twice is refused by name, the table empty");
    test.check(indexMadeUp(valid, 2u, 8u, emptyExhausted) == lpl::infer::BytePairMergesVerdict::ArenaExhausted &&
                   emptyExhausted,
               "an arena too small for the index is refused by name, the table empty");
    test.check(indexMadeUp(nullptr, 0u, 8u, emptyNone) == lpl::infer::BytePairMergesVerdict::Indexed && emptyNone,
               "no merge is indexed, claiming nothing");

    lpl::infer::TensorArena arena{2048u};
    lpl::infer::Vocab vocab;
    lpl::infer::BytePairMerges table;
    const bool indexed = vocab.borrow(arena, kMadeUpText, sizeof(kMadeUpText) - 1u, kMadeUpEntries, kMadeUpTokens) &&
                         table.borrow(arena, valid, 2u, vocab) == lpl::infer::BytePairMergesVerdict::Indexed &&
                         table.rank(kA, kB) == 0u;
    const bool refusedAfter =
        table.borrow(arena, repeated, 2u, vocab) == lpl::infer::BytePairMergesVerdict::RepeatedPair;
    test.check(indexed && refusedAfter && table.size() == 0u && table.rank(kA, kB) == lpl::infer::kNoMerge,
               "a table indexed then refused holds no merge, not the ones it held");
}

/**
 * @brief A byte that starts no valid sequence of UTF-8 is a character of class other, so it breaks a run of letters,
 *        whether it leads an overlong form, a surrogate, a code point past U+10FFFF or a sequence cut short, or is a
 *        lone continuation or a byte UTF-8 never uses; a letter of two bytes keeps the run whole.
 */
LPL_TEST(an_invalid_byte_of_utf_8_breaks_a_run_of_letters)
{
    const InvalidCase cases[] = {
        {{0xE0u, 0x81u, 0x81u, 0u},    3u, 3u},
        {{0xF0u, 0x80u, 0x81u, 0x81u}, 4u, 3u},
        {{0xC1u, 0x81u, 0u, 0u},       2u, 3u},
        {{0xEDu, 0xA0u, 0x80u, 0u},    3u, 3u},
        {{0xF4u, 0x90u, 0x80u, 0x80u}, 4u, 3u},
        {{0xC3u, 0u, 0u, 0u},          1u, 3u},
        {{0xE2u, 0x82u, 0u, 0u},       2u, 3u},
        {{0x80u, 0u, 0u, 0u},          1u, 3u},
        {{0xFFu, 0u, 0u, 0u},          1u, 3u},
        {{0xC3u, 0xA9u, 0u, 0u},       2u, 1u},
    };
    const lpl::core::u32 caseCount = sizeof(cases) / sizeof(cases[0]);
    lpl::core::u32 agreeing = 0u;

    for (lpl::core::u32 index = 0u; index < caseCount; ++index)
        agreeing += splitsAsExpected(cases[index]) ? 1u : 0u;

    test.check(agreeing == caseCount, "each invalid sequence breaks the letters around it, and the letter does not");
    test.measure("cases", caseCount);
}

/**
 * @brief A model carries its merges and its pre-tokenizer through its image, read into an arena and in place: the
 *        table names them, the merges come back in order, and the tokenizer bound to the model read in place encodes
 *        every prompt as the reference does.
 */
LPL_TEST(an_image_carries_its_merges_and_pre_tokenizer)
{
    lpl::infer::TensorArena arena{kImageArenaBytes};
    Fixture fixture{};
    lpl::infer::Model model;
    lpl::core::u32 size = 0u;

    readFixture(arena, fixture);
    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();
    shape.vocabSize = smollm2_byte_pairs::kTokenCount;
    const bool synthesised = fixture.bound && model.synthesise(arena, shape, fixture.vocab, fixture.merges,
                                                               lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel,
                                                               lpl::infer::parityModelSeed());
    lpl::core::u8 *const image = synthesised ? writeImage(arena, model, size) : nullptr;
    if (!test.check(image != nullptr, "a model over the fixture's tables writes its image"))
        return;

    lpl::infer::ModelSegment segments[16];
    const lpl::infer::ModelImageInspection inspection = lpl::infer::inspectModelImage(image, size, segments, 16u);
    if (!test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                        inspection.header.segmentCount >= 2u,
                    "and its header and table are accepted"))
        return;

    const lpl::infer::ModelSegment &last = segments[inspection.header.segmentCount - 1u];
    lpl::infer::Model copied;
    lpl::infer::Model viewed;
    const lpl::infer::ModelReadReport copy = copied.copy(arena, image, size);
    const lpl::infer::ModelReadReport view = viewed.view(arena, image, size);
    lpl::infer::Tokenizer tokenizer;
    lpl::core::u32 *const ids = arena.claim<lpl::core::u32>(kMostIdsOfAPrompt);
    const bool bound = view.verdict == lpl::infer::ModelReadVerdict::Read && ids != nullptr &&
                       tokenizer.bind(arena, viewed.vocab(), viewed.merges(), viewed.preTokenizer());
    const PromptRun run = bound ? encodeEveryPrompt(tokenizer, viewed.vocab(), ids) : PromptRun{};

    test.check(inspection.header.mergeCount == smollm2_byte_pairs::kMergeCount &&
                   inspection.header.preTokenizer == lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel,
               "the header gives the merge count and the pre-tokenizer");
    test.check(last.role == lpl::infer::ModelSegmentRole::Merges &&
                   last.length == static_cast<lpl::core::u64>(smollm2_byte_pairs::kMergeCount) * kMergeEntryBytes &&
                   last.offset >= segments[inspection.header.segmentCount - 2u].offset,
               "the merges are the last segment, twelve bytes each, after the experts");
    test.check(copy.verdict == lpl::infer::ModelReadVerdict::Read && holdTheFixtureMerges(copied.merges()) &&
                   copied.preTokenizer() == lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel,
               "read into an arena, the model has the merges in order and the pre-tokenizer");
    test.check(view.verdict == lpl::infer::ModelReadVerdict::Read && holdTheFixtureMerges(viewed.merges()) &&
                   viewed.merges().merges() == reinterpret_cast<const lpl::infer::BytePairMerge *>(image + last.offset),
               "read in place, the merges are the image's own bytes");
    test.check(bound && run.agreeing == smollm2_byte_pairs::kPromptCount,
               "and the model read in place encodes every prompt as the reference does");
    test.measure("image_bytes", size);
    test.measureHexadecimal("id_signature", run.signature);
}

/**
 * @brief An image whose merge names another result, its signature written again, is refused as merges_refused naming
 *        the merges' segment; an unknown pre-tokenizer is refused by its name at the header.
 */
LPL_TEST(an_image_whose_merges_or_pre_tokenizer_do_not_hold_is_refused)
{
    lpl::infer::TensorArena arena{kImageArenaBytes};
    Fixture fixture{};
    lpl::infer::Model model;
    lpl::core::u32 size = 0u;

    readFixture(arena, fixture);
    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();
    shape.vocabSize = smollm2_byte_pairs::kTokenCount;
    const bool synthesised = fixture.bound && model.synthesise(arena, shape, fixture.vocab, fixture.merges,
                                                               lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel,
                                                               lpl::infer::parityModelSeed());
    lpl::core::u8 *const image = synthesised ? writeImage(arena, model, size) : nullptr;
    if (!test.check(image != nullptr, "a model over the fixture's tables writes its image"))
        return;

    const lpl::infer::ModelImageInspection inspection = lpl::infer::inspectModelImage(image, size, nullptr, 0u);
    if (!test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                        inspection.header.segmentCount >= 1u,
                    "and its header and table are accepted"))
        return;

    const lpl::core::u32 lastEntry = inspection.header.segmentCount - 1u;
    lpl::infer::ModelSegment merges = lpl::infer::readModelSegmentEntry(image, lastEntry);
    putWordAt(image, merges.offset + 8u, smollm2_byte_pairs::kMerges[0].left);
    putWordAt(image,
              lpl::infer::kModelImageHeaderBytes + lastEntry * lpl::infer::kModelSegmentEntryBytes +
                  kSegmentSignatureAt,
              lpl::infer::modelSegmentSignature(image, merges));
    lpl::infer::Model refused;
    const lpl::infer::ModelReadReport report = refused.view(arena, image, size);

    putWordAt(image, kPreTokenizerWordIndex, 2u);
    const lpl::infer::ModelImageInspection unknown = lpl::infer::inspectModelImage(image, size, nullptr, 0u);

    test.check(report.verdict == lpl::infer::ModelReadVerdict::MergesRefused && report.segment == lastEntry + 1u,
               "a merge whose result is not its pair's bytes is refused as merges_refused, naming its segment");
    test.check(unknown.verdict == lpl::infer::ModelImageVerdict::UnknownPreTokenizer && unknown.segment == 0u &&
                   sameText(lpl::infer::modelImageVerdictName(unknown.verdict), "unknown_pre_tokenizer"),
               "a pre-tokenizer this reader does not know is refused at the header");
}

/**
 * @brief A word of exactly kBytePairWorkspaceSymbols bytes is merged whole; one byte more and it is cut, and counted
 *        once; either way its tokens spell it.
 */
LPL_TEST(a_word_longer_than_the_workspace_is_cut_and_counted)
{
    lpl::infer::TensorArena arena{kTokenizerArenaBytes + 8u * lpl::infer::kBytePairWorkspaceSymbols};
    Fixture fixture{};
    const lpl::core::u32 longest = lpl::infer::kBytePairWorkspaceSymbols + 1u;
    char *const text = arena.claim<char>(longest);
    char *const decoded = arena.claim<char>(longest);
    lpl::core::u32 *const ids = arena.claim<lpl::core::u32>(longest);

    readFixture(arena, fixture);
    if (!test.check(fixture.bound && text != nullptr && decoded != nullptr && ids != nullptr,
                    "the fixture's tables and a word's room fit"))
        return;

    for (lpl::core::u32 i = 0u; i < longest; ++i)
        text[i] = 'a';
    bool unspellable[256];
    markUnspellableBytes(fixture.vocab, unspellable);

    lpl::core::u32 skipped = 0u;
    const lpl::core::u32 before = lpl::infer::bytePairWordsCut();
    const lpl::core::u32 whole = fixture.tokenizer.encode(text, longest - 1u, ids, longest, skipped);
    const bool wholeSpells =
        decodesToTheSpellableBytes(fixture.tokenizer, ids, whole, text, longest - 1u, unspellable, decoded);
    const lpl::core::u32 afterWhole = lpl::infer::bytePairWordsCut();
    const lpl::core::u32 cut = fixture.tokenizer.encode(text, longest, ids, longest, skipped);
    const bool cutSpells = decodesToTheSpellableBytes(fixture.tokenizer, ids, cut, text, longest, unspellable, decoded);
    const lpl::core::u32 afterCut = lpl::infer::bytePairWordsCut();

    test.check(afterWhole == before && wholeSpells, "a word that fits is merged whole, and its tokens spell it");
    test.check(afterCut == before + 1u && cutSpells,
               "one byte more and it is cut, counted once, its tokens spelling it");
    test.measure("tokens_whole", whole);
    test.measure("tokens_cut", cut);
}

/**
 * @brief Made-up texts of every byte value, invalid UTF-8 and bytes without a token among them, encode to tokens that
 *        decode to every byte that has a token, the others counted as skipped; encoding into less room writes the
 *        first tokens only.
 */
LPL_TEST(encoding_then_decoding_gives_back_every_byte_that_has_a_token)
{
    lpl::infer::TensorArena arena{kTokenizerArenaBytes + 8u * kMadeUpTextBytes};
    Fixture fixture{};
    char *const text = arena.claim<char>(kMadeUpTextBytes);
    char *const decoded = arena.claim<char>(kMadeUpTextBytes);
    lpl::core::u32 *const ids = arena.claim<lpl::core::u32>(kMadeUpTextBytes);
    lpl::core::u32 *const truncated = arena.claim<lpl::core::u32>(kMadeUpTextBytes);

    readFixture(arena, fixture);
    if (!test.check(fixture.bound && text != nullptr && decoded != nullptr && ids != nullptr && truncated != nullptr,
                    "the fixture's tables and the texts' room fit"))
        return;

    bool unspellable[256];
    markUnspellableBytes(fixture.vocab, unspellable);
    lpl::core::u32 state = kMadeUpSeed;
    lpl::core::u32 spelling = 0u;
    lpl::core::u32 countingSkipped = 0u;
    lpl::core::u32 truncating = 0u;
    lpl::core::u32 skippedOverall = 0u;
    lpl::core::u32 signature = kFnv1aOffsetBasis;

    for (lpl::core::u32 index = 0u; index < kMadeUpTexts; ++index)
    {
        lpl::core::u32 expectedSkipped = 0u;
        for (lpl::core::u32 i = 0u; i < kMadeUpTextBytes; ++i)
        {
            text[i] = madeUpByte(state);
            expectedSkipped += unspellable[static_cast<lpl::core::u8>(text[i])] ? 1u : 0u;
        }

        lpl::core::u32 skipped = 0u;
        lpl::core::u32 truncatedSkipped = 0u;
        const lpl::core::u32 count = fixture.tokenizer.encode(text, kMadeUpTextBytes, ids, kMadeUpTextBytes, skipped);
        const lpl::core::u32 shortCount =
            fixture.tokenizer.encode(text, kMadeUpTextBytes, truncated, kTruncatedCapacity, truncatedSkipped);
        bool samePrefix = shortCount == (count < kTruncatedCapacity ? count : kTruncatedCapacity);

        for (lpl::core::u32 i = 0u; samePrefix && i < shortCount; ++i)
            samePrefix = truncated[i] == ids[i];
        for (lpl::core::u32 i = 0u; i < count; ++i)
            foldWord(signature, ids[i]);
        spelling +=
            decodesToTheSpellableBytes(fixture.tokenizer, ids, count, text, kMadeUpTextBytes, unspellable, decoded) ?
                1u :
                0u;
        countingSkipped += skipped == expectedSkipped ? 1u : 0u;
        truncating += samePrefix ? 1u : 0u;
        skippedOverall += skipped;
    }

    test.check(skippedOverall > 0u, "the texts hold bytes without a token");
    test.check(spelling == kMadeUpTexts, "every text's tokens decode to its bytes that have a token");
    test.check(countingSkipped == kMadeUpTexts, "and the others are counted as skipped");
    test.check(truncating == kMadeUpTexts, "encoding into three slots writes the first three tokens");
    test.measure("skipped", skippedOverall);
    test.measureHexadecimal("id_signature", signature);
}

/**
 * @brief On a thousand words drawn from three letters, merged by two hundred made-up merges, the encoder writes the
 *        tokens the plain way of merging gives, worked apart: lowest rank first, leftmost first among equals. Few
 * letters make long competing runs, which reach the heap's rarer paths, such as a removal whose replacement sifts up.
 */
LPL_TEST(the_merging_agrees_with_lowest_rank_first_on_made_up_merges)
{
    lpl::infer::TensorArena arena{kMadeUpArenaBytes};
    MadeUpMerges table{};
    lpl::infer::Vocab vocab;
    lpl::infer::BytePairMerges merges;
    lpl::infer::Tokenizer tokenizer;
    char *const word = arena.claim<char>(kMadeUpWordMostBytes);
    lpl::core::u32 *const expected = arena.claim<lpl::core::u32>(kMadeUpWordMostBytes);
    lpl::core::u32 *const ids = arena.claim<lpl::core::u32>(kMadeUpWordMostBytes);

    const bool made =
        word != nullptr && expected != nullptr && ids != nullptr && makeUpMerges(arena, table) &&
        vocab.borrow(arena, table.text, table.textBytes, table.entries, table.tokens) &&
        merges.borrow(arena, table.merges, table.mergeCount, vocab) == lpl::infer::BytePairMergesVerdict::Indexed &&
        tokenizer.bind(arena, vocab, merges, lpl::infer::PreTokenizer::WholeText);
    if (!test.check(made && table.mergeCount == kMadeUpMergeCount, "two hundred merges are made and bound"))
        return;

    lpl::core::u32 state = kMadeUpSeed ^ kFnv1aPrime;
    lpl::core::u32 agreeing = 0u;
    lpl::core::u32 applied = 0u;
    lpl::core::u32 signature = kFnv1aOffsetBasis;
    for (lpl::core::u32 index = 0u; index < kMadeUpWords; ++index)
    {
        lpl::core::u32 count = 1u + nextDraw(state) % kMadeUpWordMostBytes;
        for (lpl::core::u32 i = 0u; i < count; ++i)
        {
            const lpl::core::u32 letter = nextDraw(state) % kMadeUpAlphabet;
            word[i] = static_cast<char>('a' + letter);
            expected[i] = letter;
        }

        lpl::core::u32 skipped = 0u;
        const lpl::core::u32 written = tokenizer.encode(word, count, ids, kMadeUpWordMostBytes, skipped);
        applied += mergeLowestRankFirst(table, expected, count);
        bool same = written == count && skipped == 0u;
        for (lpl::core::u32 i = 0u; same && i < count; ++i)
            same = ids[i] == expected[i];
        for (lpl::core::u32 i = 0u; i < written; ++i)
            foldWord(signature, ids[i]);
        agreeing += same ? 1u : 0u;
    }

    test.check(applied > kMadeUpWords, "the words merge, more than once each on average");
    test.check(agreeing == kMadeUpWords, "and every word's tokens are those of the plain merging");
    test.measure("words", kMadeUpWords);
    test.measure("merges_applied", applied);
    test.measureHexadecimal("id_signature", signature);
}

/**
 * @brief Five merges, ab, cb, ac, cc then bb, encode abbaccccb as ab b ac cc cb, worked by hand: ab first, cb at the
 *        end, ac, then the middle cc, no pair left. Merging ab takes the second b's pair out of the middle of the heap,
 *        and the pair that replaces it there must climb above its new parent for ac to come before cc.
 */
LPL_TEST(the_lowest_rank_still_merges_first_after_a_pair_leaves_the_middle_of_the_heap)
{
    constexpr char kText[] = "abcabcbaccccbb";
    constexpr lpl::infer::VocabEntry kEntries[] = {
        {0u,  1u},
        {1u,  1u},
        {2u,  1u},
        {3u,  2u},
        {5u,  2u},
        {7u,  2u},
        {9u,  2u},
        {12u, 2u}
    };
    constexpr lpl::infer::BytePairMerge kHandMerges[] = {
        {0u, 1u, 3u},
        {2u, 1u, 4u},
        {0u, 2u, 5u},
        {2u, 2u, 6u},
        {1u, 1u, 7u}
    };
    constexpr char kWord[] = "abbaccccb";
    constexpr lpl::core::u32 kExpected[] = {3u, 1u, 5u, 6u, 4u};
    constexpr lpl::core::u32 kExpectedCount = sizeof(kExpected) / sizeof(kExpected[0]);
    lpl::infer::TensorArena arena{kTokenizerArenaBytes};
    lpl::infer::Vocab vocab;
    lpl::infer::BytePairMerges merges;
    lpl::infer::Tokenizer tokenizer;

    if (!test.check(vocab.borrow(arena, kText, sizeof(kText) - 1u, kEntries, 8u) &&
                        merges.borrow(arena, kHandMerges, 5u, vocab) == lpl::infer::BytePairMergesVerdict::Indexed &&
                        tokenizer.bind(arena, vocab, merges, lpl::infer::PreTokenizer::WholeText),
                    "the five merges are indexed and bound"))
        return;

    lpl::core::u32 ids[16];
    lpl::core::u32 skipped = 0u;
    const lpl::core::u32 count = tokenizer.encode(kWord, sizeof(kWord) - 1u, ids, 16u, skipped);
    bool same = count == kExpectedCount && skipped == 0u;
    for (lpl::core::u32 i = 0u; same && i < count; ++i)
        same = ids[i] == kExpected[i];

    test.check(same, "abbaccccb encodes as ab b ac cc cb");
    test.measure("tokens", count);
}

/**
 * @brief A tokenizer bound to the parity vocabulary with no merge and the whole text as one piece claims nothing and
 *        encodes the parity prompt as the tokenizer of a vocabulary alone does.
 */
LPL_TEST(a_vocabulary_without_merges_is_still_segmented_by_longest_match)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::BytePairMerges none;
    lpl::infer::Tokenizer bound;
    const char *const prompt = lpl::infer::parityPrompt();
    lpl::core::u32 promptBytes = 0u;

    while (prompt[promptBytes] != '\0')
        ++promptBytes;
    if (!test.check(lpl::infer::buildParityVocab(arena, vocab) &&
                        none.borrow(arena, nullptr, 0u, vocab) == lpl::infer::BytePairMergesVerdict::Indexed,
                    "the parity vocabulary builds, with no merge"))
        return;

    const lpl::core::usize before = arena.used();
    const bool claimedNothing =
        bound.bind(arena, vocab, none, lpl::infer::PreTokenizer::WholeText) && arena.used() == before;
    lpl::core::u32 boundIds[32];
    lpl::core::u32 aloneIds[32];
    lpl::core::u32 boundSkipped = 0u;
    lpl::core::u32 aloneSkipped = 0u;
    const lpl::core::u32 boundCount = bound.encode(prompt, promptBytes, boundIds, 32u, boundSkipped);
    const lpl::core::u32 aloneCount =
        lpl::infer::Tokenizer{vocab}.encode(prompt, promptBytes, aloneIds, 32u, aloneSkipped);
    bool same = boundCount == aloneCount && boundSkipped == aloneSkipped && boundCount > 0u;

    for (lpl::core::u32 i = 0u; same && i < boundCount; ++i)
        same = boundIds[i] == aloneIds[i];

    test.check(claimedNothing, "binding no merge claims nothing");
    test.check(same, "and encodes the parity prompt as the vocabulary alone does");
    test.measure("tokens", boundCount);
}
