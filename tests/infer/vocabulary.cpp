#include <lpl/infer/TensorArena.hpp>
#include <lpl/infer/Vocab.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(vocabulary);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;
constexpr lpl::core::u32 kMadeUpSeed = 0x5EED0097u;
constexpr lpl::core::u32 kLinearCongruentialMultiplier = 1664525u;
constexpr lpl::core::u32 kLinearCongruentialIncrement = 1013904223u;
constexpr lpl::core::u32 kDrawShift = 16u;
constexpr lpl::core::u32 kArenaSlackBytes = 64u;

/** The bytes a made-up text is spelled with, the first ones the most frequent; half of them above 0x7F. */
constexpr lpl::core::u8 kMadeUpSymbols[] = {' ',   'e',   't',   'a',   'o',   'i',   'n',   's',
                                            'h',   'r',   'd',   'l',   0x80u, 0x9Fu, 0xA0u, 0xBFu,
                                            0xC3u, 0xC5u, 0xD0u, 0xE2u, 0xE3u, 0xEFu, 0xF0u, 0xFFu};
constexpr lpl::core::u32 kMadeUpSymbolCount = sizeof(kMadeUpSymbols) / sizeof(kMadeUpSymbols[0]);

constexpr lpl::core::u32 kOrderedWindows = 1000u;
constexpr lpl::core::u32 kLongTokens = 48u;
constexpr lpl::core::u32 kLongTailBytes = 8u;
constexpr lpl::core::u32 kLongLengthStride = 5u;
constexpr lpl::core::u32 kRepeatedTokens = 64u;
constexpr lpl::core::u32 kEmptyTokens = 1u;
constexpr lpl::core::u32 kLongRegionBytes = lpl::infer::kMaxTokenBytes + kLongTailBytes;
constexpr lpl::core::u8 kHighTailByte = 0xFFu;
constexpr lpl::core::u8 kLowTailByte = 0x01u;

/** Sixteen bytes every long token starts with, so the order of those tokens is decided past kMaxTokenBytes. */
constexpr char kLongHead[] = "the-sixteen-head";
static_assert(sizeof(kLongHead) - 1u == lpl::infer::kMaxTokenBytes, "the long head spans exactly kMaxTokenBytes");

/** The token repeated kRepeatedTokens times: a high byte, a continuation byte and a space. */
constexpr lpl::core::u8 kRepeatedText[] = {0xC3u, 0xA9u, ' '};
constexpr lpl::core::u32 kRepeatedTextBytes = sizeof(kRepeatedText);

constexpr lpl::core::u32 kEdgeTextBytes = 2u * kLongRegionBytes + kRepeatedTextBytes;
constexpr lpl::core::u32 kEdgeTokens = kLongTokens + kRepeatedTokens + kEmptyTokens;

/** Texts no made-up vocabulary holds: a byte below every symbol, one between them, and one past every long token. */
constexpr char kAbsentLow[] = "\x01";
constexpr char kAbsentBetween[] = "zz";
constexpr char kAbsentLong[] = "the-sixteen-head\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF";

constexpr lpl::core::u32 kAThousand = 1000u;
constexpr lpl::core::u32 kTenThousand = 10000u;
constexpr lpl::core::u32 kAHundredThousand = 100000u;

/** What find answers for every token of each size, folded, as the insertion sort of LplAssistant 0.6.0 orders it. */
constexpr lpl::core::u32 kAThousandSignature = 0xEA9BA889u;
constexpr lpl::core::u32 kTenThousandSignature = 0xCF7736DFu;
constexpr lpl::core::u32 kAHundredThousandSignature = 0xB57703E3u;

/**
 * @brief A vocabulary the test made up, in an arena: the text its tokens are windows of, and one entry per token.
 */
struct MadeUpVocabulary {
    char *text{nullptr};                      ///< The bytes every token is a run of.
    lpl::infer::VocabEntry *entries{nullptr}; ///< One per token, in identifier order.
    lpl::core::u32 textBytes{0u};             ///< Bytes in @ref text.
    lpl::core::u32 count{0u};                 ///< Tokens.
};

/**
 * @brief What looking every token of a vocabulary up found.
 */
struct LookUp {
    lpl::core::u32 signature{kFnv1aOffsetBasis}; ///< FNV-1a fold of what find returned for each token, in order.
    lpl::core::u32 found{0u};                    ///< Tokens find answered with a token of the same bytes.
    lpl::core::u32 foundElsewhere{0u};           ///< Of those, the ones answered with another identifier.
};

/**
 * @brief Folds one word into an FNV-1a hash.
 */
void foldWord(lpl::core::u32 &hash, lpl::core::u32 word) { hash = (hash ^ word) * kFnv1aPrime; }

/**
 * @brief The next draw of a linear congruential generator, its high half.
 */
[[nodiscard]] lpl::core::u32 nextDraw(lpl::core::u32 &state)
{
    state = state * kLinearCongruentialMultiplier + kLinearCongruentialIncrement;
    return state >> kDrawShift;
}

/**
 * @brief One byte of made-up text: the smaller of two draws, so the first symbols are the most frequent.
 */
[[nodiscard]] char madeUpByte(lpl::core::u32 &state)
{
    const lpl::core::u32 first = nextDraw(state) % kMadeUpSymbolCount;
    const lpl::core::u32 second = nextDraw(state) % kMadeUpSymbolCount;

    return static_cast<char>(kMadeUpSymbols[first < second ? first : second]);
}

/**
 * @brief The arena a made-up vocabulary of @p count tokens over @p textBytes of text and its index take.
 */
[[nodiscard]] lpl::core::usize madeUpArenaBytes(lpl::core::u32 count, lpl::core::u32 textBytes)
{
    return static_cast<lpl::core::usize>(count) * (sizeof(lpl::infer::VocabEntry) + sizeof(lpl::core::u32)) +
           textBytes + kArenaSlackBytes;
}

/**
 * @brief Claims a vocabulary of @p windows tokens over a made-up text, with room for @p extraTokens more tokens and
 *        @p extraTextBytes more bytes of text after them.
 *
 * Token i starts at byte i of the text and spans 1 to kMaxTokenBytes bytes: the tokens arrive in no text order, the
 * short ones repeat, and the long ones share their first bytes, as the runs of a real text do.
 *
 * @return Whether the arena held it.
 */
[[nodiscard]] bool makeUpWindows(lpl::infer::TensorArena &arena, lpl::core::u32 windows, lpl::core::u32 extraTokens,
                                 lpl::core::u32 extraTextBytes, MadeUpVocabulary &out)
{
    const lpl::core::u32 windowTextBytes = windows + lpl::infer::kMaxTokenBytes;

    out.textBytes = windowTextBytes + extraTextBytes;
    out.count = windows + extraTokens;
    out.text = arena.claim<char>(out.textBytes);
    out.entries = arena.claim<lpl::infer::VocabEntry>(out.count);
    if (out.text == nullptr || out.entries == nullptr)
        return false;

    lpl::core::u32 state = kMadeUpSeed;

    for (lpl::core::u32 at = 0u; at < windowTextBytes; ++at)
        out.text[at] = madeUpByte(state);
    for (lpl::core::u32 token = 0u; token < windows; ++token)
        out.entries[token] = lpl::infer::VocabEntry{token, 1u + nextDraw(state) % lpl::infer::kMaxTokenBytes};
    return true;
}

/**
 * @brief Writes kLongHead then kLongTailBytes copies of @p tail at @p at of @p text.
 * @return The offset of what it wrote.
 */
lpl::core::u32 writeLongRegion(char *text, lpl::core::u32 at, lpl::core::u8 tail)
{
    for (lpl::core::u32 i = 0u; i < lpl::infer::kMaxTokenBytes; ++i)
        text[at + i] = kLongHead[i];
    for (lpl::core::u32 i = 0u; i < kLongTailBytes; ++i)
        text[at + lpl::infer::kMaxTokenBytes + i] = static_cast<char>(tail);
    return at;
}

/**
 * @brief Appends the tokens no window makes, after @p windows of them: tokens past kMaxTokenBytes that share their
 *        first sixteen bytes and part from a high byte or a low one, a token repeated more times than a small sort
 *        sees at once, and an empty token.
 */
void addEdgeTokens(MadeUpVocabulary &vocabulary, lpl::core::u32 windows)
{
    const lpl::core::u32 high = writeLongRegion(vocabulary.text, windows + lpl::infer::kMaxTokenBytes, kHighTailByte);
    const lpl::core::u32 low = writeLongRegion(vocabulary.text, high + kLongRegionBytes, kLowTailByte);
    const lpl::core::u32 repeated = low + kLongRegionBytes;

    for (lpl::core::u32 i = 0u; i < kRepeatedTextBytes; ++i)
        vocabulary.text[repeated + i] = static_cast<char>(kRepeatedText[i]);

    lpl::core::u32 token = windows;

    for (lpl::core::u32 k = 0u; k < kLongTokens; ++k)
        vocabulary.entries[token++] = lpl::infer::VocabEntry{
            k % 2u == 0u ? high : low, lpl::infer::kMaxTokenBytes + (k * kLongLengthStride) % (kLongTailBytes + 1u)};
    for (lpl::core::u32 k = 0u; k < kRepeatedTokens; ++k)
        vocabulary.entries[token++] = lpl::infer::VocabEntry{repeated, kRepeatedTextBytes};
    vocabulary.entries[token] = lpl::infer::VocabEntry{0u, 0u};
}

/**
 * @brief Orders @p length bytes at @p bytes against token @p token: negative, zero or positive as the token's bytes,
 *        read as unsigned, come before, equal or after them, the shorter first when one is a prefix of the other.
 *
 * Written apart from the module's comparison, which is what it checks.
 */
[[nodiscard]] lpl::core::i32 compareTokenWith(const MadeUpVocabulary &vocabulary, lpl::core::u32 token,
                                              const char *bytes, lpl::core::u32 length)
{
    const lpl::infer::VocabEntry entry = vocabulary.entries[token];
    const lpl::core::u8 *const left = reinterpret_cast<const lpl::core::u8 *>(vocabulary.text + entry.offset);
    const lpl::core::u8 *const right = reinterpret_cast<const lpl::core::u8 *>(bytes);

    for (lpl::core::u32 at = 0u; at < entry.length && at < length; ++at)
    {
        if (left[at] != right[at])
            return left[at] < right[at] ? -1 : 1;
    }
    if (entry.length == length)
        return 0;
    return entry.length < length ? -1 : 1;
}

/**
 * @brief Whether token @p lhs comes before token @p rhs in the index: by their bytes, and the smaller identifier first
 *        when the bytes are the same.
 */
[[nodiscard]] bool comesBefore(const MadeUpVocabulary &vocabulary, lpl::core::u32 lhs, lpl::core::u32 rhs)
{
    const lpl::infer::VocabEntry right = vocabulary.entries[rhs];
    const lpl::core::i32 side = compareTokenWith(vocabulary, lhs, vocabulary.text + right.offset, right.length);

    return side != 0 ? side < 0 : lhs < rhs;
}

/**
 * @brief Fills @p order with every identifier of @p vocabulary in the order comesBefore gives, one token at a time.
 */
void referenceOrder(const MadeUpVocabulary &vocabulary, lpl::core::u32 *order)
{
    for (lpl::core::u32 placed = 0u; placed < vocabulary.count; ++placed)
    {
        lpl::core::u32 rank = placed;

        while (rank > 0u && comesBefore(vocabulary, placed, order[rank - 1u]))
        {
            order[rank] = order[rank - 1u];
            --rank;
        }
        order[rank] = placed;
    }
}

/**
 * @brief The token Vocab::find answers for @p length bytes at @p bytes when its index is @p order: the binary search it
 *        walks, over the reference order.
 */
[[nodiscard]] lpl::core::u32 referenceFind(const MadeUpVocabulary &vocabulary, const lpl::core::u32 *order,
                                           const char *bytes, lpl::core::u32 length)
{
    if (length == 0u)
        return lpl::infer::kNoToken;

    lpl::core::u32 low = 0u;
    lpl::core::u32 high = vocabulary.count;

    while (low < high)
    {
        const lpl::core::u32 middle = low + (high - low) / 2u;
        const lpl::core::i32 side = compareTokenWith(vocabulary, order[middle], bytes, length);

        if (side == 0)
            return order[middle];
        if (side < 0)
            low = middle + 1u;
        else
            high = middle;
    }
    return lpl::infer::kNoToken;
}

/**
 * @brief Looks every token of @p vocabulary up in @p vocab, by its own bytes.
 */
[[nodiscard]] LookUp lookEveryTokenUp(const MadeUpVocabulary &vocabulary, const lpl::infer::Vocab &vocab)
{
    LookUp result{};

    for (lpl::core::u32 token = 0u; token < vocabulary.count; ++token)
    {
        const lpl::infer::VocabEntry entry = vocabulary.entries[token];
        const lpl::core::u32 answer = vocab.find(vocabulary.text + entry.offset, entry.length);

        foldWord(result.signature, answer);
        if (answer >= vocabulary.count)
            continue;

        const lpl::infer::VocabEntry found = vocabulary.entries[answer];
        if (compareTokenWith(vocabulary, token, vocabulary.text + found.offset, found.length) != 0)
            continue;
        ++result.found;
        if (answer != token)
            ++result.foundElsewhere;
    }
    return result;
}

/**
 * @brief Whether none of the absent texts is found.
 */
[[nodiscard]] bool findsNoAbsentText(const lpl::infer::Vocab &vocab)
{
    return vocab.find(kAbsentLow, sizeof(kAbsentLow) - 1u) == lpl::infer::kNoToken &&
           vocab.find(kAbsentBetween, sizeof(kAbsentBetween) - 1u) == lpl::infer::kNoToken &&
           vocab.find(kAbsentLong, sizeof(kAbsentLong) - 1u) == lpl::infer::kNoToken;
}

/**
 * @brief Borrows a vocabulary of @p count made-up tokens and checks that find answers every one of them with a token
 *        of its bytes, as today's order does, which @p expectedSignature records.
 */
void checkMadeUpVocabulary(lpl::testing::Test &test, lpl::core::u32 count, lpl::core::u32 expectedSignature)
{
    lpl::infer::TensorArena arena{madeUpArenaBytes(count, count + lpl::infer::kMaxTokenBytes)};
    MadeUpVocabulary vocabulary{};
    lpl::infer::Vocab vocab;

    if (!test.check(makeUpWindows(arena, count, 0u, 0u, vocabulary), "the made-up vocabulary fits its arena"))
        return;
    if (!test.check(vocab.borrow(arena, vocabulary.text, vocabulary.textBytes, vocabulary.entries, vocabulary.count),
                    "and its index too"))
        return;

    const LookUp lookUp = lookEveryTokenUp(vocabulary, vocab);

    test.check(lookUp.found == count, "every token is found by its own bytes");
    test.check(lookUp.foundElsewhere > 0u, "some by a token of the same bytes, so equal tokens are ordered");
    test.check(findsNoAbsentText(vocab), "and no text the vocabulary does not hold");
    test.check(lookUp.signature == expectedSignature, "each token is found where today's order puts it");
    test.measure("tokens", count);
    test.measure("found", lookUp.found);
    test.measure("found_elsewhere", lookUp.foundElsewhere);
    test.measureHexadecimal("find_signature", lookUp.signature);
}

} // namespace

/**
 * @brief The index orders a vocabulary by its bytes read as unsigned, the shorter first when one token is a prefix of
 *        another and the smaller identifier first when two tokens are the same bytes, past kMaxTokenBytes too, and
 *        borrowing claims the index and nothing more: find answers every token as a binary search over that order,
 *        worked apart, does.
 */
LPL_TEST(the_index_orders_unsigned_bytes_the_shorter_first_and_equal_tokens_by_identifier)
{
    const lpl::core::u32 count = kOrderedWindows + kEdgeTokens;
    const lpl::core::u32 textBytes = kOrderedWindows + lpl::infer::kMaxTokenBytes + kEdgeTextBytes;
    lpl::infer::TensorArena arena{madeUpArenaBytes(count, textBytes) + count * sizeof(lpl::core::u32)};
    MadeUpVocabulary vocabulary{};
    lpl::infer::Vocab vocab;

    if (!test.check(makeUpWindows(arena, kOrderedWindows, kEdgeTokens, kEdgeTextBytes, vocabulary),
                    "the made-up vocabulary fits its arena"))
        return;
    addEdgeTokens(vocabulary, kOrderedWindows);

    lpl::core::u32 *const order = arena.claim<lpl::core::u32>(count);
    if (!test.check(order != nullptr, "and so does its reference order"))
        return;
    referenceOrder(vocabulary, order);

    const lpl::core::usize before = arena.used();
    const bool borrowed =
        vocab.borrow(arena, vocabulary.text, vocabulary.textBytes, vocabulary.entries, vocabulary.count);
    const lpl::core::usize growth = arena.used() - before;

    if (!test.check(borrowed, "the vocabulary borrows its index"))
        return;
    test.check(growth >= count * sizeof(lpl::core::u32) && growth < (count + 1u) * sizeof(lpl::core::u32),
               "which claims one identifier per token and nothing more");

    lpl::core::u32 disagreements = 0u;

    for (lpl::core::u32 token = 0u; token < count; ++token)
    {
        const lpl::infer::VocabEntry entry = vocabulary.entries[token];
        const char *const bytes = vocabulary.text + entry.offset;

        if (vocab.find(bytes, entry.length) != referenceFind(vocabulary, order, bytes, entry.length))
            ++disagreements;
    }

    const LookUp lookUp = lookEveryTokenUp(vocabulary, vocab);

    test.check(disagreements == 0u, "find answers every token as the search over the reference order does");
    test.check(lookUp.found == count - kEmptyTokens, "every token but the empty one is found by its own bytes");
    test.check(lookUp.foundElsewhere > kRepeatedTokens / 2u, "the repeated ones and others by a token of theirs");
    test.check(findsNoAbsentText(vocab), "and no text the vocabulary does not hold");
    test.measure("tokens", count);
    test.measure("disagreements", disagreements);
    test.measure("found_elsewhere", lookUp.foundElsewhere);
    test.measureHexadecimal("find_signature", lookUp.signature);
}

/**
 * @brief A thousand made-up tokens are each found by their bytes where today's order puts them.
 */
LPL_TEST(a_thousand_made_up_tokens_are_found_where_today_s_order_puts_them)
{
    checkMadeUpVocabulary(test, kAThousand, kAThousandSignature);
}

/**
 * @brief Ten thousand made-up tokens are each found by their bytes where today's order puts them.
 */
LPL_TEST(ten_thousand_made_up_tokens_are_found_where_today_s_order_puts_them)
{
    checkMadeUpVocabulary(test, kTenThousand, kTenThousandSignature);
}

/**
 * @brief A hundred thousand made-up tokens are each found by their bytes where today's order puts them.
 */
LPL_TEST(a_hundred_thousand_made_up_tokens_are_found_where_today_s_order_puts_them)
{
    checkMadeUpVocabulary(test, kAHundredThousand, kAHundredThousandSignature);
}
