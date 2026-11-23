/**
 * @file Tokenizer.cpp
 * @brief Text to tokens and back, deterministically.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Tokenizer.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include "UnicodeCharacterClasses.hpp"

namespace lpl::infer {

namespace {

/** The most merges an index sizes: twice as many slots must still be a power of two of 32 bits. */
constexpr core::u32 kMostIndexedMerges = 1u << 30;

/** The odd constant a pair's left token is spread by: 2^32 / phi, rounded to odd. */
constexpr core::u32 kLeftSpread = 0x9E3779B1u;

/** The odd constant a pair's right token is spread by: a multiplier of xxHash32. */
constexpr core::u32 kRightSpread = 0x85EBCA77u;

/** The odd constant the mixed pair is spread by again before its bits are kept: a multiplier of xxHash32. */
constexpr core::u32 kFinalSpread = 0xC2B2AE3Du;

/** Marks no symbol where a symbol's index is expected: past the left or right end, or out of the heap. */
constexpr core::u32 kNoSymbol = 0xFFFFFFFFu;

/**
 * Words the workspace holds per symbol: its token, its pair's rank, the next and previous symbols, a heap slot and
 * where the symbol sits in the heap.
 */
constexpr core::u32 kWorkspaceWordsPerSymbol = 6u;

/** Byte values, one token each at most. */
constexpr core::u32 kByteValues = 256u;

/** The code point of the one space GPT-2's pattern takes before a run: ` ?`. */
constexpr core::u32 kSpaceCodePoint = 0x20u;

/** The byte GPT-2's contractions start with. */
constexpr char kApostrophe = '\'';

/** The code point that stands for a byte that does not start a valid sequence of UTF-8. */
constexpr core::u32 kInvalidCodePoint = 0xFFFFFFFFu;

core::u32 gBytePairWordsCut = 0u;

/**
 * @brief The slot a pair hashes to, in an index of @p slotMask + 1 slots.
 */
core::u32 pairSlot(core::u32 left, core::u32 right, core::u32 slotMask) noexcept
{
    core::u32 mixed = left * kLeftSpread ^ right * kRightSpread;
    mixed ^= mixed >> 15;
    mixed *= kFinalSpread;
    mixed ^= mixed >> 13;
    return mixed & slotMask;
}

/**
 * @brief The smallest power of two at least twice @p count.
 */
core::u32 slotsFor(core::u32 count) noexcept
{
    core::u32 slots = 1u;
    while (slots < 2u * count)
        slots <<= 1;
    return slots;
}

/**
 * @brief Whether @p merge joins two tokens of @p vocab into a third whose bytes are theirs one after the other.
 */
BytePairMergesVerdict judgeMerge(const BytePairMerge &merge, const Vocab &vocab) noexcept
{
    core::u32 leftBytes = 0u;
    core::u32 rightBytes = 0u;
    core::u32 resultBytes = 0u;
    const char *const left = vocab.text(merge.left, leftBytes);
    const char *const right = vocab.text(merge.right, rightBytes);
    const char *const result = vocab.text(merge.result, resultBytes);

    if (left == nullptr || right == nullptr || result == nullptr)
        return BytePairMergesVerdict::UnknownToken;
    if (resultBytes != leftBytes + rightBytes)
        return BytePairMergesVerdict::NotTheConcatenation;
    for (core::u32 i = 0u; i < leftBytes; ++i)
    {
        if (result[i] != left[i])
            return BytePairMergesVerdict::NotTheConcatenation;
    }
    for (core::u32 i = 0u; i < rightBytes; ++i)
    {
        if (result[leftBytes + i] != right[i])
            return BytePairMergesVerdict::NotTheConcatenation;
    }
    return BytePairMergesVerdict::Indexed;
}

/**
 * @brief One character of UTF-8 text: its code point, its bytes and its class.
 */
struct Character {
    core::u32 codePoint; ///< kInvalidCodePoint for a byte that starts no valid sequence.
    core::u32 bytes;     ///< 1 to 4; 0 past the end of the text.
    CharacterClass type; ///< Other past the end of the text.
};

/**
 * @brief The bytes of the sequence a lead byte starts, and the least and most second byte it allows (RFC 3629, §4).
 */
struct SequenceShape {
    core::u32 bytes;      ///< 0 for a byte that starts no sequence.
    core::u32 leadBits;   ///< The lead byte's bits of the code point.
    core::u8 leastSecond; ///< The least second byte: past the overlong forms and the surrogates.
    core::u8 mostSecond;  ///< The most second byte: before the surrogates and U+110000.
};

/**
 * @brief The shape of the sequence @p lead starts.
 */
SequenceShape sequenceShapeOf(core::u8 lead) noexcept
{
    if (lead >= 0xC2u && lead <= 0xDFu)
        return SequenceShape{2u, lead & 0x1Fu, 0x80u, 0xBFu};
    if (lead == 0xE0u)
        return SequenceShape{3u, lead & 0x0Fu, 0xA0u, 0xBFu};
    if (lead == 0xEDu)
        return SequenceShape{3u, lead & 0x0Fu, 0x80u, 0x9Fu};
    if (lead >= 0xE1u && lead <= 0xEFu)
        return SequenceShape{3u, lead & 0x0Fu, 0x80u, 0xBFu};
    if (lead == 0xF0u)
        return SequenceShape{4u, lead & 0x07u, 0x90u, 0xBFu};
    if (lead == 0xF4u)
        return SequenceShape{4u, lead & 0x07u, 0x80u, 0x8Fu};
    if (lead >= 0xF1u && lead <= 0xF3u)
        return SequenceShape{4u, lead & 0x07u, 0x80u, 0xBFu};
    return SequenceShape{0u, 0u, 0u, 0u};
}

/**
 * @brief The character at @p at of @p bytes of @p text; one byte of class other when no valid sequence starts there.
 */
Character characterAt(const core::u8 *text, core::u32 bytes, core::u32 at) noexcept
{
    if (at >= bytes)
        return Character{kInvalidCodePoint, 0u, CharacterClass::Other};
    if (text[at] < 0x80u)
        return Character{text[at], 1u, characterClassOf(text[at])};

    const SequenceShape shape = sequenceShapeOf(text[at]);
    if (shape.bytes == 0u || shape.bytes > bytes - at || text[at + 1u] < shape.leastSecond ||
        text[at + 1u] > shape.mostSecond)
        return Character{kInvalidCodePoint, 1u, CharacterClass::Other};

    core::u32 codePoint = shape.leadBits;
    for (core::u32 i = 1u; i < shape.bytes; ++i)
    {
        if ((text[at + i] & 0xC0u) != 0x80u)
            return Character{kInvalidCodePoint, 1u, CharacterClass::Other};
        codePoint = codePoint << 6 | (text[at + i] & 0x3Fu);
    }
    return Character{codePoint, shape.bytes, characterClassOf(codePoint)};
}

/**
 * @brief The end of the run of characters of class @p type that starts at @p at.
 */
core::u32 endOfRun(const core::u8 *text, core::u32 bytes, core::u32 at, CharacterClass type) noexcept
{
    for (;;)
    {
        const Character next = characterAt(text, bytes, at);
        if (next.bytes == 0u || next.type != type)
            return at;
        at += next.bytes;
    }
}

/**
 * @brief The bytes of a contraction GPT-2's pattern takes at @p at, `'s|'t|'re|'ve|'m|'ll|'d`, or 0.
 */
core::u32 contractionBytes(const core::u8 *text, core::u32 bytes, core::u32 at) noexcept
{
    if (text[at] != kApostrophe || bytes - at < 2u)
        return 0u;

    const core::u8 first = text[at + 1u];
    if (first == 's' || first == 't' || first == 'm' || first == 'd')
        return 2u;
    if (bytes - at < 3u)
        return 0u;

    const core::u8 second = text[at + 2u];
    const bool twoLetters =
        (first == 'r' && second == 'e') || (first == 'v' && second == 'e') || (first == 'l' && second == 'l');
    return twoLetters ? 3u : 0u;
}

/**
 * @brief The bytes of the run of spaces at @p at as GPT-2's `\s+(?!\S)|\s+` takes it, a number ending the text the
 *        pattern sees: the whole run before the end or a number, else all of it but its last space when that leaves
 *        one, which then goes with the word after it.
 */
core::u32 spaceRunBytes(const core::u8 *text, core::u32 bytes, core::u32 at) noexcept
{
    core::u32 end = at;
    core::u32 lastStart = at;
    core::u32 spaces = 0u;
    for (Character next = characterAt(text, bytes, end); next.bytes != 0u && next.type == CharacterClass::Space;
         next = characterAt(text, bytes, end))
    {
        lastStart = end;
        end += next.bytes;
        ++spaces;
    }

    const Character after = characterAt(text, bytes, end);
    if (after.bytes == 0u || after.type == CharacterClass::Number || spaces < 2u)
        return end - at;
    return lastStart - at;
}

/**
 * @brief The bytes of the piece at @p at for Hugging Face's Digits with individual_digits, then its ByteLevel: a
 *        number alone, a contraction, a run of letters or of others after one optional space, or a run of spaces.
 */
core::u32 individualDigitsThenByteLevelBytes(const core::u8 *text, core::u32 bytes, core::u32 at) noexcept
{
    const Character first = characterAt(text, bytes, at);
    if (first.type == CharacterClass::Number)
        return first.bytes;

    const core::u32 contraction = contractionBytes(text, bytes, at);
    if (contraction != 0u)
        return contraction;

    const Character second = characterAt(text, bytes, at + first.bytes);
    const bool spaceThenRun = first.codePoint == kSpaceCodePoint && second.bytes != 0u &&
                              (second.type == CharacterClass::Letter || second.type == CharacterClass::Other);
    if (first.type == CharacterClass::Letter || first.type == CharacterClass::Other)
        return endOfRun(text, bytes, at + first.bytes, first.type) - at;
    if (spaceThenRun)
        return endOfRun(text, bytes, at + first.bytes + second.bytes, second.type) - at;
    return spaceRunBytes(text, bytes, at);
}

/**
 * @brief The arrays of one word's merging, each kBytePairWorkspaceSymbols long, cut out of the tokenizer's workspace.
 */
struct MergeWorkspace {
    core::u32 *tokens;    ///< Each symbol's token.
    core::u32 *ranks;     ///< The rank of each symbol's pair with the next one, or kNoMerge.
    core::u32 *next;      ///< The symbol after each, or kNoSymbol.
    core::u32 *previous;  ///< The symbol before each, or kNoSymbol.
    core::u32 *heap;      ///< The symbols whose pair is a merge, the one to merge first at 0.
    core::u32 *heapSlots; ///< Where each symbol sits in the heap, or kNoSymbol.
};

/**
 * @brief The workspace's arrays, one after the other in @p words.
 */
MergeWorkspace cutWorkspace(core::u32 *words) noexcept
{
    const core::u32 symbols = kBytePairWorkspaceSymbols;
    return MergeWorkspace{
        words, words + symbols, words + 2u * symbols, words + 3u * symbols, words + 4u * symbols, words + 5u * symbols};
}

/**
 * @brief The pairs of a word still to merge, a binary heap of their left symbols ordered by rank, then by position:
 *        the lowest rank first, the leftmost first among equal ranks.
 */
struct PairHeap {
    MergeWorkspace &work; ///< Where the heap and the ranks live.
    core::u32 size{0u};   ///< Symbols in the heap.

    /**
     * @brief Whether the pair at symbol @p lhs merges before the pair at symbol @p rhs.
     */
    [[nodiscard]] bool mergesFirst(core::u32 lhs, core::u32 rhs) const noexcept
    {
        return work.ranks[lhs] != work.ranks[rhs] ? work.ranks[lhs] < work.ranks[rhs] : lhs < rhs;
    }

    /**
     * @brief Puts @p symbol at heap slot @p slot.
     */
    void place(core::u32 slot, core::u32 symbol) noexcept
    {
        work.heap[slot] = symbol;
        work.heapSlots[symbol] = slot;
    }

    /**
     * @brief Moves the symbol at @p slot towards the root until its parent merges first.
     */
    void siftUp(core::u32 slot) noexcept
    {
        const core::u32 symbol = work.heap[slot];
        while (slot > 0u && mergesFirst(symbol, work.heap[(slot - 1u) / 2u]))
        {
            place(slot, work.heap[(slot - 1u) / 2u]);
            slot = (slot - 1u) / 2u;
        }
        place(slot, symbol);
    }

    /**
     * @brief Moves the symbol at @p slot towards the leaves until it merges before both its children.
     */
    void siftDown(core::u32 slot) noexcept
    {
        const core::u32 symbol = work.heap[slot];
        for (core::u32 child = 2u * slot + 1u; child < size; child = 2u * slot + 1u)
        {
            if (child + 1u < size && mergesFirst(work.heap[child + 1u], work.heap[child]))
                ++child;
            if (!mergesFirst(work.heap[child], symbol))
                break;
            place(slot, work.heap[child]);
            slot = child;
        }
        place(slot, symbol);
    }

    /**
     * @brief Takes @p symbol out of the heap, if it is in.
     */
    void remove(core::u32 symbol) noexcept
    {
        const core::u32 slot = work.heapSlots[symbol];
        if (slot == kNoSymbol)
            return;

        work.heapSlots[symbol] = kNoSymbol;
        const core::u32 last = work.heap[--size];
        if (slot == size)
            return;
        place(slot, last);
        siftDown(slot);
        siftUp(work.heapSlots[last]);
    }

    /**
     * @brief Puts @p symbol where its pair's rank, just changed, says: out of the heap when its pair is no merge.
     */
    void reposition(core::u32 symbol) noexcept
    {
        if (work.ranks[symbol] == kNoMerge)
        {
            remove(symbol);
            return;
        }
        if (work.heapSlots[symbol] == kNoSymbol)
        {
            place(size, symbol);
            siftUp(size++);
            return;
        }
        siftUp(work.heapSlots[symbol]);
        siftDown(work.heapSlots[symbol]);
    }
};

/**
 * @brief Links the @p count symbols of a word, ranks each one's pair with the next, and heaps the pairs that merge.
 */
void prepareWord(MergeWorkspace &work, PairHeap &heap, core::u32 count, const BytePairMerges &merges) noexcept
{
    for (core::u32 symbol = 0u; symbol < count; ++symbol)
    {
        work.next[symbol] = symbol + 1u < count ? symbol + 1u : kNoSymbol;
        work.previous[symbol] = symbol > 0u ? symbol - 1u : kNoSymbol;
        work.ranks[symbol] =
            symbol + 1u < count ? merges.rank(work.tokens[symbol], work.tokens[symbol + 1u]) : kNoMerge;
        work.heapSlots[symbol] = kNoSymbol;
        heap.reposition(symbol);
    }
}

/**
 * @brief Merges the pair at symbol @p left: it takes the merge's token, its right neighbour leaves the word, and the
 *        pairs it now forms on each side are ranked again.
 */
void mergePairAt(MergeWorkspace &work, PairHeap &heap, core::u32 left, const BytePairMerges &merges) noexcept
{
    const core::u32 right = work.next[left];

    work.tokens[left] = merges.merges()[work.ranks[left]].result;
    heap.remove(right);
    work.next[left] = work.next[right];
    if (work.next[left] != kNoSymbol)
        work.previous[work.next[left]] = left;

    work.ranks[left] =
        work.next[left] != kNoSymbol ? merges.rank(work.tokens[left], work.tokens[work.next[left]]) : kNoMerge;
    heap.reposition(left);

    const core::u32 before = work.previous[left];
    if (before == kNoSymbol)
        return;
    work.ranks[before] = merges.rank(work.tokens[before], work.tokens[left]);
    heap.reposition(before);
}

/**
 * @brief Merges the @p count symbols of a word until no adjacent pair is a merge, then writes its tokens.
 * @return Tokens written, @p room at most.
 */
core::u32 mergeWord(MergeWorkspace &work, core::u32 count, const BytePairMerges &merges, core::u32 *out,
                    core::u32 room) noexcept
{
    if (count == 0u)
        return 0u;

    PairHeap heap{work};
    prepareWord(work, heap, count, merges);
    while (heap.size > 0u)
        mergePairAt(work, heap, work.heap[0], merges);

    core::u32 written = 0u;
    for (core::u32 symbol = 0u; symbol != kNoSymbol && written < room; symbol = work.next[symbol])
        out[written++] = work.tokens[symbol];
    return written;
}

} // namespace

BytePairMergesVerdict BytePairMerges::borrow(TensorArena &arena, const BytePairMerge *merges, core::u32 count,
                                             const Vocab &vocab)
{
    *this = BytePairMerges{};
    if (count == 0u)
        return BytePairMergesVerdict::Indexed;
    if (merges == nullptr || count > kMostIndexedMerges)
        return BytePairMergesVerdict::ArenaExhausted;

    const core::u32 slotCount = slotsFor(count);
    core::u32 *const slots = arena.claim<core::u32>(slotCount);
    if (slots == nullptr)
        return BytePairMergesVerdict::ArenaExhausted;
    for (core::u32 slot = 0u; slot < slotCount; ++slot)
        slots[slot] = 0u;

    const core::u32 mask = slotCount - 1u;
    for (core::u32 rank = 0u; rank < count; ++rank)
    {
        const BytePairMerge &merge = merges[rank];
        const BytePairMergesVerdict verdict = judgeMerge(merge, vocab);
        if (verdict != BytePairMergesVerdict::Indexed)
            return verdict;

        core::u32 slot = pairSlot(merge.left, merge.right, mask);
        for (; slots[slot] != 0u; slot = (slot + 1u) & mask)
        {
            const BytePairMerge &held = merges[slots[slot] - 1u];
            if (held.left == merge.left && held.right == merge.right)
                return BytePairMergesVerdict::RepeatedPair;
        }
        slots[slot] = rank + 1u;
    }

    _merges = merges;
    _slots = slots;
    _count = count;
    _slotMask = mask;
    return BytePairMergesVerdict::Indexed;
}

core::u32 BytePairMerges::rank(core::u32 left, core::u32 right) const noexcept
{
    if (_count == 0u)
        return kNoMerge;

    for (core::u32 slot = pairSlot(left, right, _slotMask); _slots[slot] != 0u; slot = (slot + 1u) & _slotMask)
    {
        const BytePairMerge &held = _merges[_slots[slot] - 1u];
        if (held.left == left && held.right == right)
            return _slots[slot] - 1u;
    }
    return kNoMerge;
}

bool isKnownPreTokenizer(core::u32 value) noexcept
{
    return value == static_cast<core::u32>(PreTokenizer::WholeText) ||
           value == static_cast<core::u32>(PreTokenizer::IndividualDigitsThenByteLevel);
}

CharacterClass characterClassOf(core::u32 codePoint) noexcept
{
    if (codePoint >= unicode::kFirstUnindexedCodePoint)
        return CharacterClass::Other;

    const core::u32 block = unicode::kClassBlockOfCodePoints[codePoint >> unicode::kClassBlockShift];
    const core::u32 inBlock = codePoint & ((1u << unicode::kClassBlockShift) - 1u);
    const core::u32 packed = unicode::kClassBlocks[block][inBlock / 4u];
    return static_cast<CharacterClass>((packed >> (2u * (inBlock % 4u))) & 3u);
}

core::u32 preTokenizedPieceBytes(PreTokenizer preTokenizer, const char *text, core::u32 bytes, core::u32 at) noexcept
{
    if (text == nullptr || at >= bytes)
        return 0u;

    switch (preTokenizer)
    {
    case PreTokenizer::WholeText: return bytes - at;
    case PreTokenizer::IndividualDigitsThenByteLevel:
        return individualDigitsThenByteLevelBytes(reinterpret_cast<const core::u8 *>(text), bytes, at);
    }
    return bytes - at;
}

core::u32 bytePairWordsCut() noexcept { return gBytePairWordsCut; }

bool Tokenizer::bind(TensorArena &arena, const Vocab &vocab, const BytePairMerges &merges, PreTokenizer preTokenizer)
{
    *this = Tokenizer{};
    if (merges.size() == 0u)
    {
        _vocab = &vocab;
        _merges = &merges;
        _preTokenizer = preTokenizer;
        return true;
    }

    core::u32 *const byteTokens = arena.claim<core::u32>(kByteValues);
    core::u32 *const workspace = arena.claim<core::u32>(kWorkspaceWordsPerSymbol * kBytePairWorkspaceSymbols);
    if (byteTokens == nullptr || workspace == nullptr)
        return false;

    for (core::u32 value = 0u; value < kByteValues; ++value)
    {
        const char byte = static_cast<char>(value);
        byteTokens[value] = vocab.find(&byte, 1u);
    }
    _vocab = &vocab;
    _merges = &merges;
    _preTokenizer = preTokenizer;
    _byteTokens = byteTokens;
    _workspace = workspace;
    return true;
}

core::u32 Tokenizer::encode(const char *text, core::u32 bytes, core::u32 *out, core::u32 capacity,
                            core::u32 &outSkipped) const noexcept
{
    outSkipped = 0u;
    if (_vocab == nullptr || text == nullptr || out == nullptr)
        return 0u;

    core::u32 written = 0u;
    core::u32 cursor = 0u;
    while (cursor < bytes && written < capacity)
    {
        const core::u32 piece = preTokenizedPieceBytes(_preTokenizer, text, bytes, cursor);
        written += _workspace != nullptr ?
                       mergePiece(text + cursor, piece, out + written, capacity - written, outSkipped) :
                       matchPiece(text + cursor, piece, out + written, capacity - written, outSkipped);
        cursor += piece;
    }
    return written;
}

core::u32 Tokenizer::matchPiece(const char *piece, core::u32 bytes, core::u32 *out, core::u32 room,
                                core::u32 &skipped) const noexcept
{
    core::u32 written = 0u;
    core::u32 cursor = 0u;
    while (cursor < bytes && written < room)
    {
        core::u32 length = 0u;
        const core::u32 token = _vocab->longestPrefix(piece + cursor, bytes - cursor, length);
        if (token == kNoToken || length == 0u)
        {
            ++skipped;
            ++cursor;
            continue;
        }
        out[written++] = token;
        cursor += length;
    }
    return written;
}

core::u32 Tokenizer::mergePiece(const char *piece, core::u32 bytes, core::u32 *out, core::u32 room,
                                core::u32 &skipped) const noexcept
{
    MergeWorkspace work = cutWorkspace(_workspace);
    core::u32 written = 0u;
    core::u32 symbols = 0u;
    bool cut = false;

    for (core::u32 at = 0u; at < bytes; ++at)
    {
        const core::u32 token = _byteTokens[static_cast<core::u8>(piece[at])];
        if (token == kNoToken)
        {
            ++skipped;
            continue;
        }
        if (symbols == kBytePairWorkspaceSymbols)
        {
            written += mergeWord(work, symbols, *_merges, out + written, room - written);
            symbols = 0u;
            cut = true;
        }
        work.tokens[symbols++] = token;
    }
    if (cut)
        ++gBytePairWordsCut;
    return written + mergeWord(work, symbols, *_merges, out + written, room - written);
}

core::u32 Tokenizer::decode(const core::u32 *tokens, core::u32 count, char *out, core::u32 capacity) const noexcept
{
    if (_vocab == nullptr || tokens == nullptr || out == nullptr)
        return 0u;

    core::u32 written = 0u;
    for (core::u32 i = 0u; i < count; ++i)
    {
        core::u32 length = 0u;
        const char *const bytes = _vocab->text(tokens[i], length);
        if (bytes == nullptr)
            continue;
        // Stop rather than truncate mid-token: half a token's bytes is not a
        // shorter answer, it is a different one.
        if (written + length > capacity)
            break;
        for (core::u32 b = 0u; b < length; ++b)
            out[written++] = bytes[b];
    }
    return written;
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
