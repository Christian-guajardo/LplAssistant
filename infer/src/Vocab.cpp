/**
 * @file Vocab.cpp
 * @brief The token table.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Vocab.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::infer {

namespace {

/** Buckets of one radix pass: a token that has ended, then each of the 256 byte values. */
constexpr core::u32 kRadixBuckets = 257u;

/**
 * Runs this short or shorter are sorted by comparison, where a radix pass costs more than it saves: Engineering Radix
 * Sort (McIlroy, Bostic and McIlroy, 1993, §3) finds that thresholds from 10 to 50 work well.
 */
constexpr core::u32 kComparisonSortMaximum = 32u;

/**
 * @brief Orders two token texts.
 *
 * Bytes compared as unsigned, then the shorter first. Unsigned matters: a token
 * carrying a high byte would sort before the ASCII ones on a target whose @c char is
 * signed and after them on a target whose @c char is not, and the binary search would
 * then find different tokens on the two sides of the gate.
 *
 * @param lhs       First text.
 * @param lhsLength Its length.
 * @param rhs       Second text.
 * @param rhsLength Its length.
 * @return Negative, zero or positive.
 */
core::i32 compareText(const char *lhs, core::u32 lhsLength, const char *rhs, core::u32 rhsLength) noexcept
{
    const core::u32 shared = lhsLength < rhsLength ? lhsLength : rhsLength;
    for (core::u32 i = 0u; i < shared; ++i)
    {
        const core::u32 a = static_cast<core::u8>(lhs[i]);
        const core::u32 b = static_cast<core::u8>(rhs[i]);
        if (a != b)
            return a < b ? -1 : 1;
    }
    if (lhsLength == rhsLength)
        return 0;
    return lhsLength < rhsLength ? -1 : 1;
}

/**
 * @brief The texts of a table's tokens, read the way the index orders them.
 */
struct TokenTexts {
    const char *blob;          ///< Concatenated token bytes.
    const VocabEntry *entries; ///< One per token, in identifier order.

    /**
     * @brief The radix digit of @p token at @p depth: 0 once its bytes have ended, else one more than its byte there,
     *        read as unsigned, so an ended token goes before every byte.
     */
    [[nodiscard]] core::u32 digitAt(core::u32 token, core::u32 depth) const noexcept
    {
        const VocabEntry entry = entries[token];
        return depth < entry.length ? 1u + static_cast<core::u8>(blob[entry.offset + depth]) : 0u;
    }

    /**
     * @brief Whether @p lhs comes before @p rhs, two tokens whose first @p depth bytes are the same: by the bytes after
     *        them, the shorter first, and the smaller identifier first when the bytes are the same.
     */
    [[nodiscard]] bool comesBefore(core::u32 lhs, core::u32 rhs, core::u32 depth) const noexcept
    {
        const VocabEntry left = entries[lhs];
        const VocabEntry right = entries[rhs];
        const core::i32 order = compareText(blob + left.offset + depth, left.length - depth,
                                            blob + right.offset + depth, right.length - depth);
        return order != 0 ? order < 0 : lhs < rhs;
    }
};

/**
 * @brief A run of the index whose tokens share their first @p depth bytes.
 */
struct IndexRange {
    core::u32 begin; ///< First rank.
    core::u32 end;   ///< One past the last rank.
    core::u32 depth; ///< Bytes every token of the run shares.
};

/**
 * @brief Sorts @p range by insertion, its tokens compared from its depth on.
 */
void insertionSort(core::u32 *order, const TokenTexts &texts, IndexRange range) noexcept
{
    for (core::u32 i = range.begin + 1u; i < range.end; ++i)
    {
        const core::u32 candidate = order[i];
        core::u32 j = i;
        while (j > range.begin && texts.comesBefore(candidate, order[j - 1u], range.depth))
        {
            order[j] = order[j - 1u];
            --j;
        }
        order[j] = candidate;
    }
}

/**
 * @brief Moves the token at @p root of the heap of @p size tokens at @p heap down to where it holds.
 */
void siftDown(core::u32 *heap, core::u32 root, core::u32 size, const TokenTexts &texts, core::u32 depth) noexcept
{
    while (root < size / 2u)
    {
        core::u32 child = 2u * root + 1u;
        if (child + 1u < size && texts.comesBefore(heap[child], heap[child + 1u], depth))
            ++child;
        if (!texts.comesBefore(heap[root], heap[child], depth))
            return;
        const core::u32 moved = heap[root];
        heap[root] = heap[child];
        heap[child] = moved;
        root = child;
    }
}

/**
 * @brief Sorts @p range by heap sort, its tokens compared from its depth on: n log n whatever the run holds.
 */
void heapSort(core::u32 *order, const TokenTexts &texts, IndexRange range) noexcept
{
    core::u32 *const heap = order + range.begin;
    const core::u32 size = range.end - range.begin;
    for (core::u32 root = size / 2u; root > 0u; --root)
        siftDown(heap, root - 1u, size, texts, range.depth);
    for (core::u32 last = size - 1u; last > 0u; --last)
    {
        const core::u32 largest = heap[0];
        heap[0] = heap[last];
        heap[last] = largest;
        siftDown(heap, 0u, last, texts, range.depth);
    }
}

/**
 * @brief Sorts @p range by comparing its tokens: by insertion when it is short, by heap sort else.
 */
void comparisonSort(core::u32 *order, const TokenTexts &texts, IndexRange range) noexcept
{
    if (range.end - range.begin <= kComparisonSortMaximum)
        insertionSort(order, texts, range);
    else
        heapSort(order, texts, range);
}

/**
 * @brief The bucket counts of one radix pass, then where each bucket's next token goes.
 */
struct RadixBuckets {
    core::u32 ends[kRadixBuckets]; ///< One past each bucket's last rank, once placed.
    core::u32 next[kRadixBuckets]; ///< The rank the bucket's next token goes to.
};

/**
 * @brief Counts the tokens of @p range by their digit at its depth.
 * @return The bucket count of the first token's digit, which is the range's size when every token has it.
 */
core::u32 countDigits(const core::u32 *order, const TokenTexts &texts, IndexRange range, RadixBuckets &buckets) noexcept
{
    for (core::u32 digit = 0u; digit < kRadixBuckets; ++digit)
        buckets.ends[digit] = 0u;
    for (core::u32 i = range.begin; i < range.end; ++i)
        ++buckets.ends[texts.digitAt(order[i], range.depth)];
    return buckets.ends[texts.digitAt(order[range.begin], range.depth)];
}

/**
 * @brief Moves every token of @p range into its bucket, the buckets in digit order, in place: an American flag sort's
 *        pass, whose counts @p buckets holds.
 */
void placeInBuckets(core::u32 *order, const TokenTexts &texts, IndexRange range, RadixBuckets &buckets) noexcept
{
    core::u32 start = range.begin;
    for (core::u32 digit = 0u; digit < kRadixBuckets; ++digit)
    {
        buckets.next[digit] = start;
        start += buckets.ends[digit];
        buckets.ends[digit] = start;
    }
    for (core::u32 digit = 0u; digit < kRadixBuckets; ++digit)
    {
        while (buckets.next[digit] < buckets.ends[digit])
        {
            core::u32 token = order[buckets.next[digit]];
            core::u32 home = texts.digitAt(token, range.depth);
            while (home != digit)
            {
                const core::u32 displaced = order[buckets.next[home]];
                order[buckets.next[home]] = token;
                ++buckets.next[home];
                token = displaced;
                home = texts.digitAt(token, range.depth);
            }
            order[buckets.next[digit]] = token;
            ++buckets.next[digit];
        }
    }
}

/**
 * @brief Sorts @p range or splits it into buckets by its tokens' next byte.
 *
 * A range whose tokens all have the same next byte goes one byte deeper without moving. The tokens that end at the
 * range's depth are the same bytes, so they are sorted by identifier at once.
 *
 * @return The rank where the range's unsorted buckets start, at the depth @p range then holds, or @p range's end when
 *         nothing is left to sort.
 */
core::u32 sortOrSplit(core::u32 *order, const TokenTexts &texts, IndexRange &range, RadixBuckets &buckets) noexcept
{
    const core::u32 size = range.end - range.begin;
    for (;;)
    {
        if (size <= kComparisonSortMaximum || range.depth == kMaxTokenBytes)
        {
            comparisonSort(order, texts, range);
            return range.end;
        }
        if (countDigits(order, texts, range, buckets) != size)
            break;
        if (texts.digitAt(order[range.begin], range.depth) == 0u)
        {
            comparisonSort(order, texts, range);
            return range.end;
        }
        ++range.depth;
    }

    const core::u32 ended = buckets.ends[0];
    placeInBuckets(order, texts, range, buckets);
    comparisonSort(order, texts, IndexRange{range.begin, range.begin + ended, range.depth});
    return range.begin + ended;
}

/**
 * @brief One past the last rank from @p cursor whose token has the same digit at @p depth as the token at @p cursor,
 *        in a run grouped by that digit in ascending order.
 */
core::u32 endOfBucket(const core::u32 *order, const TokenTexts &texts, core::u32 cursor, core::u32 end,
                      core::u32 depth) noexcept
{
    const core::u32 digit = texts.digitAt(order[cursor], depth);
    core::u32 low = cursor + 1u;
    core::u32 high = end;
    while (low < high)
    {
        const core::u32 middle = low + (high - low) / 2u;
        if (texts.digitAt(order[middle], depth) <= digit)
            low = middle + 1u;
        else
            high = middle;
    }
    return low;
}

/**
 * @brief The runs a pass split and left to sort, one bucket at a time, the deepest last.
 *
 * Each run waits at a depth deeper than the one under it, and only a run less than kMaxTokenBytes deep is split, so
 * kMaxTokenBytes runs are the most that ever wait.
 */
struct PendingRuns {
    IndexRange runs[kMaxTokenBytes]; ///< Buckets still to sort, grouped by their digit at the run's depth.
    core::u32 count{0u};             ///< Runs waiting.

    /**
     * @brief Takes the next bucket of the deepest run still waiting, one byte deeper than that run.
     * @return false when no run waits any more.
     */
    [[nodiscard]] bool takeNextBucket(const core::u32 *order, const TokenTexts &texts, IndexRange &out) noexcept
    {
        while (count > 0u && runs[count - 1u].begin == runs[count - 1u].end)
            --count;
        if (count == 0u)
            return false;

        IndexRange &run = runs[count - 1u];
        const core::u32 bucketEnd = endOfBucket(order, texts, run.begin, run.end, run.depth);
        out = IndexRange{run.begin, bucketEnd, run.depth + 1u};
        run.begin = bucketEnd;
        return true;
    }
};

/**
 * @brief Fills @p order with the identifiers 0 to @p count - 1, sorted by their tokens' text, the smaller identifier
 *        first when two texts are the same.
 *
 * A most-significant-digit radix sort, in place: each pass splits a run by one more byte, until the run is short
 * enough to compare, holds tokens that ended together, or reaches kMaxTokenBytes, and comparing finishes it. A tie
 * goes to the smaller identifier by that comparison, never by the order a pass left, so the order is a total one, no
 * pass needs to be stable, and no scratch copy of the index is claimed. The stack holds the buckets, 2 KiB, and at
 * most kMaxTokenBytes pending runs.
 */
void sortByText(core::u32 *order, const char *blob, const VocabEntry *entries, core::u32 count) noexcept
{
    for (core::u32 i = 0u; i < count; ++i)
        order[i] = i;

    const TokenTexts texts{blob, entries};
    RadixBuckets buckets;
    PendingRuns pending;
    IndexRange range{0u, count, 0u};
    do
    {
        const core::u32 unsorted = sortOrSplit(order, texts, range, buckets);
        if (unsorted < range.end)
            pending.runs[pending.count++] = IndexRange{unsorted, range.end, range.depth};
    } while (pending.takeNextBucket(order, texts, range));
}

} // namespace

bool Vocab::build(TensorArena &arena, const char *blob, core::u32 blobBytes, const VocabEntry *entries, core::u32 count)
{
    if (blob == nullptr || entries == nullptr || count == 0u)
        return false;

    char *const blobCopy = arena.claim<char>(blobBytes);
    VocabEntry *const entryCopy = arena.claim<VocabEntry>(count);
    if (blobCopy == nullptr || entryCopy == nullptr)
        return false;

    for (core::u32 i = 0u; i < blobBytes; ++i)
        blobCopy[i] = blob[i];
    for (core::u32 i = 0u; i < count; ++i)
        entryCopy[i] = entries[i];
    return borrow(arena, blobCopy, blobBytes, entryCopy, count);
}

bool Vocab::borrow(TensorArena &arena, const char *blob, core::u32 blobBytes, const VocabEntry *entries,
                   core::u32 count)
{
    if (blob == nullptr || entries == nullptr || count == 0u)
        return false;

    core::u32 *const order = arena.claim<core::u32>(count);
    if (order == nullptr)
        return false;

    sortByText(order, blob, entries, count);
    _blob = blob;
    _entries = entries;
    _sorted = order;
    _count = count;
    _blobBytes = blobBytes;
    return true;
}

const char *Vocab::text(core::u32 token, core::u32 &outLength) const noexcept
{
    outLength = 0u;
    if (_blob == nullptr || token >= _count)
        return nullptr;
    outLength = _entries[token].length;
    return _blob + _entries[token].offset;
}

core::u32 Vocab::find(const char *bytes, core::u32 length) const noexcept
{
    if (_blob == nullptr || bytes == nullptr || length == 0u)
        return kNoToken;

    core::u32 low = 0u;
    core::u32 high = _count;
    while (low < high)
    {
        const core::u32 middle = low + (high - low) / 2u;
        const core::u32 token = _sorted[middle];
        const core::i32 order = compareText(_blob + _entries[token].offset, _entries[token].length, bytes, length);
        if (order == 0)
            return token;
        if (order < 0)
            low = middle + 1u;
        else
            high = middle;
    }
    return kNoToken;
}

core::u32 Vocab::longestPrefix(const char *bytes, core::u32 available, core::u32 &outLength) const noexcept
{
    outLength = 0u;
    if (bytes == nullptr || available == 0u)
        return kNoToken;

    // Longest first, so the first hit is the answer and there is no tie to break.
    core::u32 limit = available < kMaxTokenBytes ? available : kMaxTokenBytes;
    for (core::u32 length = limit; length > 0u; --length)
    {
        const core::u32 token = find(bytes, length);
        if (token != kNoToken)
        {
            outLength = length;
            return token;
        }
    }
    return kNoToken;
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
