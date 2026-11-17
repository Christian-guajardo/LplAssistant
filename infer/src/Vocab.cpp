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
 * @brief Fills @p order with the identifiers 0 to @p count - 1, sorted by their tokens' text.
 *
 * An insertion sort: the table is sorted once at load and never again, so the cost is paid before the first token,
 * and a faster sort would be a second thing to keep deterministic for no gain the gate can see.
 */
void sortByText(core::u32 *order, const char *blob, const VocabEntry *entries, core::u32 count) noexcept
{
    for (core::u32 i = 0u; i < count; ++i)
        order[i] = i;

    for (core::u32 i = 1u; i < count; ++i)
    {
        const core::u32 candidate = order[i];
        const char *const candidateText = blob + entries[candidate].offset;
        const core::u32 candidateLength = entries[candidate].length;

        core::u32 j = i;
        while (j > 0u)
        {
            const core::u32 previous = order[j - 1u];
            if (compareText(blob + entries[previous].offset, entries[previous].length, candidateText,
                            candidateLength) <= 0)
                break;
            order[j] = previous;
            --j;
        }
        order[j] = candidate;
    }
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
