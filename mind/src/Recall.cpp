/**
 * @file Recall.cpp
 * @brief Implementation of the interface to the knowledge side.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/mind/Recall.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/Fold.hpp>
#    include <lpl/mind/Intent.hpp>

namespace lpl::mind {

namespace {

/**
 * @brief Counts the set bits in a word.
 *
 * Written out rather than delegated to a compiler builtin. A builtin may expand to a
 * POPCNT instruction on one target and a library call on another, and this module is
 * linked into a kernel that has no library to call. The SWAR form is a handful of
 * shifts and gives the same answer everywhere.
 *
 * @param value Word to count.
 * @return Bits set in it.
 */
constexpr core::u32 populationCount(core::u32 value) noexcept
{
    value = value - ((value >> 1) & 0x55555555u);
    value = (value & 0x33333333u) + ((value >> 2) & 0x33333333u);
    value = (value + (value >> 4)) & 0x0F0F0F0Fu;
    return (value * 0x01010101u) >> 24;
}

/**
 * @brief Does this byte belong to a word?
 * @param value Byte to test.
 * @return true for letters and digits.
 */
constexpr bool wordByte(char value) noexcept
{
    return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9');
}

} // namespace

TextSignature signatureOf(const char *text, core::u32 count) noexcept
{
    TextSignature signature;
    if (text == nullptr)
        return signature;

    core::u32 start = 0u;
    while (start < count)
    {
        while (start < count && !wordByte(text[start]))
            ++start;
        core::u32 end = start;
        while (end < count && wordByte(text[end]))
            ++end;
        if (end == start)
            break;

        /* One word sets one bit, chosen by its topic hash. Reusing topicOf rather than
           hashing here is deliberate: a note filed under a topic and a query matching
           its words must agree on what a word IS, down to the case folding, or a
           lookup finds nothing while every part looks correct on its own. */
        const core::u32 bit = topicOf(text + start, end - start) & 63u;
        if (bit < 32u)
            signature.low |= (1u << bit);
        else
            signature.high |= (1u << (bit - 32u));

        start = end;
    }
    return signature;
}

math::Fixed32 similarity(const TextSignature &left, const TextSignature &right) noexcept
{
    const core::u32 shared = populationCount(left.low & right.low) + populationCount(left.high & right.high);
    const core::u32 total = populationCount(left.low | right.low) + populationCount(left.high | right.high);
    if (total == 0u)
        return math::Fixed32::zero();

    /* Shifted into Q16.16 BEFORE the divide, so the quotient keeps its fraction. The
       other order is an integer division that is zero for everything short of a
       perfect match, which would look like a working filter that never matches. */
    return math::Fixed32::fromRaw(static_cast<core::i32>((shared << 16) / total));
}

core::u32 recall(const MemoryStore &store, core::u32 topic, const char *query, core::u32 queryBytes,
                 math::Fixed32 minimumScore, RecallHit *out, core::u32 capacity) noexcept
{
    if (out == nullptr || capacity == 0u)
        return 0u;

    const TextSignature wanted = signatureOf(query, queryBytes);
    core::u32 written = 0u;

    for (core::u32 i = 0u; i < store.count(); ++i)
    {
        const MemoryNote &note = store.note(i);

        // Structured filter, first and on its own terms: a note about something else
        // is not a weak match, it is not a match.
        if (topic != 0u && note.topic != topic)
            continue;

        const math::Fixed32 score = similarity(wanted, signatureOf(note.text, note.bytes));
        if (score < minimumScore)
            continue;

        /* Insertion sort into the output. The result set is small by construction — it
           is what a demon is about to read — so the simple form is also the fast one,
           and it puts the tie rule in view instead of leaving it to whichever sort was
           reached for. A hit that cannot beat the weakest kept one is dropped, which is
           what makes this a top-N rather than a truncation of arrival order. */
        if (written == capacity && !(out[capacity - 1u].score < score))
            continue;

        core::u32 position = written < capacity ? written : capacity - 1u;
        while (position > 0u && out[position - 1u].score < score)
        {
            out[position] = out[position - 1u];
            --position;
        }
        out[position] = RecallHit{i, score};
        if (written < capacity)
            ++written;
    }
    return written;
}

core::u32 foldRecall(const RecallHit *hits, core::u32 count, core::u32 hash) noexcept
{
    foldWord(hash, count);
    for (core::u32 i = 0u; i < count; ++i)
    {
        foldWord(hash, hits[i].index);
        foldWord(hash, static_cast<core::u32>(hits[i].score.raw()));
    }
    return hash;
}

} // namespace lpl::mind

#endif // LPL_HAS_FOUNDATION
