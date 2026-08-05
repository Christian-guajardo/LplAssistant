/**
 * @file GrammarSampler.cpp
 * @brief Constrained decoding against a grammar.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/GrammarSampler.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::infer {

namespace {

/**
 * @brief Length of a null-terminated string, bounded.
 * @param text  The string.
 * @param limit Longest accepted.
 * @return The length, or @p limit when it is longer.
 */
core::u32 boundedLength(const char *text, core::u32 limit) noexcept
{
    core::u32 length = 0u;
    while (length < limit && text[length] != '\0')
        ++length;
    return length;
}

} // namespace

bool Grammar::build(TensorArena &arena, const char *const *phrases, core::u32 count)
{
    if (phrases == nullptr || count == 0u || count > kMaxGrammarPhrases)
        return false;

    core::u32 total = 0u;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const core::u32 length = boundedLength(phrases[i], 4096u);
        if (length == 0u)
            return false;
        total += length;
    }

    char *const blob = arena.claim<char>(total);
    core::u32 *const offsets = arena.claim<core::u32>(count);
    core::u32 *const lengths = arena.claim<core::u32>(count);
    if (blob == nullptr || offsets == nullptr || lengths == nullptr)
        return false;

    core::u32 cursor = 0u;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const core::u32 length = boundedLength(phrases[i], 4096u);
        offsets[i] = cursor;
        lengths[i] = length;
        for (core::u32 b = 0u; b < length; ++b)
            blob[cursor + b] = phrases[i][b];
        cursor += length;
    }

    _blob = blob;
    _offsets = offsets;
    _lengths = lengths;
    _count = count;
    return true;
}

GrammarState Grammar::start() const noexcept
{
    GrammarState state{};
    if (_count == 0u)
        return state;
    // All ones over _count bits. Written as a shift on 64 bits so a full thirty-two
    // phrases does not shift a 32-bit word by its own width, which is undefined.
    state.candidates = static_cast<core::u32>((core::u64{1} << _count) - 1u);
    state.depth = 0u;
    return state;
}

bool Grammar::accepts(const GrammarState &state, const char *bytes, core::u32 length, GrammarState &outNext) const
    noexcept
{
    if (_blob == nullptr || bytes == nullptr || length == 0u || state.candidates == 0u)
        return false;

    core::u32 surviving = 0u;
    for (core::u32 i = 0u; i < _count; ++i)
    {
        if ((state.candidates & (1u << i)) == 0u)
            continue;
        if (state.depth + length > _lengths[i])
            continue;

        bool matches = true;
        for (core::u32 b = 0u; b < length && matches; ++b)
            matches = _blob[_offsets[i] + state.depth + b] == bytes[b];
        if (matches)
            surviving |= (1u << i);
    }

    if (surviving == 0u)
        return false;

    outNext.candidates = surviving;
    outNext.depth = state.depth + length;
    return true;
}

bool Grammar::complete(const GrammarState &state) const noexcept
{
    if (_blob == nullptr)
        return false;
    for (core::u32 i = 0u; i < _count; ++i)
        if ((state.candidates & (1u << i)) != 0u && _lengths[i] == state.depth)
            return true;
    return false;
}

core::u32 maskAllowedTokens(const Grammar &grammar, const GrammarState &state, const Vocab &vocab,
                            bool *allowed) noexcept
{
    if (allowed == nullptr)
        return 0u;

    core::u32 admitted = 0u;
    for (core::u32 t = 0u; t < vocab.size(); ++t)
    {
        core::u32 length = 0u;
        const char *const bytes = vocab.text(t, length);
        GrammarState next{};
        allowed[t] = bytes != nullptr && grammar.accepts(state, bytes, length, next);
        if (allowed[t])
            ++admitted;
    }
    return admitted;
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
