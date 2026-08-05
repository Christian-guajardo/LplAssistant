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

namespace lpl::infer {

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
        core::u32 length = 0u;
        const core::u32 token = _vocab->longestPrefix(text + cursor, bytes - cursor, length);
        if (token == kNoToken || length == 0u)
        {
            ++outSkipped;
            ++cursor;
            continue;
        }
        out[written++] = token;
        cursor += length;
    }
    return written;
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
