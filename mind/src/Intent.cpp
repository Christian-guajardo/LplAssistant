/**
 * @file Intent.cpp
 * @brief Implementation of what the sovereign asked for, parsed and bounded.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/mind/Intent.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/Fold.hpp>

namespace lpl::mind {

namespace {

/**
 * @brief Is this byte one an intent may carry?
 *
 * Space through tilde, and nothing else. Deliberately narrower than "not a control
 * character": a byte above 0x7E is the middle of a multi-byte sequence this module
 * has no encoding for, and half a character in a prompt is worse than none.
 *
 * @param value Byte to test.
 * @return true when it survives parsing.
 */
constexpr bool acceptable(core::u8 value) noexcept { return value >= 0x20u && value <= 0x7Eu; }

/**
 * @brief Lower-cases one byte, for keyword matching only.
 * @param value Byte.
 * @return The lower-case form of an ASCII letter, or @p value.
 */
constexpr char lowered(char value) noexcept
{
    return (value >= 'A' && value <= 'Z') ? static_cast<char>(value - 'A' + 'a') : value;
}

/**
 * @brief Does the text begin with this keyword, followed by a break?
 *
 * The break matters: without it "stopwatch" would be read as a Stop, which is the
 * kind of misreading that only shows up when somebody says the wrong word at the
 * wrong moment.
 *
 * @param text    Parsed text.
 * @param bytes   Its length.
 * @param keyword Null-terminated lower-case keyword.
 * @return true when @p text opens with it.
 */
bool opensWith(const char *text, core::u32 bytes, const char *keyword) noexcept
{
    core::u32 i = 0u;
    for (; keyword[i] != '\0'; ++i)
    {
        if (i >= bytes || lowered(text[i]) != keyword[i])
            return false;
    }
    return i == bytes || text[i] == ' ';
}

/**
 * @brief Is this word one that carries no subject?
 *
 * A closed list, written out rather than derived. Deriving "unimportant word" from
 * frequency would make the topic of an utterance depend on what had been said before
 * it, so the same question would file under different topics on two machines that had
 * heard different conversations — which is the end of any replayable transcript.
 *
 * @param text  Word bytes.
 * @param count How many.
 * @return true when it should be skipped while looking for the subject.
 */
bool functionWord(const char *text, core::u32 count) noexcept
{
    static const char *const kFunctionWords[] = {"is",  "are", "was", "were", "the", "a",
                                                 "an",  "do",  "does", "did", "of",  "to",
                                                 "my",  "your", "it",  "that", "this"};
    for (const char *candidate : kFunctionWords)
    {
        core::u32 i = 0u;
        for (; candidate[i] != '\0'; ++i)
        {
            if (i >= count || lowered(text[i]) != candidate[i])
                break;
        }
        if (candidate[i] == '\0' && i == count)
            return true;
    }
    return false;
}

} // namespace

core::u32 topicOf(const char *text, core::u32 count) noexcept
{
    if (text == nullptr || count == 0u)
        return 0u;

    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < count; ++i)
        foldWord(hash, static_cast<core::u32>(static_cast<core::u8>(lowered(text[i]))));

    /* Zero is reserved for "no topic", so a word that happens to hash to it is nudged.
       One value being very slightly more likely than the rest costs nothing; a topic
       that silently means "match anything" would let a lookup return the whole store. */
    return hash == 0u ? 1u : hash;
}

Intent parseIntent(const core::u8 *bytes, core::u32 count) noexcept
{
    Intent intent;
    if (bytes == nullptr)
        count = 0u;

    /* Leading blanks are skipped rather than stored, because the keyword test reads
       from the front and a single stray space would make every command Unknown. */
    core::u32 start = 0u;
    while (start < count && bytes[start] == static_cast<core::u8>(' '))
        ++start;

    for (core::u32 i = start; i < count; ++i)
    {
        if (!acceptable(bytes[i]))
        {
            ++intent.droppedBytes;
            continue;
        }
        if (intent.bytes >= kIntentBytes)
        {
            ++intent.truncatedBytes;
            continue;
        }
        intent.text[intent.bytes++] = static_cast<char>(bytes[i]);
    }

    /* Stop is tested first, and that ordering is the point rather than an accident:
       an utterance that both asks something and says stop is a sovereign changing
       their mind mid-sentence, and the later half is the one that counts. */
    if (opensWith(intent.text, intent.bytes, "stop") || opensWith(intent.text, intent.bytes, "cancel"))
        intent.kind = IntentKind::Stop;
    else if (opensWith(intent.text, intent.bytes, "no") || opensWith(intent.text, intent.bytes, "actually"))
        intent.kind = IntentKind::Correction;
    else if (opensWith(intent.text, intent.bytes, "what") || opensWith(intent.text, intent.bytes, "where") ||
             opensWith(intent.text, intent.bytes, "who") || opensWith(intent.text, intent.bytes, "why") ||
             opensWith(intent.text, intent.bytes, "how") || opensWith(intent.text, intent.bytes, "when"))
        intent.kind = IntentKind::Question;
    else if (intent.bytes > 0u)
        intent.kind = IntentKind::Command;

    /* The topic is the first word that CARRIES one. Filing "what is the reactor
       pressure" under "what" would put every question in the project in one bucket;
       filing it under "is" is the same mistake one word later, which is what the first
       version of this did — the structured filter then matched nothing at all, and a
       filter that matches nothing looks exactly like a store that holds nothing.
       Function words are skipped until something substantive turns up. */
    core::u32 wordStart = 0u;
    core::u32 wordEnd = 0u;
    core::u32 cursor = 0u;
    bool skipLeadingKeyword = intent.kind == IntentKind::Question || intent.kind == IntentKind::Correction;

    while (cursor < intent.bytes)
    {
        while (cursor < intent.bytes && intent.text[cursor] == ' ')
            ++cursor;
        const core::u32 start = cursor;
        while (cursor < intent.bytes && intent.text[cursor] != ' ')
            ++cursor;
        if (cursor == start)
            break;

        wordStart = start;
        wordEnd = cursor;

        if (skipLeadingKeyword)
        {
            skipLeadingKeyword = false; // The interrogative itself.
            continue;
        }
        if (!functionWord(intent.text + start, cursor - start))
            break;
    }
    intent.topic = topicOf(intent.text + wordStart, wordEnd - wordStart);
    return intent;
}

core::u32 foldIntent(const Intent &intent, core::u32 hash) noexcept
{
    foldWord(hash, static_cast<core::u32>(intent.kind));
    foldWord(hash, intent.bytes);
    foldBytes(hash, reinterpret_cast<const core::u8 *>(intent.text), intent.bytes);
    foldWord(hash, intent.topic);
    foldWord(hash, intent.droppedBytes);
    foldWord(hash, intent.truncatedBytes);
    return hash;
}

} // namespace lpl::mind

#endif // LPL_HAS_FOUNDATION
