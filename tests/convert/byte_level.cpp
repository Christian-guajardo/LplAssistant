#include "fixtures/gguf_reference.hpp"

#include <lpl/convert/ByteLevel.hpp>
#include <lpl/testing/Test.hpp>

#include <string>

LPL_TEST_SUITE(convert_byte_level);

namespace {

/**
 * @brief @p codePoint in UTF-8.
 */
[[nodiscard]] std::string utf8(std::uint32_t codePoint)
{
    std::string text;

    if (codePoint < 0x80u)
    {
        text.push_back(static_cast<char>(codePoint));
        return text;
    }
    if (codePoint < 0x800u)
    {
        text.push_back(static_cast<char>(0xC0u | (codePoint >> 6)));
        text.push_back(static_cast<char>(0x80u | (codePoint & 0x3Fu)));
        return text;
    }
    text.push_back(static_cast<char>(0xE0u | (codePoint >> 12)));
    text.push_back(static_cast<char>(0x80u | ((codePoint >> 6) & 0x3Fu)));
    text.push_back(static_cast<char>(0x80u | (codePoint & 0x3Fu)));
    return text;
}

} // namespace

/**
 * @brief Every byte is written as the character openai/gpt-2's bytes_to_unicode gives it, and each of those characters
 *        reads back as its byte alone.
 */
LPL_TEST(every_byte_is_the_character_gpt_2_s_alphabet_writes_it_as_and_back)
{
    std::uint32_t written = 0u;
    std::uint32_t readBack = 0u;

    for (std::uint32_t byte = 0u; byte < 256u; ++byte)
    {
        const std::uint32_t codePoint = lpl::convert::byteLevelCodePoint(static_cast<std::uint8_t>(byte));
        written += codePoint == gguf_reference::kByteLevelCodePoints[byte] ? 1u : 0u;
        const std::optional<std::string> bytes =
            lpl::convert::byteLevelBytes(utf8(gguf_reference::kByteLevelCodePoints[byte]));
        readBack +=
            bytes.has_value() && bytes->size() == 1u && static_cast<std::uint8_t>((*bytes)[0]) == byte ? 1u : 0u;
    }
    test.check(written == 256u, "each of the 256 bytes is written as the tool's alphabet writes it");
    test.check(readBack == 256u, "and each of those 256 characters reads back as its byte alone");
    test.measure("bytes_written", written);
    test.measure("bytes_read_back", readBack);
}

/**
 * @brief A token reads back as its bytes, one per character; text outside the alphabet or not UTF-8 is refused.
 */
LPL_TEST(a_token_reads_back_as_its_bytes_and_text_outside_the_alphabet_does_not)
{
    const std::optional<std::string> spaced = lpl::convert::byteLevelBytes("\xC4\xA0hello");
    const std::optional<std::string> newline = lpl::convert::byteLevelBytes("\xC4\x8A");
    const std::optional<std::string> empty = lpl::convert::byteLevelBytes("");

    test.check(spaced.has_value() && *spaced == " hello", "Ġhello is a space then hello");
    test.check(newline.has_value() && *newline == "\n", "Ċ is a newline");
    test.check(empty.has_value() && empty->empty(), "no character is no byte");
    test.check(!lpl::convert::byteLevelBytes("a\xE4\xB8\x80").has_value(), "a character past U+0143 is refused");
    test.check(!lpl::convert::byteLevelBytes(" ").has_value() && !lpl::convert::byteLevelBytes("\xC2\xAD").has_value(),
               "and so are the space and U+00AD, which the alphabet writes as other characters");
    test.check(!lpl::convert::byteLevelBytes("\xC4").has_value() && !lpl::convert::byteLevelBytes("\x80").has_value(),
               "a sequence cut short or a lone continuation byte is not UTF-8");
    test.check(!lpl::convert::byteLevelBytes("\xA1").has_value() && !lpl::convert::byteLevelBytes("a\xBF"
                                                                                                  "b")
                                                                         .has_value(),
               "even one whose value, read as a code point, the alphabet holds: U+00A1 is written as two bytes");
    test.check(!lpl::convert::byteLevelBytes("\xC0\xA1").has_value(), "nor is an overlong encoding of !");
}
