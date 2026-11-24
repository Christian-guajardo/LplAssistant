#include "fixtures/gguf_reference.hpp"

#include <lpl/convert/Rotary.hpp>
#include <lpl/testing/Test.hpp>

#include <limits>
#include <vector>

LPL_TEST_SUITE(convert_rotary);

namespace {

/**
 * @brief Whether @p words are the @p count words of @p expected.
 */
[[nodiscard]] bool sameWords(const std::vector<std::uint32_t> &words, const std::uint32_t *expected,
                             std::uint32_t count)
{
    if (words.size() != count)
        return false;
    for (std::uint32_t i = 0u; i < count; ++i)
    {
        if (words[i] != expected[i])
            return false;
    }
    return true;
}

/**
 * @brief The name of the refusal computing the table of @p base over @p dimensions gives, or "accepted".
 */
[[nodiscard]] std::string refusalOf(double base, std::uint32_t dimensions)
{
    std::vector<std::uint32_t> words;
    const std::optional<lpl::convert::Refusal> refusal = lpl::convert::rotaryTurnsPerPosition(base, dimensions, words);

    return refusal.has_value() ? refusal->name : "accepted";
}

} // namespace

/**
 * @brief SmolLM2's table, base 100000 over heads of 64 channels, is the 32 words worked apart in 60-digit decimal
 *        arithmetic; LLaMA's, base 10000, is its own 32.
 */
LPL_TEST(a_table_is_the_words_worked_apart_in_decimal_arithmetic)
{
    std::vector<std::uint32_t> smolLm2;
    std::vector<std::uint32_t> llama;

    test.check(!lpl::convert::rotaryTurnsPerPosition(gguf_reference::kSmolLm2RotaryBase,
                                                     gguf_reference::kRotaryDimensions, smolLm2)
                       .has_value() &&
                   sameWords(smolLm2, gguf_reference::kSmolLm2Turns, gguf_reference::kRotaryPairs),
               "base 100000 gives SmolLM2's 32 words, the first 683565276 units and the last 9796");
    test.check(!lpl::convert::rotaryTurnsPerPosition(gguf_reference::kLlamaRotaryBase,
                                                     gguf_reference::kRotaryDimensions, llama)
                       .has_value() &&
                   sameWords(llama, gguf_reference::kLlamaTurns, gguf_reference::kRotaryPairs),
               "base 10000 gives LLaMA's");
    test.check(smolLm2.size() == gguf_reference::kRotaryPairs && llama.size() == gguf_reference::kRotaryPairs &&
                   smolLm2[0] == llama[0] && smolLm2[1] != llama[1],
               "the two share the first pair, which turns a radian per position at any base, and no other");
}

/**
 * @brief A word within 10^-6 of a rounding tie is refused, naming its pair, where a base one below is not; a base that
 *        is not finite or not past 1, and a head of no or an odd count of channels, are refused as out of range.
 */
LPL_TEST(a_word_near_a_tie_or_a_base_out_of_range_is_refused_by_name)
{
    std::vector<std::uint32_t> words;
    const std::optional<lpl::convert::Refusal> nearTie =
        lpl::convert::rotaryTurnsPerPosition(gguf_reference::kNearTieBase, gguf_reference::kRotaryDimensions, words);

    test.check(
        nearTie.has_value() && nearTie->name == "rotary_near_tie" &&
            nearTie->detail.find("pair " + std::to_string(gguf_reference::kNearTiePair)) != std::string::npos,
        "the base whose pair 31 lies 7.7e-7 of a unit from a tie is refused as rotary_near_tie, naming the pair");
    test.check(refusalOf(gguf_reference::kNearTieBase - 1.0, gguf_reference::kRotaryDimensions) == "accepted",
               "the base below it is accepted");
    test.check(refusalOf(1.0, gguf_reference::kRotaryDimensions) == "metadata_out_of_range" &&
                   refusalOf(0.5, gguf_reference::kRotaryDimensions) == "metadata_out_of_range" &&
                   refusalOf(-10000.0, gguf_reference::kRotaryDimensions) == "metadata_out_of_range",
               "a base of 1 or below turns no pair slower than the first and is refused as metadata_out_of_range");
    test.check(refusalOf(std::numeric_limits<double>::quiet_NaN(), gguf_reference::kRotaryDimensions) ==
                       "metadata_out_of_range" &&
                   refusalOf(std::numeric_limits<double>::infinity(), gguf_reference::kRotaryDimensions) ==
                       "metadata_out_of_range",
               "and so is a base that is not finite");
    test.check(refusalOf(gguf_reference::kSmolLm2RotaryBase, 0u) == "metadata_out_of_range" &&
                   refusalOf(gguf_reference::kSmolLm2RotaryBase, 63u) == "metadata_out_of_range",
               "and a head of no channel or of an odd count of them");
    test.check(refusalOf(gguf_reference::kSmolLm2RotaryBase, 2u) == "accepted",
               "while a head of two channels has its one pair");
}
