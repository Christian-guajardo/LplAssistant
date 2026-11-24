#include <lpl/convert/ByteLevel.hpp>

#include <array>

namespace lpl::convert {

namespace {

constexpr std::uint32_t kBytes = 256u;
constexpr std::uint32_t kShiftedBase = 0x100u;   ///< The first code point a byte outside the printable runs takes.
constexpr std::uint32_t kNoByte = 0xFFFFFFFFu;   ///< A code point that stands for no byte.
constexpr std::uint32_t kLastCodePoint = 0x143u; ///< The last code point the alphabet uses: U+0100 plus 67.

/**
 * @brief Whether @p byte is in one of the three runs GPT-2 writes as themselves: `!` to `~`, `¡` to `¬`, `®` to `ÿ`.
 */
[[nodiscard]] constexpr bool standsForItself(std::uint32_t byte) noexcept
{
    return (byte >= 0x21u && byte <= 0x7Eu) || (byte >= 0xA1u && byte <= 0xACu) || (byte >= 0xAEu && byte <= 0xFFu);
}

/**
 * @brief The alphabet, by the byte: a byte of the printable runs is its own code point, each other byte the next code
 *        point from U+0100 in the bytes' order.
 */
[[nodiscard]] constexpr std::array<std::uint32_t, kBytes> alphabet() noexcept
{
    std::array<std::uint32_t, kBytes> codePoints{};
    std::uint32_t shifted = 0u;

    for (std::uint32_t byte = 0u; byte < kBytes; ++byte)
        codePoints[byte] = standsForItself(byte) ? byte : kShiftedBase + shifted++;
    return codePoints;
}

/**
 * @brief The alphabet read back: the byte each code point up to kLastCodePoint stands for, kNoByte for none.
 */
[[nodiscard]] constexpr std::array<std::uint32_t, kLastCodePoint + 1u> inverseAlphabet() noexcept
{
    std::array<std::uint32_t, kLastCodePoint + 1u> bytes{};
    const std::array<std::uint32_t, kBytes> codePoints = alphabet();

    for (std::uint32_t &byte : bytes)
        byte = kNoByte;
    for (std::uint32_t byte = 0u; byte < kBytes; ++byte)
        bytes[codePoints[byte]] = byte;
    return bytes;
}

constexpr std::array<std::uint32_t, kBytes> kAlphabet = alphabet();
constexpr std::array<std::uint32_t, kLastCodePoint + 1u> kInverse = inverseAlphabet();

/**
 * @brief Whether @p byte is a UTF-8 continuation byte, 10xxxxxx.
 */
[[nodiscard]] constexpr bool continues(unsigned char byte) noexcept { return (byte & 0xC0u) == 0x80u; }

/**
 * @brief Decodes the one or two bytes of the character at @p at of @p text, the only lengths the alphabet's code points
 *        take, and advances @p at past it.
 * @return The code point; kNoByte for a sequence that is not UTF-8, is overlong, or is longer than two bytes.
 */
[[nodiscard]] std::uint32_t nextCodePoint(std::string_view text, std::size_t &at) noexcept
{
    const unsigned char lead = static_cast<unsigned char>(text[at]);
    if (lead < 0x80u)
    {
        ++at;
        return lead;
    }
    if ((lead & 0xE0u) != 0xC0u || lead < 0xC2u || at + 1u >= text.size() ||
        !continues(static_cast<unsigned char>(text[at + 1u])))
        return kNoByte;
    const std::uint32_t codePoint =
        (static_cast<std::uint32_t>(lead & 0x1Fu) << 6) | (static_cast<unsigned char>(text[at + 1u]) & 0x3Fu);
    at += 2u;
    return codePoint;
}

} // namespace

std::uint32_t byteLevelCodePoint(std::uint8_t byte) noexcept { return kAlphabet[byte]; }

std::optional<std::string> byteLevelBytes(std::string_view text)
{
    std::string bytes;
    std::size_t at = 0u;

    bytes.reserve(text.size());
    while (at < text.size())
    {
        const std::uint32_t codePoint = nextCodePoint(text, at);
        if (codePoint > kLastCodePoint || kInverse[codePoint] == kNoByte)
            return std::nullopt;
        bytes.push_back(static_cast<char>(kInverse[codePoint]));
    }
    return bytes;
}

} // namespace lpl::convert
