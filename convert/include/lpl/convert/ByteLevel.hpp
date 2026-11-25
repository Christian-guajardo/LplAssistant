/**
 * @file ByteLevel.hpp
 * @brief GPT-2's byte-level alphabet: each byte written as one printable character, and back.
 *
 * A GGUF of a byte-level BPE vocabulary stores each NORMAL token as the characters of its bytes in this alphabet: a
 * space is `Ġ` (U+0120), a newline `Ċ` (U+010A). The alphabet is openai/gpt-2's bytes_to_unicode: the printable bytes
 * `!` to `~`, `¡` to `¬` and `®` to `ÿ` stand for themselves, and the 68 others take U+0100 onwards in their order.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CONVERT_BYTELEVEL_HPP
#    define LPL_LPL_CONVERT_BYTELEVEL_HPP

#    include <cstdint>
#    include <optional>
#    include <string>
#    include <string_view>

namespace lpl::convert {

/**
 * @brief The character GPT-2's alphabet writes @p byte as.
 * @param byte A byte.
 * @return Its code point, from U+0021 to U+0143.
 */
[[nodiscard]] std::uint32_t byteLevelCodePoint(std::uint8_t byte) noexcept;

/**
 * @brief The bytes a token written in GPT-2's alphabet stands for.
 * @param text The token, UTF-8.
 * @return One byte per character; nothing when @p text is not UTF-8 or holds a character outside the alphabet.
 */
[[nodiscard]] std::optional<std::string> byteLevelBytes(std::string_view text);

} // namespace lpl::convert

#endif /* LPL_LPL_CONVERT_BYTELEVEL_HPP */
