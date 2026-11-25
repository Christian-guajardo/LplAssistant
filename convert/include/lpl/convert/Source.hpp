/**
 * @file Source.hpp
 * @brief A model as a GGUF file holds it, read into plain values: what the converter decides from.
 *
 * The reader of a file (ModelFile) fills a SourceModel and the converter reads nothing else, so that every decision
 * the converter takes is tested on a model a test builds in memory, without a file. A key keeps the type the file gave
 * it, so that a key of the wrong type is refused by name instead of being read as another.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CONVERT_SOURCE_HPP
#    define LPL_LPL_CONVERT_SOURCE_HPP

#    include <cstdint>
#    include <map>
#    include <string>
#    include <vector>

namespace lpl::convert {

/**
 * @struct Refusal
 * @brief Why a conversion stopped: a stable name, and what it refuses.
 */
struct Refusal {
    std::string name;   ///< Stable and snake_case, as scripts and tests match it: `tensor_missing`.
    std::string detail; ///< What is refused, with the values that refused it: the key, the tensor, the token.
};

/**
 * @enum MetadataKind
 * @brief The type a key's value has in the file.
 */
enum class MetadataKind : std::uint8_t {
    UnsignedWord,   ///< A 32-bit unsigned integer.
    Real,           ///< A 32-bit float.
    Boolean,        ///< A boolean.
    Text,           ///< A string.
    TextList,       ///< An array of strings.
    SignedWordList, ///< An array of 32-bit signed integers.
    Other,          ///< Any other type, kept by its name only.
};

/**
 * @struct MetadataValue
 * @brief One key's value, in the member its kind names.
 */
struct MetadataValue {
    MetadataKind kind{MetadataKind::Other}; ///< Which member holds the value.
    std::uint32_t unsignedWord{0u};         ///< For UnsignedWord.
    float real{0.0f};                       ///< For Real.
    bool boolean{false};                    ///< For Boolean.
    std::string text;                       ///< For Text.
    std::vector<std::string> texts;         ///< For TextList.
    std::vector<std::int32_t> signedWords;  ///< For SignedWordList.
    std::string typeName;                   ///< The file's name for the type, as a refusal says it.
};

/**
 * @enum TensorType
 * @brief How a tensor's values are stored.
 */
enum class TensorType : std::uint8_t {
    Float32,           ///< F32: a 32-bit float per value.
    Float16,           ///< F16: a 16-bit float per value.
    BrainFloat16,      ///< BF16: the high half of a 32-bit float per value.
    QuantisedEightBit, ///< Q8_0: per 32 values, a 16-bit float step d, then 32 signed lanes; a value is lane * d.
    Other,             ///< Any other type, kept by its name only.
};

/**
 * @struct SourceTensor
 * @brief One tensor: its type, its extents and where its bytes lie.
 */
struct SourceTensor {
    TensorType type{TensorType::Other}; ///< How its values are stored.
    std::string typeName;               ///< The file's name for the type.
    std::vector<std::uint64_t> extents; ///< Fastest first: a matrix is {columns, rows}.
    const std::uint8_t *bytes{nullptr}; ///< Its values, in the file's mapping or a test's storage.
    std::uint64_t byteCount{0u};        ///< Bytes from @c bytes to the end of what it may read.
};

/**
 * @struct SourceModel
 * @brief Every key and every tensor of a model, by name.
 */
struct SourceModel {
    std::map<std::string, MetadataValue> metadata; ///< Every key.
    std::map<std::string, SourceTensor> tensors;   ///< Every tensor.
};

} // namespace lpl::convert

#endif /* LPL_LPL_CONVERT_SOURCE_HPP */
