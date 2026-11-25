/**
 * @file synthetic_model.hpp
 * @brief A small `llama` model the converter's tests build in memory, change, and write as a GGUF file.
 *
 * Its shape is two layers of two query heads of 64 channels reading one key/value head, a feed-forward width of 256,
 * a context of 64 and a rotary base of 10000, so that its table is LLaMA's of the fixture. Its vocabulary is
 * SmolLM2's three turn tokens, CONTROL, then the 256 single bytes in GPT-2's alphabet, then five merged tokens, each
 * NORMAL; its template is ChatML. Each tensor draws its values from a stream of its own, so that no two tensors, two
 * layers or two rows hold the same values and a tensor in the wrong slot shows.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_TESTS_CONVERT_SYNTHETIC_MODEL_HPP
#    define LPL_TESTS_CONVERT_SYNTHETIC_MODEL_HPP

#    include <lpl/convert/Source.hpp>

#    include <cstdint>
#    include <map>
#    include <string>
#    include <vector>

namespace lpl::convert::tests {

inline constexpr std::uint32_t kSyntheticLayers = 2u;         ///< Blocks.
inline constexpr std::uint32_t kSyntheticDim = 128u;          ///< Width of the residual stream.
inline constexpr std::uint32_t kSyntheticHeads = 2u;          ///< Query heads, of 64 channels.
inline constexpr std::uint32_t kSyntheticKeyValueHeads = 1u;  ///< Key/value heads, each read by both query heads.
inline constexpr std::uint32_t kSyntheticHeadDim = 64u;       ///< Channels of a head.
inline constexpr std::uint32_t kSyntheticFeedForward = 256u;  ///< Width inside the position-wise network.
inline constexpr std::uint32_t kSyntheticContext = 64u;       ///< The GGUF's context length.
inline constexpr std::uint32_t kSyntheticRotaryBase = 10000u; ///< Its rope_theta.
inline constexpr std::uint32_t kSyntheticTurnTokens = 3u;     ///< `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`.
inline constexpr std::uint32_t kSyntheticByteTokens = 256u;   ///< One per byte, from id 3.
inline constexpr std::uint32_t kSyntheticMerges = 5u;         ///< Merged tokens, from id 259.
inline constexpr std::uint32_t kSyntheticVocabulary = kSyntheticTurnTokens + kSyntheticByteTokens + kSyntheticMerges;
inline constexpr double kSyntheticNormEpsilon = 1e-5; ///< Its RMS epsilon.

/**
 * @enum SyntheticWeights
 * @brief How the model's matrices are stored; its norms are F32, as llama.cpp's converter writes them.
 */
enum class SyntheticWeights : std::uint8_t {
    Float32,           ///< F32.
    Float16,           ///< F16.
    BrainFloat16,      ///< BF16.
    QuantisedEightBit, ///< Q8_0, quantised by ggml's own quantize_row_q8_0.
};

/**
 * @class SyntheticModel
 * @brief The model, its keys and tensors held here, and the values each tensor was drawn as.
 */
class SyntheticModel {
public:
    /**
     * @brief Builds the model with its matrices stored as @p weights.
     * @param weights How the matrices are stored.
     */
    explicit SyntheticModel(SyntheticWeights weights);
    SyntheticModel(const SyntheticModel &) = delete;
    SyntheticModel &operator=(const SyntheticModel &) = delete;

    /**
     * @brief The model, as a file would hold it.
     * @return Its keys and tensors.
     */
    [[nodiscard]] const SourceModel &source() const noexcept { return _source; }

    /**
     * @brief The value tensor @p name was drawn as at @p index, before its type stored it.
     * @param name  A tensor of the model.
     * @param index Fastest extent first.
     * @return The value.
     */
    [[nodiscard]] double drawn(const std::string &name, std::uint64_t index) const;

    /**
     * @brief The bytes token @p id stands for.
     * @param id Below kSyntheticVocabulary.
     * @return Its bytes, with GPT-2's mapping undone.
     */
    [[nodiscard]] const std::string &tokenBytes(std::uint32_t id) const;

    /**
     * @brief The merges, in the order of their ranks: their left, right and result ids.
     * @return kSyntheticMerges of them.
     */
    [[nodiscard]] const std::vector<std::uint32_t> &mergeIds() const noexcept { return _mergeIds; }

    /**
     * @brief Sets key @p key to a value of kind @p value.kind, replacing any.
     * @param key   The key.
     * @param value Its value.
     */
    void set(const std::string &key, const MetadataValue &value);

    /**
     * @brief Removes key @p key.
     * @param key The key.
     */
    void removeKey(const std::string &key);

    /**
     * @brief Adds or replaces tensor @p name, its @p values stored as @p type.
     * @param name    The tensor.
     * @param type    F32, F16, BF16 or Q8_0.
     * @param extents Fastest first.
     * @param values  One per element, row after row.
     */
    void putTensor(const std::string &name, TensorType type, const std::vector<std::uint64_t> &extents,
                   const std::vector<float> &values);

    /**
     * @brief Changes tensor @p name as a test needs, its bytes kept.
     * @param name The tensor.
     * @return The tensor.
     */
    [[nodiscard]] SourceTensor &tensor(const std::string &name);

    /**
     * @brief Removes tensor @p name.
     * @param name The tensor.
     */
    void removeTensor(const std::string &name);

    /**
     * @brief Writes the model as a GGUF file through ggml's gguf API.
     * @param path The file.
     */
    void write(const std::string &path) const;

private:
    void drawMatrix(const std::string &name, std::uint32_t rows, std::uint32_t columns, double amplitude,
                    TensorType type);
    void drawNorm(const std::string &name, std::uint32_t count);

    SourceModel _source{};
    std::map<std::string, std::vector<std::uint8_t>> _storage{};
    std::map<std::string, std::vector<float>> _drawn{};
    std::vector<std::string> _tokenBytes{};
    std::vector<std::uint32_t> _mergeIds{};
    std::uint32_t _stream{0u};
};

/**
 * @brief A key's value of kind UnsignedWord.
 * @param value The word.
 * @return The value.
 */
[[nodiscard]] MetadataValue unsignedWord(std::uint32_t value);

/**
 * @brief A key's value of kind Real.
 * @param value The float.
 * @return The value.
 */
[[nodiscard]] MetadataValue real(float value);

/**
 * @brief A key's value of kind Boolean.
 * @param value The boolean.
 * @return The value.
 */
[[nodiscard]] MetadataValue boolean(bool value);

/**
 * @brief A key's value of kind Text.
 * @param value The string.
 * @return The value.
 */
[[nodiscard]] MetadataValue text(const std::string &value);

/**
 * @brief A path for a test's file, in the system's temporary directory, unique to this process and @p name.
 * @param name What the file holds.
 * @return The path.
 */
[[nodiscard]] std::string temporaryPath(const std::string &name);

} // namespace lpl::convert::tests

#endif /* LPL_TESTS_CONVERT_SYNTHETIC_MODEL_HPP */
