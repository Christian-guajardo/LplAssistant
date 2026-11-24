/**
 * @file Converter.hpp
 * @brief A GGUF model turned into a `.lplmind` image of version 1, or refused by name.
 *
 * Every key of the architecture's and the tokenizer's namespaces, every tensor, every type and every token is checked
 * against what ring 0's transformer executes, and anything the converter does not know is refused: an image that
 * converts is one lpl::infer runs. `general.architecture` picks the architecture's row; the other `general.*` keys only
 * describe the file.
 *
 * The weights are requantised (Requantise.hpp), the query and key rows brought to ring 0's pairs (Pairing.hpp), the
 * rotary table computed (Rotary.hpp), the vocabulary decoded from GPT-2's alphabet (ByteLevel.hpp) and the merges
 * resolved to token ids. lpl::infer::Model::assemble then takes the parts, and lpl::infer::Model::writeBlob, the one
 * writer of the layout, writes them. The image is read back by lpl::infer::Model::view, its weights must fold as the
 * assembled model's, and each of its slots is compared with the source tensor that slot names: a relative root mean
 * square past kRequantisationBound refuses the conversion.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CONVERT_CONVERTER_HPP
#    define LPL_LPL_CONVERT_CONVERTER_HPP

#    include <lpl/convert/Requantise.hpp>
#    include <lpl/convert/Source.hpp>
#    include <lpl/infer/Model.hpp>
#    include <lpl/infer/ModelImage.hpp>

#    include <cstdint>
#    include <optional>
#    include <string>
#    include <vector>

namespace lpl::convert {

/**
 * The relative root mean square error past which a slot of the image is refused: SmolLM2's 8-bit blocks lose
 * 5.4e-3 to 7.0e-3, and a key and value matrix swapped about 1.4.
 */
inline constexpr double kRequantisationBound = 2e-2;

/**
 * @struct ConversionOptions
 * @brief What the caller chooses: how many positions the image's cache holds, and the experts' alignment.
 */
struct ConversionOptions {
    std::uint32_t contextLength{0u};                                     ///< Positions; at most the GGUF's.
    std::uint32_t alignmentLog2{infer::kModelImageDefaultAlignmentLog2}; ///< Each expert's alignment.
};

/**
 * @struct TensorReport
 * @brief One slot of the image, the source tensor it was made from, and how far it lies from it.
 */
struct TensorReport {
    std::string name;          ///< The source tensor's name.
    std::string type;          ///< The source tensor's type, as the file names it.
    std::uint64_t rows{0u};    ///< Rows; 1 for a vector.
    std::uint64_t columns{0u}; ///< Columns.
    ErrorStatistics error{};   ///< The image's values, read back, against the source's.
};

/**
 * @struct Conversion
 * @brief A conversion's image, or its refusal, and what it found.
 */
struct Conversion {
    std::optional<Refusal> refusal;                                   ///< Why it stopped; nothing when it did not.
    std::vector<std::uint8_t> image;                                  ///< The image; empty when refused.
    std::string architecture;                                         ///< `general.architecture`.
    infer::ModelConfig shape{};                                       ///< The image's shape.
    infer::PreTokenizer preTokenizer{infer::PreTokenizer::WholeText}; ///< How its text splits.
    infer::ChatFormat chatFormat{infer::ChatFormat::Plain};           ///< How it puts messages in turns.
    std::uint32_t mergeCount{0u};                                     ///< Its merges.
    std::uint32_t vocabularyTextBytes{0u};                            ///< Bytes of its vocabulary's text.
    std::vector<TensorReport> tensors;                                ///< Every slot, in the image's order.
};

/**
 * @brief Converts @p source into an image of version 1.
 *
 * @param source  The model, as its file holds it.
 * @param options The context and the alignment.
 * @return The image and its report, or the first refusal met, which names what it refuses: a key (`metadata_*`), a
 *         tensor (`tensor_*`, `bias_unsupported`, `output_untied`, `weight_*`, `q8_0_scale_refused`, `lane_past_range`,
 *         `requantisation_past_bound`), the tokenizer (`tokenizer_unknown`, `pre_tokenizer_unknown`, `token_*`,
 *         `merges_*`, `merge_unresolved`, `chat_template_unknown`, `end_of_turn_disagrees`, `chat_tokens_missing`), the
 *         rotation (`rotary_*`), the shape (`architecture_unknown`, `shape_refused`, `context_missing`,
 *         `context_past_training`) or the image (`image_too_large`, `image_unreadable`).
 */
[[nodiscard]] Conversion convertModel(const SourceModel &source, const ConversionOptions &options);

} // namespace lpl::convert

#endif /* LPL_LPL_CONVERT_CONVERTER_HPP */
