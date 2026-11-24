/**
 * @file Agreement.hpp
 * @brief An image run by lpl::infer against its GGUF run by llama.cpp, position by position, on the same tokens.
 *
 * llama.cpp runs a sequence in one decode, every position's logits kept, its keys cached in F32 rather than its default
 * F16 (llama.cpp:18523), its values in F16, the only type b3775 caches them in without flash attention
 * (llama.cpp:18719-18720), and the residual stream's peak read after each layer's attention and feed-forward
 * (`ffn_inp` and `l_out`). lpl::infer runs the
 * image layer by layer through attentionBlock and feedForwardBlock, so that its residual stream's peak and the values
 * it saturates are counted, and its logits must equal Transformer::forward's, bit for bit. Two engines agree at a
 * position when their most likely tokens are the same; a position is judged when llama.cpp's own margin between its
 * first two tokens is at least kAgreementMargin, declared before any run.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CONVERT_AGREEMENT_HPP
#    define LPL_LPL_CONVERT_AGREEMENT_HPP

#    include <lpl/convert/Source.hpp>
#    include <lpl/infer/Model.hpp>
#    include <lpl/infer/TensorArena.hpp>

#    include <cstdint>
#    include <memory>
#    include <optional>
#    include <string>
#    include <vector>

namespace lpl::convert {

/** The margin, in logits, past which llama.cpp's most likely token is judged: a probability e times its runner-up's. */
inline constexpr double kAgreementMargin = 1.0;

/**
 * @struct PositionAgreement
 * @brief What the two engines give after one position.
 */
struct PositionAgreement {
    std::uint32_t imageToken{0u};       ///< lpl::infer's most likely token.
    std::uint32_t referenceToken{0u};   ///< llama.cpp's.
    double referenceMargin{0.0};        ///< llama.cpp's first logit less its second.
    double largestLogitDifference{0.0}; ///< The largest |difference| over the vocabulary.
};

/**
 * @struct SequenceAgreement
 * @brief What the two engines give over one sequence.
 */
struct SequenceAgreement {
    std::vector<PositionAgreement> positions; ///< One per token.
    double referenceResidualPeak{0.0};        ///< llama.cpp's largest |value| of the residual stream.
    double imageResidualPeak{0.0};            ///< lpl::infer's, after each block.
    std::uint64_t imageSaturations{0u};       ///< Residual values lpl::infer held at Q16.16's bounds.
    bool imageMatchesForward{false};          ///< Its logits are Transformer::forward's, bit for bit.

    /**
     * @brief Positions whose margin is at least kAgreementMargin.
     * @return The count.
     */
    [[nodiscard]] std::uint32_t judged() const noexcept;

    /**
     * @brief Judged positions whose most likely tokens differ.
     * @return The count.
     */
    [[nodiscard]] std::uint32_t judgedDisagreements() const noexcept;

    /**
     * @brief Positions, judged or not, whose most likely tokens are the same.
     * @return The count.
     */
    [[nodiscard]] std::uint32_t agreements() const noexcept;
};

/**
 * @class ReferenceModel
 * @brief A GGUF run by llama.cpp b3775 on the CPU.
 */
class ReferenceModel {
public:
    ReferenceModel();
    ReferenceModel(const ReferenceModel &) = delete;
    ReferenceModel &operator=(const ReferenceModel &) = delete;
    ~ReferenceModel();

    /**
     * @brief Loads the GGUF at @p path.
     * @param path    The file.
     * @param threads Threads llama.cpp computes on.
     * @return Nothing once loaded; `reference_unloadable` when llama.cpp refuses it.
     */
    [[nodiscard]] std::optional<Refusal> open(const std::string &path, std::uint32_t threads);

    /**
     * @brief Runs @p tokens from position 0 in one decode, in a context of their own.
     * @param tokens       The sequence.
     * @param logits       Receives one row of the vocabulary's logits per position.
     * @param residualPeak Receives the largest |value| of the residual stream after each block.
     * @return false for no token, or when llama.cpp refuses the context or the decode.
     */
    [[nodiscard]] bool run(const std::vector<std::uint32_t> &tokens, std::vector<float> &logits, double &residualPeak);

    /**
     * @brief How llama.cpp splits @p text into tokens, with no begin-of-text token added.
     * @param text         The text.
     * @param parseSpecial Whether the text of a CONTROL token is taken for the token.
     * @return The ids.
     */
    [[nodiscard]] std::vector<std::uint32_t> tokenize(const std::string &text, bool parseSpecial) const;

    /**
     * @brief Tokens of the vocabulary.
     * @return The count; 0 before open() succeeds.
     */
    [[nodiscard]] std::uint32_t vocabularySize() const noexcept;

    /**
     * @brief Threads llama.cpp computes on.
     * @return The count open() was given.
     */
    [[nodiscard]] std::uint32_t threads() const noexcept { return _threads; }

private:
    struct Loaded;
    std::unique_ptr<Loaded> _loaded;
    std::uint32_t _threads{0u};
};

/**
 * @brief Runs @p tokens through @p model from position 0, layer by layer.
 * @param model  A model ready to run, whose context is at least the sequence.
 * @param tokens The sequence.
 * @param out    Receives the agreement's image half: its residual peak, its saturations, and whether its logits are
 *               Transformer::forward's.
 * @param logits Receives one row of the vocabulary's logits per position.
 * @return false when the arena cannot hold the run or a token is past the vocabulary.
 */
[[nodiscard]] bool runImage(const infer::Model &model, const std::vector<std::uint32_t> &tokens, SequenceAgreement &out,
                            std::vector<float> &logits);

/**
 * @brief Runs @p tokens through both engines and compares them at each position.
 * @param reference The GGUF, in llama.cpp.
 * @param model     Its image, in lpl::infer.
 * @param tokens    The sequence.
 * @param out       Receives the comparison.
 * @return false when either engine cannot run the sequence.
 */
[[nodiscard]] bool compareSequence(ReferenceModel &reference, const infer::Model &model,
                                   const std::vector<std::uint32_t> &tokens, SequenceAgreement &out);

/**
 * @brief Whether an image agrees with llama.cpp over @p sequences: each ran, its logits Transformer::forward's, and
 *        no judged position of any disagrees, at least one judged in all.
 * @param sequences The comparisons of the image's sequences.
 * @return The verdict; false for no sequence.
 */
[[nodiscard]] bool agreesAtEveryJudgedPosition(const std::vector<SequenceAgreement> &sequences) noexcept;

/**
 * @brief Whether a control fails the agreement as a control must: it disagrees with llama.cpp at a judged position,
 *        and lies further from it than the image does.
 * @param control The comparisons of the control's sequences.
 * @param image   The comparisons of the image's.
 * @return The verdict.
 */
[[nodiscard]] bool controlFails(const std::vector<SequenceAgreement> &control,
                                const std::vector<SequenceAgreement> &image) noexcept;

/**
 * @enum Control
 * @brief A model made wrong on purpose from an image's own parts, which the agreement must refuse.
 */
enum class Control : std::uint8_t {
    KeyValueSwapped, ///< Every layer's key and value matrices swapped.
    RotaryBase,      ///< The rotary table of another base.
};

/**
 * @struct ControlModel
 * @brief A control and what it borrows.
 */
struct ControlModel {
    std::vector<infer::LayerWeights> layers;   ///< The image's layers, changed as the control says.
    std::vector<std::uint32_t> rotary;         ///< The rotary table it turns by.
    std::unique_ptr<infer::TensorArena> arena; ///< Its indexes.
    infer::Model model;                        ///< The control, borrowing these and the image.
};

/**
 * @brief Assembles @p control from @p image's own parts.
 * @param image   A model read from an image; it must outlive the control.
 * @param control What to change.
 * @param base    For RotaryBase, the base of the wrong table.
 * @param out     Receives the control.
 * @return false when the control cannot be assembled.
 */
[[nodiscard]] bool assembleControl(const infer::Model &image, Control control, double base, ControlModel &out);

} // namespace lpl::convert

#endif /* LPL_LPL_CONVERT_AGREEMENT_HPP */
