#include "synthetic_model.hpp"

#include <lpl/convert/Agreement.hpp>
#include <lpl/convert/Converter.hpp>
#include <lpl/convert/ModelFile.hpp>
#include <lpl/infer/TensorArena.hpp>
#include <lpl/infer/Tokenizer.hpp>
#include <lpl/testing/Test.hpp>

#include <cmath>
#include <filesystem>

LPL_TEST_SUITE(convert_agreement);

namespace {

constexpr std::uint32_t kThreads = 2u;             ///< llama.cpp's threads: two, beside the other tests.
constexpr std::size_t kViewArenaBytes = 1u << 20;  ///< Room for a view's layers and indexes.
constexpr std::uint32_t kLeastJudged = 40u;        ///< Positions the sequences must judge, so that agreeing proves.
constexpr std::uint32_t kPseudoRandomTokens = 48u; ///< The second sequence's length.
constexpr std::uint32_t kWrongBase = 100000u;      ///< A rotary base other than the model's 10000.

/**
 * @brief The sequences both engines run: a ChatML turn of the synthetic vocabulary, and a pseudo-random one as long as
 *        the context allows.
 */
[[nodiscard]] std::vector<std::vector<std::uint32_t>> sequences()
{
    const std::uint32_t byte = lpl::convert::tests::kSyntheticTurnTokens;
    std::vector<std::uint32_t> turn = {1u,   byte + 'u', byte + 's',  byte + 'e', byte + 'r', byte + '\n',
                                       259u, 263u,       261u,        byte + '!', 2u,         byte + '\n',
                                       1u,   byte + 'a', byte + '\n', 260u};
    std::vector<std::uint32_t> drawn;
    std::uint32_t state = 12345u;
    for (std::uint32_t i = 0u; i < kPseudoRandomTokens; ++i)
    {
        state = state * 1664525u + 1013904223u;
        drawn.push_back((state >> 8) % lpl::convert::tests::kSyntheticVocabulary);
    }
    return {turn, drawn};
}

/**
 * @struct Converted
 * @brief The synthetic model written as a GGUF, loaded by llama.cpp, and its image read by lpl::infer.
 */
struct Converted {
    std::string path;                               ///< The GGUF.
    lpl::convert::ReferenceModel reference;         ///< llama.cpp's.
    lpl::convert::Conversion conversion;            ///< The converter's.
    lpl::infer::TensorArena arena{kViewArenaBytes}; ///< The view's layers and indexes.
    lpl::infer::Model image;                        ///< The image, read where it lies.
};

/**
 * @brief Writes the synthetic model of @p weights, loads it in llama.cpp and converts it, its context the GGUF's.
 * @return Whether every step held.
 */
[[nodiscard]] bool convertedModel(lpl::convert::tests::SyntheticWeights weights, const std::string &name,
                                  Converted &out)
{
    const lpl::convert::tests::SyntheticModel model(weights);
    out.path = lpl::convert::tests::temporaryPath(name);
    model.write(out.path);
    if (out.reference.open(out.path, kThreads).has_value())
        return false;

    lpl::convert::ModelFile file;
    lpl::convert::ConversionOptions options;
    options.contextLength = lpl::convert::tests::kSyntheticContext;
    if (file.open(out.path).has_value())
        return false;
    out.conversion = lpl::convert::convertModel(file.source(), options);
    return !out.conversion.refusal.has_value() &&
           out.image
                   .view(out.arena, out.conversion.image.data(),
                         static_cast<lpl::core::u32>(out.conversion.image.size()))
                   .verdict == lpl::infer::ModelReadVerdict::Read;
}

/**
 * @struct Judgement
 * @brief What comparing a model with llama.cpp over the sequences found.
 */
struct Judgement {
    bool ran{true};                        ///< Every sequence ran in both engines.
    bool everyForwardMatched{true};        ///< The layer-by-layer logits were Transformer::forward's.
    std::uint32_t judged{0u};              ///< Positions judged.
    std::uint32_t judgedDisagreements{0u}; ///< Judged positions that disagree.
    std::uint32_t positions{0u};           ///< Positions run.
    std::uint32_t agreements{0u};          ///< Positions, judged or not, that agree.
    double largestLogitDifference{0.0};    ///< Over every position.
};

/**
 * @brief Compares @p model with @p reference over the sequences.
 */
[[nodiscard]] Judgement judge(lpl::convert::ReferenceModel &reference, const lpl::infer::Model &model)
{
    Judgement judgement;
    for (const std::vector<std::uint32_t> &tokens : sequences())
    {
        lpl::convert::SequenceAgreement agreement;
        judgement.ran = judgement.ran && lpl::convert::compareSequence(reference, model, tokens, agreement);
        judgement.everyForwardMatched = judgement.everyForwardMatched && agreement.imageMatchesForward;
        judgement.judged += agreement.judged();
        judgement.judgedDisagreements += agreement.judgedDisagreements();
        judgement.agreements += agreement.agreements();
        judgement.positions += static_cast<std::uint32_t>(agreement.positions.size());
        for (const lpl::convert::PositionAgreement &position : agreement.positions)
            judgement.largestLogitDifference =
                std::fmax(judgement.largestLogitDifference, position.largestLogitDifference);
    }
    return judgement;
}

/**
 * @brief Checks that the image of the synthetic model of @p weights agrees with llama.cpp on its GGUF.
 */
void checkAgreement(lpl::testing::Test &test, lpl::convert::tests::SyntheticWeights weights, const std::string &name)
{
    Converted converted;
    test.check(convertedModel(weights, name, converted), "the GGUF loads in llama.cpp and converts to an image");
    if (converted.image.ready())
    {
        const Judgement judgement = judge(converted.reference, converted.image);
        test.check(judgement.ran && judgement.positions == 16u + kPseudoRandomTokens,
                   "both engines run both sequences, 64 positions");
        test.check(judgement.everyForwardMatched, "the image's logits, layer by layer, are Transformer::forward's");
        test.check(judgement.judged >= kLeastJudged, "at least 40 positions have a margin of a logit to judge");
        test.check(judgement.judgedDisagreements == 0u, "and at each, both engines pick the same token");
        test.measure("judged", judgement.judged);
        test.measure("agreements", judgement.agreements);
        test.measure("positions", judgement.positions);
        test.measure("largest_logit_difference_thousandths",
                     static_cast<lpl::core::u32>(judgement.largestLogitDifference * 1000.0));
    }
    std::filesystem::remove(converted.path);
}

} // namespace

/**
 * @brief The image of a synthetic model of F32 matrices agrees with llama.cpp on its GGUF at every judged position.
 */
LPL_TEST(an_f32_model_s_image_agrees_with_llama_cpp_at_every_judged_position)
{
    checkAgreement(test, lpl::convert::tests::SyntheticWeights::Float32, "agreement-f32.gguf");
}

/**
 * @brief And so does one of Q8_0 matrices, whose lanes both engines read.
 */
LPL_TEST(a_q8_0_model_s_image_agrees_with_llama_cpp_at_every_judged_position)
{
    checkAgreement(test, lpl::convert::tests::SyntheticWeights::QuantisedEightBit, "agreement-q8-0.gguf");
}

/**
 * @brief The image's tokenizer splits text as llama.cpp splits it: the vocabulary, the merges and the pre-tokenizer
 *        made it through.
 */
LPL_TEST(the_image_s_tokenizer_splits_text_as_llama_cpp_does)
{
    Converted converted;
    test.check(convertedModel(lpl::convert::tests::SyntheticWeights::Float32, "tokens.gguf", converted),
               "the GGUF loads and converts");
    if (converted.image.ready())
    {
        lpl::infer::TensorArena arena{kViewArenaBytes};
        lpl::infer::Tokenizer tokenizer;
        test.check(
            tokenizer.bind(arena, converted.image.vocab(), converted.image.merges(), converted.image.preTokenizer()),
            "the image's tokenizer binds");
        const std::string text = "hello the in, then 2024 inner\nthe end";
        lpl::core::u32 ids[64];
        lpl::core::u32 skipped = 0u;
        const lpl::core::u32 count =
            tokenizer.encode(text.data(), static_cast<lpl::core::u32>(text.size()), ids, 64u, skipped);
        const std::vector<std::uint32_t> ours(ids, ids + count);
        const std::vector<std::uint32_t> theirs = converted.reference.tokenize(text, false);
        test.check(skipped == 0u && !ours.empty() && ours == theirs, "the same ids, every byte spelled");
        test.check(ours.size() < text.size(), "and some merged, so that the merges were applied");
        test.measure("tokens", count);
    }
    std::filesystem::remove(converted.path);
}

/**
 * @brief Two models made wrong on purpose from the image's own parts, its keys and values swapped and its rotary table
 *        of another base, each disagree with llama.cpp at a judged position: the agreement can fail.
 */
LPL_TEST(a_control_made_wrong_on_purpose_disagrees_with_llama_cpp)
{
    Converted converted;
    test.check(convertedModel(lpl::convert::tests::SyntheticWeights::Float32, "controls.gguf", converted),
               "the GGUF loads and converts");
    if (converted.image.ready())
    {
        lpl::convert::ControlModel swapped;
        lpl::convert::ControlModel rotated;
        test.check(
            lpl::convert::assembleControl(converted.image, lpl::convert::Control::KeyValueSwapped, 0.0, swapped) &&
                lpl::convert::assembleControl(converted.image, lpl::convert::Control::RotaryBase, kWrongBase, rotated),
            "both controls assemble from the image's parts");
        const Judgement image = judge(converted.reference, converted.image);
        const Judgement keyValue = judge(converted.reference, swapped.model);
        const Judgement rotary = judge(converted.reference, rotated.model);
        test.check(keyValue.largestLogitDifference > image.largestLogitDifference &&
                       rotary.largestLogitDifference > image.largestLogitDifference,
                   "each lies further from llama.cpp than the image itself does");
        test.check(keyValue.ran && keyValue.judgedDisagreements > 0u,
                   "the keys and values swapped disagree at a judged position");
        test.check(rotary.ran && rotary.judgedDisagreements > 0u, "and so does the rotary table of base 100000");
        test.measure("key_value_swapped_disagreements", keyValue.judgedDisagreements);
        test.measure("rotary_base_disagreements", rotary.judgedDisagreements);
        test.measure("key_value_swapped_largest_difference_thousandths",
                     static_cast<lpl::core::u32>(keyValue.largestLogitDifference * 1000.0));
        test.measure("rotary_base_largest_difference_thousandths",
                     static_cast<lpl::core::u32>(rotary.largestLogitDifference * 1000.0));
    }
    std::filesystem::remove(converted.path);
}
