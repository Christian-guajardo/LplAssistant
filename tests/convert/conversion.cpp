#include "fixtures/gguf_reference.hpp"
#include "synthetic_model.hpp"

#include <lpl/convert/Converter.hpp>
#include <lpl/infer/ModelImage.hpp>
#include <lpl/infer/TensorArena.hpp>
#include <lpl/testing/Test.hpp>

#include <cmath>
#include <cstring>
#include <limits>

LPL_TEST_SUITE(convert_conversion);

namespace {

constexpr std::uint32_t kContext = 32u;             ///< The context a test asks for: below the GGUF's 64.
constexpr std::size_t kViewArenaBytes = 1u << 20;   ///< Room for a view's layers and indexes.
constexpr double kSlotBound = 2e-2;                 ///< A slot lies this close to its own source.
constexpr double kOtherSlot = 0.5;                  ///< And this far from another tensor of its shape.
constexpr std::uint32_t kNormEpsilonUnits = 42950u; ///< round(float(1e-5) * 2^32), worked apart.

/**
 * @brief The options of a test: kContext positions, experts on 4 KiB.
 */
[[nodiscard]] lpl::convert::ConversionOptions options()
{
    lpl::convert::ConversionOptions made;

    made.contextLength = kContext;
    return made;
}

/**
 * @brief The name of what converting @p model with @p chosen gives: its refusal, or "converted".
 */
[[nodiscard]] std::string outcomeOf(const lpl::convert::tests::SyntheticModel &model,
                                    const lpl::convert::ConversionOptions &chosen = options())
{
    const lpl::convert::Conversion conversion = lpl::convert::convertModel(model.source(), chosen);

    return conversion.refusal.has_value() ? conversion.refusal->name : "converted";
}

/**
 * @brief Whether converting @p model is refused as @p name, its detail holding @p named.
 */
[[nodiscard]] bool refusedAs(const lpl::convert::tests::SyntheticModel &model, const std::string &name,
                             const std::string &named, const lpl::convert::ConversionOptions &chosen = options())
{
    const lpl::convert::Conversion conversion = lpl::convert::convertModel(model.source(), chosen);

    return conversion.refusal.has_value() && conversion.refusal->name == name &&
           conversion.refusal->detail.find(named) != std::string::npos && conversion.image.empty();
}

/**
 * @brief A key's value of kind TextList.
 */
[[nodiscard]] lpl::convert::MetadataValue textList(const std::vector<std::string> &values)
{
    lpl::convert::MetadataValue made;

    made.kind = lpl::convert::MetadataKind::TextList;
    made.typeName = "arr[str]";
    made.texts = values;
    return made;
}

/**
 * @brief The tokens of @p model, as its file writes them.
 */
[[nodiscard]] std::vector<std::string> tokensOf(const lpl::convert::tests::SyntheticModel &model)
{
    return model.source().metadata.at("tokenizer.ggml.tokens").texts;
}

/**
 * @brief The merges of @p model, as its file writes them.
 */
[[nodiscard]] std::vector<std::string> mergesOf(const lpl::convert::tests::SyntheticModel &model)
{
    return model.source().metadata.at("tokenizer.ggml.merges").texts;
}

/**
 * @brief The relative root mean square of the difference between @p matrix, each lane standing for lane M / 127, and
 *        the values tensor @p name of @p model was drawn as, its row r compared with the source's row r + @p shift.
 */
[[nodiscard]] double distanceTo(const lpl::infer::QuantMatrixView &matrix,
                                const lpl::convert::tests::SyntheticModel &model, const std::string &name,
                                std::uint32_t shift = 0u)
{
    double squaredError = 0.0;
    double squaredSource = 0.0;

    for (std::uint32_t row = 0u; row < matrix.rows; ++row)
    {
        const lpl::infer::QuantBlock *const blocks = matrix.row(row);
        for (std::uint32_t column = 0u; column < matrix.columns; ++column)
        {
            const lpl::infer::QuantBlock &block = blocks[column / lpl::infer::kQuantBlockLanes];
            const double value = static_cast<double>(block.lanes[column % lpl::infer::kQuantBlockLanes]) *
                                 static_cast<double>(block.scaleRaw) / 127.0 / 65536.0;
            const std::uint64_t sourceRow = (row + shift) % matrix.rows;
            const double source = model.drawn(name, sourceRow * matrix.columns + column);
            squaredError += (value - source) * (value - source);
            squaredSource += source * source;
        }
    }
    return std::sqrt(squaredError / squaredSource);
}

/**
 * @brief Whether @p matrix lies close to tensor @p own of @p model and far from tensor @p other.
 */
[[nodiscard]] bool holds(const lpl::infer::QuantMatrixView &matrix, const lpl::convert::tests::SyntheticModel &model,
                         const std::string &own, const std::string &other)
{
    return distanceTo(matrix, model, own) < kSlotBound && distanceTo(matrix, model, other) > kOtherSlot;
}

/**
 * @brief Whether every gain of @p norm is the value tensor @p name was drawn as, rounded half away from zero to Q16.16.
 */
[[nodiscard]] bool normHolds(lpl::infer::ConstVectorView norm, const lpl::convert::tests::SyntheticModel &model,
                             const std::string &name)
{
    for (std::uint32_t i = 0u; i < norm.count; ++i)
    {
        const double scaled = model.drawn(name, i) * 65536.0;
        const double rounded = scaled < 0.0 ? -std::floor(-scaled + 0.5) : std::floor(scaled + 0.5);
        if (static_cast<double>(norm.at(i).raw()) != rounded)
            return false;
    }
    return norm.count != 0u;
}

/**
 * @brief Checks that @p image of @p model reads back with every slot, header field, token and merge as the source
 *        has them.
 */
void checkImage(lpl::testing::Test &test, const lpl::convert::tests::SyntheticModel &model,
                const lpl::convert::Conversion &conversion)
{
    lpl::infer::TensorArena arena{kViewArenaBytes};
    lpl::infer::Model view;
    const std::vector<std::uint8_t> &image = conversion.image;
    const lpl::infer::ModelReadReport read = view.view(arena, image.data(), static_cast<lpl::core::u32>(image.size()));
    test.check(read.verdict == lpl::infer::ModelReadVerdict::Read, "lpl::infer reads the image where it lies");
    if (read.verdict != lpl::infer::ModelReadVerdict::Read)
        return;

    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image.data(), image.size(), nullptr, 0u);
    const lpl::infer::ModelConfig &shape = view.config();
    test.check(shape.dim == 128u && shape.layers == 2u && shape.heads == 2u && shape.keyValueHeads == 1u &&
                   shape.ffnHidden == 256u && shape.vocabSize == 264u,
               "its shape is the GGUF's");
    test.check(shape.contextLength == kContext && shape.normEpsilon == kNormEpsilonUnits,
               "its context is the one asked for, and its epsilon round(1e-5 * 2^32)");
    test.check(inspection.header.preTokenizer == lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel &&
                   inspection.header.chatFormat == lpl::infer::ChatFormat::ChatMl &&
                   inspection.header.mergeCount == lpl::convert::tests::kSyntheticMerges,
               "its header says smollm's pre-tokenizer, ChatML and five merges");

    bool everySlot = distanceTo(view.embedding(), model, "token_embd.weight") < kSlotBound &&
                     distanceTo(view.embedding(), model, "token_embd.weight", 1u) > kOtherSlot &&
                     normHolds(view.finalNorm(), model, "output_norm.weight");
    for (std::uint32_t l = 0u; l < shape.layers; ++l)
    {
        const std::string block = "blk." + std::to_string(l) + ".";
        const std::string otherBlock = "blk." + std::to_string(1u - l) + ".";
        const lpl::infer::LayerWeights &layer = view.layer(l);
        everySlot = everySlot && holds(layer.query, model, block + "attn_q.weight", otherBlock + "attn_q.weight") &&
                    holds(layer.key, model, block + "attn_k.weight", block + "attn_v.weight") &&
                    holds(layer.value, model, block + "attn_v.weight", block + "attn_k.weight") &&
                    holds(layer.output, model, block + "attn_output.weight", otherBlock + "attn_output.weight") &&
                    holds(layer.gate, model, block + "ffn_gate.weight", block + "ffn_up.weight") &&
                    holds(layer.up, model, block + "ffn_up.weight", block + "ffn_gate.weight") &&
                    holds(layer.down, model, block + "ffn_down.weight", otherBlock + "ffn_down.weight") &&
                    normHolds(layer.attentionNorm, model, block + "attn_norm.weight") &&
                    normHolds(layer.feedForwardNorm, model, block + "ffn_norm.weight");
    }
    test.check(everySlot, "every matrix lies within 2e-2 of its own tensor and past 0.5 of the other of its shape, "
                          "every norm is its gains rounded to Q16.16");

    const lpl::infer::RotaryFrequencies rotary = view.rotaryFrequencies();
    bool rotaryHolds = rotary.pairs == gguf_reference::kRotaryPairs;
    for (std::uint32_t pair = 0u; rotaryHolds && pair < rotary.pairs; ++pair)
        rotaryHolds = rotary.turnsPerPosition[pair] == gguf_reference::kLlamaTurns[pair];
    test.check(rotaryHolds, "its rotary table is LLaMA's base-10000 table, worked apart");

    bool everyToken = view.vocab().size() == lpl::convert::tests::kSyntheticVocabulary;
    for (std::uint32_t id = 0u; everyToken && id < view.vocab().size(); ++id)
    {
        lpl::core::u32 length = 0u;
        const char *const bytes = view.vocab().text(id, length);
        everyToken = bytes != nullptr && std::string(bytes, length) == model.tokenBytes(id);
    }
    test.check(everyToken, "every token is the bytes it stands for, GPT-2's mapping undone, the turn tokens as text");

    const std::vector<std::uint32_t> &merges = model.mergeIds();
    bool everyMerge = view.merges().size() == lpl::convert::tests::kSyntheticMerges;
    for (std::uint32_t rank = 0u; everyMerge && rank < view.merges().size(); ++rank)
    {
        const lpl::infer::BytePairMerge &merge = view.merges().merges()[rank];
        everyMerge = merge.left == merges[3u * rank] && merge.right == merges[3u * rank + 1u] &&
                     merge.result == merges[3u * rank + 2u] && view.merges().rank(merge.left, merge.right) == rank;
    }
    test.check(everyMerge, "every merge joins its two tokens into their concatenation, at its own rank");
    test.check(view.chatTokens().turnStart == 1u && view.chatTokens().turnEnd == 2u &&
                   view.chatTokens().endOfText == 0u,
               "and the turns open on 1 and end on 2 or 0");
}

/**
 * @brief Whether every slot of @p conversion's report lies under the bound, a matrix's error above zero.
 */
[[nodiscard]] bool reportHolds(const lpl::convert::Conversion &conversion)
{
    std::uint32_t matrices = 0u;

    for (const lpl::convert::TensorReport &tensor : conversion.tensors)
    {
        const double error = tensor.error.relativeRootMeanSquare();
        if (!(error < kSlotBound) || tensor.error.values != tensor.rows * tensor.columns)
            return false;
        if (tensor.rows > 1u && !(error > 0.0))
            return false;
        matrices += tensor.rows > 1u ? 1u : 0u;
    }
    return conversion.tensors.size() == 1u + 9u * lpl::convert::tests::kSyntheticLayers + 1u &&
           matrices == 1u + 7u * lpl::convert::tests::kSyntheticLayers;
}

/**
 * @brief Converts the synthetic model with its matrices stored as @p weights and checks its image.
 */
void checkConversion(lpl::testing::Test &test, lpl::convert::tests::SyntheticWeights weights)
{
    const lpl::convert::tests::SyntheticModel model(weights);
    const lpl::convert::Conversion conversion = lpl::convert::convertModel(model.source(), options());

    test.check(!conversion.refusal.has_value() && !conversion.image.empty(), "the model converts");
    if (conversion.refusal.has_value() || conversion.image.empty())
        return;
    checkImage(test, model, conversion);
    test.check(reportHolds(conversion), "the report names all 20 slots, each under the bound, each matrix's error "
                                        "above zero and counted over all its values");
    const lpl::convert::Conversion again = lpl::convert::convertModel(model.source(), options());
    test.check(again.image == conversion.image, "and a second conversion writes the same bytes");
    test.measure("image_bytes", static_cast<lpl::core::u32>(conversion.image.size()));
}

} // namespace

/**
 * @brief A model of F32 matrices converts to an image whose every slot, header field, token and merge reads back as
 *        the source has them.
 */
LPL_TEST(a_model_of_f32_matrices_converts_slot_for_slot)
{
    checkConversion(test, lpl::convert::tests::SyntheticWeights::Float32);
}

/**
 * @brief And so does one of F16 matrices.
 */
LPL_TEST(a_model_of_f16_matrices_converts_slot_for_slot)
{
    checkConversion(test, lpl::convert::tests::SyntheticWeights::Float16);
}

/**
 * @brief And one of BF16 matrices.
 */
LPL_TEST(a_model_of_bf16_matrices_converts_slot_for_slot)
{
    checkConversion(test, lpl::convert::tests::SyntheticWeights::BrainFloat16);
}

/**
 * @brief And one of Q8_0 matrices, whose lanes the image keeps.
 */
LPL_TEST(a_model_of_q8_0_matrices_converts_slot_for_slot_its_lanes_kept)
{
    checkConversion(test, lpl::convert::tests::SyntheticWeights::QuantisedEightBit);
    const lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::QuantisedEightBit);
    const lpl::convert::Conversion conversion = lpl::convert::convertModel(model.source(), options());
    if (conversion.image.empty())
        return;

    lpl::infer::TensorArena arena{kViewArenaBytes};
    lpl::infer::Model view;
    test.check(
        view.view(arena, conversion.image.data(), static_cast<lpl::core::u32>(conversion.image.size())).verdict ==
            lpl::infer::ModelReadVerdict::Read,
        "the image reads");
    const lpl::convert::SourceTensor &source = model.source().tensors.at("blk.1.attn_v.weight");
    const lpl::infer::QuantMatrixView &value = view.layer(1u).value;
    bool everyLane = value.rows * value.blocksPerRow() * lpl::convert::kEightBitBlockBytes == source.byteCount;
    for (std::uint32_t block = 0u; everyLane && block < value.rows * value.blocksPerRow(); ++block)
    {
        const std::uint8_t *const at =
            source.bytes + static_cast<std::size_t>(block) * lpl::convert::kEightBitBlockBytes;
        everyLane = std::memcmp(value.blocks[block].lanes, at + 2u, lpl::infer::kQuantBlockLanes) == 0;
    }
    test.check(everyLane, "every lane of the second layer's value matrix is the GGUF's own, byte for byte");
}

/**
 * @brief A key the converter does not know, one it needs and lacks, one of the wrong type or out of its range, a
 *        context it cannot give and an alignment the image cannot take are each refused by name.
 */
LPL_TEST(a_key_that_does_not_hold_is_refused_by_name)
{
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("general.architecture", lpl::convert::tests::text("gpt2"));
        test.check(refusedAs(model, "architecture_unknown", "gpt2"),
                   "an architecture of no row is architecture_unknown");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.removeKey("general.architecture");
        test.check(refusedAs(model, "metadata_missing", "general.architecture"),
                   "no architecture is metadata_missing, naming the key");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.expert_count", lpl::convert::tests::unsignedWord(8u));
        test.check(refusedAs(model, "metadata_unknown", "llama.expert_count"),
                   "a key of the architecture it does not know is metadata_unknown, naming it");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.add_eos_token", lpl::convert::tests::boolean(false));
        test.check(refusedAs(model, "metadata_unknown", "tokenizer.ggml.add_eos_token"),
                   "and so is one of the tokenizer");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("vision.patch_size", lpl::convert::tests::unsignedWord(14u));
        model.set("general.description", lpl::convert::tests::text("described, not read"));
        test.check(refusedAs(model, "metadata_unknown", "vision.patch_size"),
                   "and one of another namespace, while general.* only describes the file");
    }
    const char *const required[] = {"llama.block_count",           "llama.context_length",
                                    "llama.embedding_length",      "llama.feed_forward_length",
                                    "llama.attention.head_count",  "llama.attention.head_count_kv",
                                    "llama.rope.freq_base",        "llama.attention.layer_norm_rms_epsilon",
                                    "llama.rope.dimension_count",  "tokenizer.ggml.model",
                                    "tokenizer.ggml.pre",          "tokenizer.ggml.tokens",
                                    "tokenizer.ggml.token_type",   "tokenizer.ggml.eos_token_id",
                                    "tokenizer.ggml.add_bos_token"};
    std::uint32_t missingNamed = 0u;
    for (const char *key : required)
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.removeKey(key);
        missingNamed += refusedAs(model, "metadata_missing", key) ? 1u : 0u;
    }
    test.check(missingNamed == sizeof(required) / sizeof(required[0]),
               "each of the 15 keys ring 0 needs, missing, is metadata_missing naming it");
    test.measure("required_keys_refused", missingNamed);
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.removeKey("llama.vocab_size");
        model.removeKey("tokenizer.ggml.add_space_prefix");
        model.removeKey("tokenizer.ggml.bos_token_id");
        model.removeKey("tokenizer.ggml.unknown_token_id");
        model.removeKey("tokenizer.ggml.padding_token_id");
        test.check(outcomeOf(model) == "converted", "while the keys it can do without may be missing");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.block_count", lpl::convert::tests::real(2.0f));
        test.check(refusedAs(model, "metadata_type_wrong", "llama.block_count"),
                   "a count stored as a float is metadata_type_wrong, naming the key");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.tokens", lpl::convert::tests::text("one"));
        test.check(refusedAs(model, "metadata_type_wrong", "tokenizer.ggml.tokens"),
                   "and a token list stored as a string");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.attention.layer_norm_rms_epsilon", lpl::convert::tests::real(-1e-5f));
        test.check(refusedAs(model, "metadata_out_of_range", "layer_norm_rms_epsilon"),
                   "a negative epsilon is metadata_out_of_range");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.attention.layer_norm_rms_epsilon", lpl::convert::tests::real(1.0f));
        test.check(refusedAs(model, "metadata_out_of_range", "layer_norm_rms_epsilon"),
                   "and so is one of 1, which 32 bits of 2^-32 cannot hold");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.rope.freq_base", lpl::convert::tests::real(1.0f));
        test.check(refusedAs(model, "metadata_out_of_range", "freq_base"), "and a rotary base of 1");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.rope.dimension_count", lpl::convert::tests::unsignedWord(32u));
        test.check(refusedAs(model, "rotary_partial", "32"), "a rotation of half a head is rotary_partial");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.rope.scaling.type", lpl::convert::tests::text("linear"));
        test.check(refusedAs(model, "rotary_scaled", "llama.rope.scaling.type"), "a scaled rotation is rotary_scaled");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.attention.head_count", lpl::convert::tests::unsignedWord(3u));
        test.check(refusedAs(model, "shape_refused", "head_count"),
                   "heads that do not divide the width are shape_refused");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.attention.head_count_kv", lpl::convert::tests::unsignedWord(0u));
        test.check(refusedAs(model, "shape_refused", "head_count_kv"), "and no key/value head");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.embedding_length", lpl::convert::tests::unsignedWord(131072u));
        model.set("llama.attention.head_count", lpl::convert::tests::unsignedWord(2048u));
        model.set("llama.attention.head_count_kv", lpl::convert::tests::unsignedWord(2048u));
        test.check(refusedAs(model, "image_too_large", "bytes"),
                   "a shape whose image passes 32 bits is image_too_large, before any tensor is read");
    }
    {
        const lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        lpl::convert::ConversionOptions none = options();
        none.contextLength = 0u;
        lpl::convert::ConversionOptions past = options();
        past.contextLength = lpl::convert::tests::kSyntheticContext + 1u;
        lpl::convert::ConversionOptions trained = options();
        trained.contextLength = lpl::convert::tests::kSyntheticContext;
        test.check(refusedAs(model, "context_missing", "context", none), "no context is context_missing");
        test.check(refusedAs(model, "context_past_training", "65", past),
                   "a context past the GGUF's is context_past_training");
        test.check(outcomeOf(model, trained) == "converted", "while the GGUF's own converts");
        lpl::convert::ConversionOptions small = options();
        small.alignmentLog2 = 11u;
        lpl::convert::ConversionOptions large = options();
        large.alignmentLog2 = 23u;
        lpl::convert::ConversionOptions page = options();
        page.alignmentLog2 = 22u;
        test.check(refusedAs(model, "alignment_refused", "11", small) &&
                       refusedAs(model, "alignment_refused", "23", large),
                   "an alignment below 4 KiB or past 4 MiB is alignment_refused");
        test.check(outcomeOf(model, page) == "converted", "while 4 MiB converts");
    }
}

/**
 * @brief A tensor the converter does not know, one it needs and lacks, one of a type or a shape ring 0 does not take,
 *        and a value it cannot hold are each refused by name.
 */
LPL_TEST(a_tensor_that_does_not_hold_is_refused_by_name)
{
    const std::vector<float> norm(lpl::convert::tests::kSyntheticDim, 1.0f);
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.putTensor("blk.0.attn_extra.weight", lpl::convert::TensorType::Float32, {128u}, norm);
        test.check(refusedAs(model, "tensor_unknown", "blk.0.attn_extra.weight"),
                   "an unknown tensor is tensor_unknown");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.putTensor("blk.2.attn_norm.weight", lpl::convert::TensorType::Float32, {128u}, norm);
        test.check(refusedAs(model, "tensor_unknown", "blk.2.attn_norm.weight"), "and so is one of a third layer");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.removeTensor("blk.1.ffn_down.weight");
        test.check(refusedAs(model, "tensor_missing", "blk.1.ffn_down.weight"),
                   "a missing tensor is tensor_missing, naming it");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.putTensor("blk.0.attn_q.bias", lpl::convert::TensorType::Float32, {128u}, norm);
        test.check(refusedAs(model, "bias_unsupported", "blk.0.attn_q.bias"), "a bias is bias_unsupported");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.putTensor("output.weight", lpl::convert::TensorType::Float32, {128u, 264u},
                        std::vector<float>(128u * 264u, 0.5f));
        test.check(refusedAs(model, "output_untied", "output.weight"),
                   "an output head of its own is output_untied: the image ties it to the embedding");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.putTensor("rope_freqs.weight", lpl::convert::TensorType::Float32, {32u}, std::vector<float>(32u, 1.0f));
        test.check(refusedAs(model, "rotary_scaled", "rope_freqs.weight"), "rotary factors are rotary_scaled");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.tensor("blk.0.attn_q.weight").type = lpl::convert::TensorType::Other;
        model.tensor("blk.0.attn_q.weight").typeName = "q4_K";
        test.check(refusedAs(model, "tensor_type_unsupported", "q4_K"), "a matrix of q4_K is tensor_type_unsupported");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.putTensor("blk.0.ffn_norm.weight", lpl::convert::TensorType::QuantisedEightBit, {128u}, norm);
        test.check(refusedAs(model, "tensor_type_unsupported", "blk.0.ffn_norm.weight"), "and so is a norm of Q8_0");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.tensor("blk.1.attn_k.weight").extents = {128u, 63u};
        test.check(refusedAs(model, "tensor_shape_wrong", "blk.1.attn_k.weight"),
                   "a key matrix a row short is tensor_shape_wrong");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.tensor("output_norm.weight").extents = {128u, 1u, 1u};
        test.check(refusedAs(model, "tensor_shape_wrong", "output_norm.weight"), "and so is a norm of three extents");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.tensor("token_embd.weight").byteCount -= 1u;
        test.check(refusedAs(model, "tensor_outside_file", "token_embd.weight"),
                   "a tensor whose bytes end past what the file holds is tensor_outside_file");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<float> values(128u * 128u, 0.01f);
        values[5000] = std::numeric_limits<float>::quiet_NaN();
        model.putTensor("blk.1.attn_output.weight", lpl::convert::TensorType::Float32, {128u, 128u}, values);
        test.check(refusedAs(model, "weight_not_finite", "blk.1.attn_output.weight"),
                   "a NaN weight is weight_not_finite, naming its tensor");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<float> gains = norm;
        gains[3] = std::numeric_limits<float>::infinity();
        model.putTensor("blk.0.attn_norm.weight", lpl::convert::TensorType::Float32, {128u}, gains);
        test.check(refusedAs(model, "weight_not_finite", "blk.0.attn_norm.weight"), "and so is an infinite gain");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<float> values(256u * 128u, 0.01f);
        values[7] = 40000.0f;
        model.putTensor("blk.0.ffn_up.weight", lpl::convert::TensorType::Float32, {128u, 256u}, values);
        test.check(refusedAs(model, "weight_past_fixed32", "blk.0.ffn_up.weight"),
                   "a weight past 32767.99998 is weight_past_fixed32");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::QuantisedEightBit);
        lpl::convert::SourceTensor &key = model.tensor("blk.0.attn_k.weight");
        std::uint8_t *const bytes = const_cast<std::uint8_t *>(key.bytes);
        bytes[lpl::convert::kEightBitBlockBytes + 1u] |= 0x80u;
        test.check(refusedAs(model, "q8_0_scale_refused", "blk.0.attn_k.weight"),
                   "a Q8_0 block of negative step is q8_0_scale_refused");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::QuantisedEightBit);
        lpl::convert::SourceTensor &gate = model.tensor("blk.1.ffn_gate.weight");
        std::uint8_t *const bytes = const_cast<std::uint8_t *>(gate.bytes);
        bytes[2u + 9u] = 0x80u;
        test.check(refusedAs(model, "lane_past_range", "blk.1.ffn_gate.weight"),
                   "a Q8_0 lane of -128 is lane_past_range");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.putTensor("blk.1.ffn_gate.weight", lpl::convert::TensorType::Float32, {128u, 256u},
                        std::vector<float>(128u * 256u, 3e-6f));
        test.check(refusedAs(model, "requantisation_past_bound", "blk.1.ffn_gate.weight"),
                   "a matrix of values Q16.16 cannot tell apart from zero is requantisation_past_bound");
    }
}

/**
 * @brief A tokenizer ring 0 does not run, a token it cannot hold, merges it cannot resolve, a template it does not
 *        know and an end of turn it would not stop on are each refused by name.
 */
LPL_TEST(a_tokenizer_that_does_not_hold_is_refused_by_name)
{
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.model", lpl::convert::tests::text("llama"));
        test.check(refusedAs(model, "tokenizer_unknown", "llama"), "a SentencePiece tokenizer is tokenizer_unknown");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.pre", lpl::convert::tests::text("qwen2"));
        test.check(refusedAs(model, "pre_tokenizer_unknown", "qwen2"),
                   "qwen2's pre-tokenizer is pre_tokenizer_unknown");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.add_bos_token", lpl::convert::tests::boolean(true));
        test.check(refusedAs(model, "begin_of_text_unsupported", "add_bos_token"),
                   "a begin-of-text token added to every prompt is begin_of_text_unsupported");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.add_space_prefix", lpl::convert::tests::boolean(true));
        test.check(refusedAs(model, "space_prefix_unsupported", "add_space_prefix"),
                   "a space put before every prompt is space_prefix_unsupported");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        lpl::convert::MetadataValue types = model.source().metadata.at("tokenizer.ggml.token_type");
        types.signedWords[100] = 4;
        model.set("tokenizer.ggml.token_type", types);
        test.check(refusedAs(model, "token_type_unsupported", "100"),
                   "a USER_DEFINED token is token_type_unsupported, naming its id");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        lpl::convert::MetadataValue types = model.source().metadata.at("tokenizer.ggml.token_type");
        types.signedWords.pop_back();
        model.set("tokenizer.ggml.token_type", types);
        test.check(refusedAs(model, "vocabulary_size_disagrees", "token_type"),
                   "a type list a token short is vocabulary_size_disagrees");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("llama.vocab_size", lpl::convert::tests::unsignedWord(300u));
        test.check(refusedAs(model, "vocabulary_size_disagrees", "300"),
                   "and so is a vocabulary size other than the tokens'");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<std::string> tokens = tokensOf(model);
        tokens[263] = "";
        model.set("tokenizer.ggml.tokens", textList(tokens));
        test.check(refusedAs(model, "token_empty", "263"), "an empty token is token_empty");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<std::string> tokens = tokensOf(model);
        tokens[262] = "a b";
        model.set("tokenizer.ggml.tokens", textList(tokens));
        test.check(refusedAs(model, "token_not_byte_level", "262"),
                   "a NORMAL token holding a raw space is token_not_byte_level");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<std::string> tokens = tokensOf(model);
        tokens[0] = std::string(129u, 'x');
        model.set("tokenizer.ggml.tokens", textList(tokens));
        test.check(refusedAs(model, "token_too_long", "129"), "a token of 129 bytes is token_too_long");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<std::string> tokens = tokensOf(model);
        tokens[0] = std::string(128u, 'x');
        model.set("tokenizer.ggml.tokens", textList(tokens));
        test.check(outcomeOf(model) == "converted", "while 128 bytes is not too long");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<std::string> tokens = tokensOf(model);
        tokens[263] = tokens[3u + 'h'];
        model.set("tokenizer.ggml.tokens", textList(tokens));
        test.check(refusedAs(model, "token_repeated", "263"), "a token whose bytes another has is token_repeated");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.merges", textList({}));
        test.check(refusedAs(model, "merges_missing", "tokenizer.ggml.merges"),
                   "a byte-level vocabulary without a merge is merges_missing");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.removeKey("tokenizer.ggml.merges");
        test.check(refusedAs(model, "merges_missing", "tokenizer.ggml.merges"), "and so is one without the key");
    }
    const char *const unresolved[] = {"h e x", "he", "zz e", "h zz", "h t"};
    std::uint32_t unresolvedNamed = 0u;
    for (const char *merge : unresolved)
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<std::string> merges = mergesOf(model);
        merges[2] = merge;
        model.set("tokenizer.ggml.merges", textList(merges));
        unresolvedNamed += refusedAs(model, "merge_unresolved", "merge 2") ? 1u : 0u;
    }
    test.check(unresolvedNamed == 5u, "a merge of two spaces, of none, of an unknown side or whose concatenation is "
                                      "no token is merge_unresolved, naming its rank");
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<std::string> merges = mergesOf(model);
        merges[4] = merges[0];
        model.set("tokenizer.ggml.merges", textList(merges));
        test.check(refusedAs(model, "merges_refused", "repeated_pair"),
                   "a pair merged twice is merges_refused, with the index's reason");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.chat_template", lpl::convert::tests::text("[INST] {{ message }} [/INST]"));
        test.check(refusedAs(model, "chat_template_unknown", "tokenizer.chat_template"),
                   "a template without <|im_start|> is chat_template_unknown");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.eos_token_id", lpl::convert::tests::unsignedWord(5u));
        test.check(refusedAs(model, "end_of_turn_disagrees", "tokenizer.ggml.eos_token_id"),
                   "an end of sequence ring 0 would not stop on is end_of_turn_disagrees");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.eot_token_id", lpl::convert::tests::unsignedWord(1u));
        test.check(refusedAs(model, "end_of_turn_disagrees", "tokenizer.ggml.eot_token_id"),
                   "and so is an end of turn that opens a turn");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.eom_token_id", lpl::convert::tests::unsignedWord(300u));
        test.check(refusedAs(model, "end_of_turn_disagrees", "tokenizer.ggml.eom_token_id"),
                   "or an end of message past the vocabulary");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.set("tokenizer.ggml.eos_token_id", lpl::convert::tests::unsignedWord(0u));
        model.set("tokenizer.ggml.eot_token_id", lpl::convert::tests::unsignedWord(2u));
        test.check(outcomeOf(model) == "converted",
                   "while <|endoftext|> and <|im_end|>, the two ring 0 ends a turn on, convert");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        std::vector<std::string> tokens = tokensOf(model);
        tokens[2] = "<|im_stop|>";
        model.set("tokenizer.ggml.tokens", textList(tokens));
        test.check(refusedAs(model, "chat_tokens_missing", "<|im_end|>"),
                   "a ChatML template over a vocabulary without <|im_end|> is chat_tokens_missing");
    }
    {
        lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
        model.removeKey("tokenizer.chat_template");
        const lpl::convert::Conversion conversion = lpl::convert::convertModel(model.source(), options());
        test.check(!conversion.refusal.has_value() && conversion.chatFormat == lpl::infer::ChatFormat::Plain,
                   "and a model without a template converts as Plain");
    }
}
