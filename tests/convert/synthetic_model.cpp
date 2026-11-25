#include "synthetic_model.hpp"

#include "fixtures/gguf_reference.hpp"

#include <ggml.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>

namespace lpl::convert::tests {

namespace {

constexpr std::uint64_t kGoldenGamma = 0x9E3779B97F4A7C15ull;
constexpr std::uint32_t kNormalToken = 1u;
constexpr std::uint32_t kControlToken = 3u;
constexpr const char *kChatTemplate = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
                                      "message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}";

/** The turn tokens, in the order of their ids. */
constexpr const char *kTurnTokenTexts[kSyntheticTurnTokens] = {"<|endoftext|>", "<|im_start|>", "<|im_end|>"};

/**
 * @struct SyntheticMerge
 * @brief A merge of the vocabulary, by the bytes of its two sides.
 */
struct SyntheticMerge {
    const char *left;  ///< The left side's bytes.
    const char *right; ///< The right side's bytes.
};

/** The merges, in the order of their ranks; each result is the next merged token. */
constexpr SyntheticMerge kMerges[kSyntheticMerges] = {
    {"h",  "e" },
    {" ",  "t" },
    {" t", "he"},
    {"i",  "n" },
    {" ",  "in"},
};

/**
 * @brief splitmix64's next draw.
 */
[[nodiscard]] std::uint64_t nextDraw(std::uint64_t &state)
{
    state += kGoldenGamma;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/**
 * @brief A draw uniform in [-1, 1).
 */
[[nodiscard]] double uniform(std::uint64_t &state)
{
    return static_cast<double>(nextDraw(state) >> 11) / 4503599627370496.0 - 1.0;
}

/**
 * @brief @p codePoint in UTF-8.
 */
[[nodiscard]] std::string utf8(std::uint32_t codePoint)
{
    std::string out;

    if (codePoint < 0x80u)
    {
        out.push_back(static_cast<char>(codePoint));
        return out;
    }
    out.push_back(static_cast<char>(0xC0u | (codePoint >> 6)));
    out.push_back(static_cast<char>(0x80u | (codePoint & 0x3Fu)));
    return out;
}

/**
 * @brief @p bytes written in GPT-2's alphabet, as the fixture gives it.
 */
[[nodiscard]] std::string byteLevelText(const std::string &bytes)
{
    std::string out;

    for (const char byte : bytes)
        out += utf8(gguf_reference::kByteLevelCodePoints[static_cast<std::uint8_t>(byte)]);
    return out;
}

/**
 * @brief The ggml type of @p type.
 */
[[nodiscard]] ggml_type ggmlTypeOf(TensorType type)
{
    switch (type)
    {
    case TensorType::Float32: return GGML_TYPE_F32;
    case TensorType::Float16: return GGML_TYPE_F16;
    case TensorType::BrainFloat16: return GGML_TYPE_BF16;
    case TensorType::QuantisedEightBit: return GGML_TYPE_Q8_0;
    case TensorType::Other: break;
    }
    throw std::invalid_argument("a synthetic tensor of no type ggml stores");
}

/**
 * @brief The file's name for @p type.
 */
[[nodiscard]] std::string typeNameOf(TensorType type) { return ggml_type_name(ggmlTypeOf(type)); }

/**
 * @brief @p values stored as @p type, @p columns to a row.
 */
[[nodiscard]] std::vector<std::uint8_t> stored(TensorType type, const std::vector<float> &values, std::uint64_t columns)
{
    const ggml_type ggml = ggmlTypeOf(type);
    const std::uint64_t rows = values.size() / columns;
    std::vector<std::uint8_t> bytes(ggml_row_size(ggml, static_cast<std::int64_t>(columns)) * rows);

    if (type == TensorType::Float32)
        std::memcpy(bytes.data(), values.data(), bytes.size());
    else if (type == TensorType::Float16)
        ggml_fp32_to_fp16_row(values.data(), reinterpret_cast<ggml_fp16_t *>(bytes.data()),
                              static_cast<std::int64_t>(values.size()));
    else if (type == TensorType::BrainFloat16)
        ggml_fp32_to_bf16_row(values.data(), reinterpret_cast<ggml_bf16_t *>(bytes.data()),
                              static_cast<std::int64_t>(values.size()));
    else
        ggml_quantize_chunk(ggml, values.data(), bytes.data(), 0, static_cast<std::int64_t>(rows),
                            static_cast<std::int64_t>(columns), nullptr);
    return bytes;
}

/**
 * @brief The file type general.file_type names for matrices of @p weights: llama_ftype's values.
 */
[[nodiscard]] std::uint32_t fileTypeOf(SyntheticWeights weights)
{
    switch (weights)
    {
    case SyntheticWeights::Float32: return 0u;
    case SyntheticWeights::Float16: return 1u;
    case SyntheticWeights::QuantisedEightBit: return 7u;
    case SyntheticWeights::BrainFloat16: return 32u;
    }
    return 0u;
}

/**
 * @brief The tensor type of matrices of @p weights.
 */
[[nodiscard]] TensorType tensorTypeOf(SyntheticWeights weights)
{
    switch (weights)
    {
    case SyntheticWeights::Float32: return TensorType::Float32;
    case SyntheticWeights::Float16: return TensorType::Float16;
    case SyntheticWeights::BrainFloat16: return TensorType::BrainFloat16;
    case SyntheticWeights::QuantisedEightBit: return TensorType::QuantisedEightBit;
    }
    return TensorType::Other;
}

/**
 * @brief A key's value of kind TextList.
 */
[[nodiscard]] MetadataValue textList(const std::vector<std::string> &values)
{
    MetadataValue value;

    value.kind = MetadataKind::TextList;
    value.typeName = "arr[str]";
    value.texts = values;
    return value;
}

/**
 * @brief A key's value of kind SignedWordList.
 */
[[nodiscard]] MetadataValue signedWordList(const std::vector<std::int32_t> &values)
{
    MetadataValue value;

    value.kind = MetadataKind::SignedWordList;
    value.typeName = "arr[i32]";
    value.signedWords = values;
    return value;
}

/**
 * @brief Writes key @p key of @p value into @p context.
 */
void writeKey(gguf_context *context, const std::string &key, const MetadataValue &value)
{
    std::vector<const char *> texts;

    switch (value.kind)
    {
    case MetadataKind::UnsignedWord: gguf_set_val_u32(context, key.c_str(), value.unsignedWord); return;
    case MetadataKind::Real: gguf_set_val_f32(context, key.c_str(), value.real); return;
    case MetadataKind::Boolean: gguf_set_val_bool(context, key.c_str(), value.boolean); return;
    case MetadataKind::Text: gguf_set_val_str(context, key.c_str(), value.text.c_str()); return;
    case MetadataKind::TextList:
        for (const std::string &text : value.texts)
            texts.push_back(text.c_str());
        gguf_set_arr_str(context, key.c_str(), texts.data(), static_cast<int>(texts.size()));
        return;
    case MetadataKind::SignedWordList:
        gguf_set_arr_data(context, key.c_str(), GGUF_TYPE_INT32, value.signedWords.data(),
                          static_cast<int>(value.signedWords.size()));
        return;
    case MetadataKind::Other: return;
    }
}

} // namespace

SyntheticModel::SyntheticModel(SyntheticWeights weights)
{
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    std::vector<std::string> merges;

    for (const char *turn : kTurnTokenTexts)
    {
        tokens.emplace_back(turn);
        types.push_back(static_cast<std::int32_t>(kControlToken));
        _tokenBytes.emplace_back(turn);
    }
    for (std::uint32_t byte = 0u; byte < kSyntheticByteTokens; ++byte)
    {
        const std::string bytes(1u, static_cast<char>(byte));
        tokens.push_back(byteLevelText(bytes));
        types.push_back(static_cast<std::int32_t>(kNormalToken));
        _tokenBytes.push_back(bytes);
    }
    std::map<std::string, std::uint32_t> idOf;
    for (std::uint32_t id = 0u; id < _tokenBytes.size(); ++id)
        idOf[_tokenBytes[id]] = id;
    for (const SyntheticMerge &merge : kMerges)
    {
        const std::string left = merge.left;
        const std::string right = merge.right;
        const std::uint32_t result = static_cast<std::uint32_t>(tokens.size());
        merges.push_back(byteLevelText(left) + " " + byteLevelText(right));
        tokens.push_back(byteLevelText(left + right));
        types.push_back(static_cast<std::int32_t>(kNormalToken));
        _tokenBytes.push_back(left + right);
        idOf[left + right] = result;
        _mergeIds.insert(_mergeIds.end(), {idOf.at(left), idOf.at(right), result});
    }

    set("general.architecture", text("llama"));
    set("general.type", text("model"));
    set("general.name", text("synthetic"));
    set("general.file_type", unsignedWord(fileTypeOf(weights)));
    set("general.quantization_version", unsignedWord(2u));
    set("llama.block_count", unsignedWord(kSyntheticLayers));
    set("llama.context_length", unsignedWord(kSyntheticContext));
    set("llama.embedding_length", unsignedWord(kSyntheticDim));
    set("llama.feed_forward_length", unsignedWord(kSyntheticFeedForward));
    set("llama.attention.head_count", unsignedWord(kSyntheticHeads));
    set("llama.attention.head_count_kv", unsignedWord(kSyntheticKeyValueHeads));
    set("llama.rope.freq_base", real(static_cast<float>(kSyntheticRotaryBase)));
    set("llama.attention.layer_norm_rms_epsilon", real(static_cast<float>(kSyntheticNormEpsilon)));
    set("llama.vocab_size", unsignedWord(kSyntheticVocabulary));
    set("llama.rope.dimension_count", unsignedWord(kSyntheticHeadDim));
    set("tokenizer.ggml.add_space_prefix", boolean(false));
    set("tokenizer.ggml.add_bos_token", boolean(false));
    set("tokenizer.ggml.model", text("gpt2"));
    set("tokenizer.ggml.pre", text("smollm"));
    set("tokenizer.ggml.tokens", textList(tokens));
    set("tokenizer.ggml.token_type", signedWordList(types));
    set("tokenizer.ggml.merges", textList(merges));
    set("tokenizer.ggml.bos_token_id", unsignedWord(1u));
    set("tokenizer.ggml.eos_token_id", unsignedWord(2u));
    set("tokenizer.ggml.unknown_token_id", unsignedWord(0u));
    set("tokenizer.ggml.padding_token_id", unsignedWord(2u));
    set("tokenizer.chat_template", text(kChatTemplate));

    const TensorType matrices = tensorTypeOf(weights);
    const double attention = std::sqrt(3.0 / kSyntheticDim);
    drawMatrix("token_embd.weight", kSyntheticVocabulary, kSyntheticDim, 1.0, matrices);
    for (std::uint32_t layer = 0u; layer < kSyntheticLayers; ++layer)
    {
        const std::string block = "blk." + std::to_string(layer) + ".";
        drawNorm(block + "attn_norm.weight", kSyntheticDim);
        drawMatrix(block + "attn_q.weight", kSyntheticDim, kSyntheticDim, attention, matrices);
        drawMatrix(block + "attn_k.weight", kSyntheticKeyValueHeads * kSyntheticHeadDim, kSyntheticDim, attention,
                   matrices);
        drawMatrix(block + "attn_v.weight", kSyntheticKeyValueHeads * kSyntheticHeadDim, kSyntheticDim, attention,
                   matrices);
        drawMatrix(block + "attn_output.weight", kSyntheticDim, kSyntheticDim, attention, matrices);
        drawNorm(block + "ffn_norm.weight", kSyntheticDim);
        drawMatrix(block + "ffn_gate.weight", kSyntheticFeedForward, kSyntheticDim, attention, matrices);
        drawMatrix(block + "ffn_up.weight", kSyntheticFeedForward, kSyntheticDim, attention, matrices);
        drawMatrix(block + "ffn_down.weight", kSyntheticDim, kSyntheticFeedForward,
                   std::sqrt(3.0 / kSyntheticFeedForward), matrices);
    }
    drawNorm("output_norm.weight", kSyntheticDim);
}

double SyntheticModel::drawn(const std::string &name, std::uint64_t index) const
{
    return static_cast<double>(_drawn.at(name).at(index));
}

const std::string &SyntheticModel::tokenBytes(std::uint32_t id) const { return _tokenBytes.at(id); }

void SyntheticModel::set(const std::string &key, const MetadataValue &value) { _source.metadata[key] = value; }

void SyntheticModel::removeKey(const std::string &key) { _source.metadata.erase(key); }

void SyntheticModel::putTensor(const std::string &name, TensorType type, const std::vector<std::uint64_t> &extents,
                               const std::vector<float> &values)
{
    std::vector<std::uint8_t> &bytes = _storage[name];
    bytes = stored(type, values, extents.front());
    _drawn[name] = values;

    SourceTensor &tensor = _source.tensors[name];
    tensor.type = type;
    tensor.typeName = typeNameOf(type);
    tensor.extents = extents;
    tensor.bytes = bytes.data();
    tensor.byteCount = bytes.size();
}

SourceTensor &SyntheticModel::tensor(const std::string &name) { return _source.tensors.at(name); }

void SyntheticModel::removeTensor(const std::string &name)
{
    _source.tensors.erase(name);
    _storage.erase(name);
    _drawn.erase(name);
}

void SyntheticModel::drawMatrix(const std::string &name, std::uint32_t rows, std::uint32_t columns, double amplitude,
                                TensorType type)
{
    std::uint64_t state = ++_stream;
    std::vector<float> values(static_cast<std::size_t>(rows) * columns);

    for (float &value : values)
        value = static_cast<float>(amplitude * uniform(state));
    putTensor(name, type, {columns, rows}, values);
}

void SyntheticModel::drawNorm(const std::string &name, std::uint32_t count)
{
    std::uint64_t state = ++_stream;
    std::vector<float> values(count);

    for (float &value : values)
        value = static_cast<float>(1.0 + 0.25 * uniform(state));
    putTensor(name, TensorType::Float32, {count}, values);
}

void SyntheticModel::write(const std::string &path) const
{
    gguf_context *const file = gguf_init_empty();
    for (const auto &[key, value] : _source.metadata)
        writeKey(file, key, value);

    std::size_t bytes = 0u;
    for (const auto &[name, tensor] : _source.tensors)
        bytes += static_cast<std::size_t>(tensor.byteCount) + ggml_tensor_overhead();
    ggml_init_params params{bytes + (1u << 20), nullptr, false};
    ggml_context *const tensors = ggml_init(params);
    for (const auto &[name, tensor] : _source.tensors)
    {
        std::int64_t extents[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (std::size_t i = 0u; i < tensor.extents.size(); ++i)
            extents[i] = static_cast<std::int64_t>(tensor.extents[i]);
        ggml_tensor *const made =
            ggml_new_tensor(tensors, ggmlTypeOf(tensor.type), static_cast<int>(tensor.extents.size()), extents);
        std::memcpy(made->data, tensor.bytes, ggml_nbytes(made));
        ggml_set_name(made, name.c_str());
        gguf_add_tensor(file, made);
    }
    gguf_write_to_file(file, path.c_str(), false);
    ggml_free(tensors);
    gguf_free(file);
}

MetadataValue unsignedWord(std::uint32_t value)
{
    MetadataValue made;

    made.kind = MetadataKind::UnsignedWord;
    made.typeName = "u32";
    made.unsignedWord = value;
    return made;
}

MetadataValue real(float value)
{
    MetadataValue made;

    made.kind = MetadataKind::Real;
    made.typeName = "f32";
    made.real = value;
    return made;
}

MetadataValue boolean(bool value)
{
    MetadataValue made;

    made.kind = MetadataKind::Boolean;
    made.typeName = "bool";
    made.boolean = value;
    return made;
}

MetadataValue text(const std::string &value)
{
    MetadataValue made;

    made.kind = MetadataKind::Text;
    made.typeName = "str";
    made.text = value;
    return made;
}

std::string temporaryPath(const std::string &name)
{
    return (std::filesystem::temp_directory_path() /
            ("lpl-convert-" + std::to_string(static_cast<long>(getpid())) + "-" + name))
        .string();
}

} // namespace lpl::convert::tests
