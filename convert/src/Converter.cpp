#include <lpl/convert/Converter.hpp>

#include <lpl/convert/ByteLevel.hpp>
#include <lpl/convert/Pairing.hpp>
#include <lpl/convert/Rotary.hpp>
#include <lpl/infer/TensorArena.hpp>
#include <lpl/infer/Tokenizer.hpp>

#include <cmath>
#include <cstdio>
#include <map>
#include <set>

namespace lpl::convert {

namespace {

constexpr std::uint32_t kNormalToken = 1u;                  ///< llama.cpp's LLAMA_TOKEN_TYPE_NORMAL.
constexpr std::uint32_t kControlToken = 3u;                 ///< llama.cpp's LLAMA_TOKEN_TYPE_CONTROL.
constexpr double kEpsilonUnits = 4294967296.0;              ///< 2^32 units of an epsilon in 1.
constexpr double kLargestEpsilonUnits = 4294967295.0;       ///< The largest word an epsilon takes.
constexpr std::uint64_t kLargestImageBytes = 0xFFFFFFFFull; ///< Model::open and writeBlob count in 32 bits.
constexpr const char *kChatMlOpening = "<|im_start|>";      ///< What llama.cpp b3775 finds a ChatML template by.
constexpr std::size_t kArenaSlack = 1u << 20;               ///< Room an arena keeps past the indexes it holds.

/**
 * @struct ArchitectureRow
 * @brief What an architecture brings: its name, the prefix of its keys, and how it pairs its rotation's channels.
 */
struct ArchitectureRow {
    const char *name;      ///< `general.architecture`.
    RotaryPairing pairing; ///< llama.cpp's llama_rope_type for it.
};

/** The architectures the converter knows: the first converts; qwen2 is refused at its tensors, until its biases. */
constexpr ArchitectureRow kArchitectures[] = {
    {"llama", RotaryPairing::Adjacent },
    {"qwen2", RotaryPairing::HalfSplit},
};

/**
 * @enum KeyUse
 * @brief Whether a key the converter knows must be there.
 */
enum class KeyUse : std::uint8_t {
    Required, ///< Missing, it is refused as metadata_missing.
    Optional, ///< Missing, it changes nothing ring 0 runs.
};

/**
 * @struct KnownKey
 * @brief A key the converter reads: its name, past the architecture's prefix for an architecture key, and its kind.
 */
struct KnownKey {
    const char *name;  ///< The key.
    MetadataKind kind; ///< The kind it must have.
    KeyUse use;        ///< Whether it must be there.
};

/** The keys of an architecture's namespace, named past its prefix: SmolLM2's ten. */
constexpr KnownKey kArchitectureKeys[] = {
    {"block_count",                      MetadataKind::UnsignedWord, KeyUse::Required},
    {"context_length",                   MetadataKind::UnsignedWord, KeyUse::Required},
    {"embedding_length",                 MetadataKind::UnsignedWord, KeyUse::Required},
    {"feed_forward_length",              MetadataKind::UnsignedWord, KeyUse::Required},
    {"attention.head_count",             MetadataKind::UnsignedWord, KeyUse::Required},
    {"attention.head_count_kv",          MetadataKind::UnsignedWord, KeyUse::Required},
    {"rope.freq_base",                   MetadataKind::Real,         KeyUse::Required},
    {"attention.layer_norm_rms_epsilon", MetadataKind::Real,         KeyUse::Required},
    {"rope.dimension_count",             MetadataKind::UnsignedWord, KeyUse::Required},
    {"vocab_size",                       MetadataKind::UnsignedWord, KeyUse::Optional},
};

/** The keys of the tokenizer: SmolLM2's twelve, and the two other ends of a turn llama.cpp reads. */
constexpr KnownKey kTokenizerKeys[] = {
    {"tokenizer.ggml.model",            MetadataKind::Text,           KeyUse::Required},
    {"tokenizer.ggml.pre",              MetadataKind::Text,           KeyUse::Required},
    {"tokenizer.ggml.tokens",           MetadataKind::TextList,       KeyUse::Required},
    {"tokenizer.ggml.token_type",       MetadataKind::SignedWordList, KeyUse::Required},
    {"tokenizer.ggml.eos_token_id",     MetadataKind::UnsignedWord,   KeyUse::Required},
    {"tokenizer.ggml.add_bos_token",    MetadataKind::Boolean,        KeyUse::Required},
    {"tokenizer.ggml.merges",           MetadataKind::TextList,       KeyUse::Optional},
    {"tokenizer.ggml.bos_token_id",     MetadataKind::UnsignedWord,   KeyUse::Optional},
    {"tokenizer.ggml.unknown_token_id", MetadataKind::UnsignedWord,   KeyUse::Optional},
    {"tokenizer.ggml.padding_token_id", MetadataKind::UnsignedWord,   KeyUse::Optional},
    {"tokenizer.ggml.eot_token_id",     MetadataKind::UnsignedWord,   KeyUse::Optional},
    {"tokenizer.ggml.eom_token_id",     MetadataKind::UnsignedWord,   KeyUse::Optional},
    {"tokenizer.ggml.add_space_prefix", MetadataKind::Boolean,        KeyUse::Optional},
    {"tokenizer.chat_template",         MetadataKind::Text,           KeyUse::Optional},
};

/** The key that names the architecture, the one `general.*` key the converter reads. */
constexpr KnownKey kArchitectureKey{"general.architecture", MetadataKind::Text, KeyUse::Required};

/**
 * @struct PreTokenizerRow
 * @brief A pre-tokenizer of llama.cpp, by its `tokenizer.ggml.pre`, and ring 0's.
 */
struct PreTokenizerRow {
    const char *name;                 ///< `tokenizer.ggml.pre`.
    infer::PreTokenizer preTokenizer; ///< What ring 0 runs: llama-vocab.cpp:404-410's regexes for smollm.
};

/** The pre-tokenizers ring 0 runs. */
constexpr PreTokenizerRow kPreTokenizers[] = {
    {"smollm", infer::PreTokenizer::IndividualDigitsThenByteLevel},
};

/** The parts of a layer a bias may be given for, which v1 cannot hold. */
constexpr const char *kBiasedParts[] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"};

/**
 * @brief A refusal named @p name, its detail formatted as printf formats it.
 */
template <typename... Values> [[nodiscard]] Refusal refusal(const char *name, const char *format, Values... values)
{
    if constexpr (sizeof...(Values) == 0u)
        return Refusal{name, format};
    else
    {
        char detail[512];
        std::snprintf(detail, sizeof(detail), format, values...);
        return Refusal{name, detail};
    }
}

/**
 * @brief The name ring 0's index gives @p verdict, for a merges_refused's detail.
 */
[[nodiscard]] const char *mergesVerdictName(infer::BytePairMergesVerdict verdict) noexcept
{
    switch (verdict)
    {
    case infer::BytePairMergesVerdict::Indexed: return "indexed";
    case infer::BytePairMergesVerdict::UnknownToken: return "unknown_token";
    case infer::BytePairMergesVerdict::NotTheConcatenation: return "not_the_concatenation";
    case infer::BytePairMergesVerdict::RepeatedPair: return "repeated_pair";
    case infer::BytePairMergesVerdict::ArenaExhausted: return "arena_exhausted";
    case infer::BytePairMergesVerdict::Crowded: return "crowded";
    }
    return "unknown";
}

/**
 * @struct Keys
 * @brief The keys of a model, and the row of its architecture.
 */
struct Keys {
    const std::map<std::string, MetadataValue> &values; ///< Every key.
    const ArchitectureRow *architecture{nullptr};       ///< Its row, once found.
    std::string prefix;                                 ///< `<architecture>.`.

    /**
     * @brief The value of key @p name, or nullptr when the model lacks it.
     */
    [[nodiscard]] const MetadataValue *find(const std::string &name) const
    {
        const auto found = values.find(name);
        return found == values.end() ? nullptr : &found->second;
    }
};

/**
 * @brief The refusal of key @p key, whose kind is not the one @p known says; nothing when it is.
 */
[[nodiscard]] std::optional<Refusal> kindRefusal(const std::string &key, const MetadataValue &value,
                                                 const KnownKey &known)
{
    if (value.kind == known.kind)
        return std::nullopt;
    return refusal("metadata_type_wrong", "%s is of type %s, not the one ring 0 reads it as", key.c_str(),
                   value.typeName.c_str());
}

/**
 * @brief Finds the row of `general.architecture`.
 */
[[nodiscard]] std::optional<Refusal> findArchitecture(Keys &keys)
{
    const MetadataValue *const name = keys.find(kArchitectureKey.name);
    if (name == nullptr)
        return refusal("metadata_missing", "%s is missing", kArchitectureKey.name);
    if (std::optional<Refusal> wrong = kindRefusal(kArchitectureKey.name, *name, kArchitectureKey))
        return wrong;
    for (const ArchitectureRow &row : kArchitectures)
    {
        if (name->text == row.name)
        {
            keys.architecture = &row;
            keys.prefix = std::string(row.name) + ".";
            return std::nullopt;
        }
    }
    return refusal("architecture_unknown", "general.architecture %s has no row: the converter knows llama",
                   name->text.c_str());
}

/**
 * @brief The key @p name names, among @p known; nullptr when none does.
 */
template <std::size_t Count>
[[nodiscard]] const KnownKey *knownAmong(const KnownKey (&known)[Count], const std::string &name)
{
    for (const KnownKey &key : known)
    {
        if (name == key.name)
            return &key;
    }
    return nullptr;
}

/**
 * @brief Refuses a key of the architecture's or the tokenizer's namespace the converter does not know, one of another
 *        namespace, and one of the wrong kind; `general.*` keys only describe the file.
 */
[[nodiscard]] std::optional<Refusal> checkEveryKey(const Keys &keys)
{
    for (const auto &[name, value] : keys.values)
    {
        if (name.rfind("general.", 0u) == 0u)
            continue;
        const KnownKey *known = nullptr;
        if (name.rfind(keys.prefix, 0u) == 0u)
        {
            const std::string suffix = name.substr(keys.prefix.size());
            if (suffix.rfind("rope.scaling.", 0u) == 0u)
                return refusal("rotary_scaled", "%s scales the rotation, which ring 0 turns unscaled", name.c_str());
            known = knownAmong(kArchitectureKeys, suffix);
        }
        else if (name.rfind("tokenizer.", 0u) == 0u)
            known = knownAmong(kTokenizerKeys, name);
        if (known == nullptr)
            return refusal("metadata_unknown", "%s is a key the converter does not know", name.c_str());
        if (std::optional<Refusal> wrong = kindRefusal(name, value, *known))
            return wrong;
    }
    return std::nullopt;
}

/**
 * @brief Refuses the first key ring 0 needs that the model lacks.
 */
[[nodiscard]] std::optional<Refusal> checkRequiredKeys(const Keys &keys)
{
    for (const KnownKey &key : kArchitectureKeys)
    {
        const std::string name = keys.prefix + key.name;
        if (key.use == KeyUse::Required && keys.find(name) == nullptr)
            return refusal("metadata_missing", "%s is missing", name.c_str());
    }
    for (const KnownKey &key : kTokenizerKeys)
    {
        if (key.use == KeyUse::Required && keys.find(key.name) == nullptr)
            return refusal("metadata_missing", "%s is missing", key.name);
    }
    return std::nullopt;
}

/**
 * @brief The word of architecture key @p suffix, which checkRequiredKeys found.
 */
[[nodiscard]] std::uint32_t wordOf(const Keys &keys, const char *suffix)
{
    return keys.find(keys.prefix + suffix)->unsignedWord;
}

/**
 * @struct Plan
 * @brief What the keys decide: the shape, the tokenizer, the turns and the rotation.
 */
struct Plan {
    infer::ModelConfig shape{};                                       ///< The image's shape.
    RotaryPairing pairing{RotaryPairing::Adjacent};                   ///< How the source pairs its channels.
    std::vector<std::uint32_t> rotary;                                ///< The rotary table.
    std::string vocabularyText;                                       ///< The tokens' bytes, one after the other.
    std::vector<infer::VocabEntry> vocabularyEntries;                 ///< Where each token's bytes lie.
    std::vector<infer::BytePairMerge> merges;                         ///< The merges, by their ranks.
    infer::PreTokenizer preTokenizer{infer::PreTokenizer::WholeText}; ///< How text splits.
    infer::ChatFormat chatFormat{infer::ChatFormat::Plain};           ///< How messages are put in turns.
};

/**
 * @brief Decides the shape from the architecture's keys and the caller's context.
 */
[[nodiscard]] std::optional<Refusal> planShape(const Keys &keys, const ConversionOptions &options, Plan &plan)
{
    infer::ModelConfig &shape = plan.shape;
    shape.layers = wordOf(keys, "block_count");
    shape.dim = wordOf(keys, "embedding_length");
    shape.ffnHidden = wordOf(keys, "feed_forward_length");
    shape.heads = wordOf(keys, "attention.head_count");
    shape.keyValueHeads = wordOf(keys, "attention.head_count_kv");
    const std::uint32_t trained = wordOf(keys, "context_length");

    if (shape.layers == 0u || shape.dim == 0u || shape.ffnHidden == 0u)
        return refusal("shape_refused", "%sblock_count %u, embedding_length %u, feed_forward_length %u: none may be 0",
                       keys.prefix.c_str(), shape.layers, shape.dim, shape.ffnHidden);
    if (shape.heads == 0u || shape.dim % shape.heads != 0u)
        return refusal("shape_refused", "%sattention.head_count %u does not divide %sembedding_length %u",
                       keys.prefix.c_str(), shape.heads, keys.prefix.c_str(), shape.dim);
    if (shape.keyValueHeads == 0u || shape.heads % shape.keyValueHeads != 0u)
        return refusal("shape_refused", "%sattention.head_count_kv %u does not divide %sattention.head_count %u",
                       keys.prefix.c_str(), shape.keyValueHeads, keys.prefix.c_str(), shape.heads);
    if (options.contextLength == 0u)
        return refusal("context_missing", "no context was given: the image's cache needs its positions");
    if (options.contextLength > trained)
        return refusal("context_past_training", "a context of %u positions is past the %u the model was trained on",
                       options.contextLength, trained);
    if (options.alignmentLog2 < infer::kModelImageLeastAlignmentLog2 ||
        options.alignmentLog2 > infer::kModelImageMostAlignmentLog2)
        return refusal("alignment_refused", "an expert alignment of 2^%u is outside 2^12 to 2^22",
                       options.alignmentLog2);
    shape.contextLength = options.contextLength;
    return std::nullopt;
}

/**
 * @brief Decides the epsilon and the rotation from the architecture's keys.
 */
[[nodiscard]] std::optional<Refusal> planNormAndRotation(const Keys &keys, Plan &plan)
{
    const double epsilon = keys.find(keys.prefix + "attention.layer_norm_rms_epsilon")->real;
    const double epsilonUnits = std::floor(epsilon * kEpsilonUnits + 0.5);
    if (!std::isfinite(epsilon) || epsilon < 0.0 || epsilonUnits > kLargestEpsilonUnits)
        return refusal("metadata_out_of_range", "%sattention.layer_norm_rms_epsilon %.9g is not in [0, 1)",
                       keys.prefix.c_str(), epsilon);
    plan.shape.normEpsilon = static_cast<std::uint32_t>(epsilonUnits);

    const std::uint32_t rotated = wordOf(keys, "rope.dimension_count");
    if (rotated != plan.shape.headDim())
        return refusal("rotary_partial",
                       "%srope.dimension_count %u is not the head's %u channels: ring 0 turns them all",
                       keys.prefix.c_str(), rotated, plan.shape.headDim());
    plan.pairing = keys.architecture->pairing;
    return rotaryTurnsPerPosition(keys.find(keys.prefix + "rope.freq_base")->real, rotated, plan.rotary);
}

/**
 * @brief Refuses a tokenizer ring 0 does not run, and decides its pre-tokenizer.
 */
[[nodiscard]] std::optional<Refusal> planTokenizer(const Keys &keys, Plan &plan)
{
    const std::string &model = keys.find("tokenizer.ggml.model")->text;
    if (model != "gpt2")
        return refusal("tokenizer_unknown", "tokenizer.ggml.model %s is not the byte-level BPE (gpt2) ring 0 runs",
                       model.c_str());
    const std::string &pre = keys.find("tokenizer.ggml.pre")->text;
    bool known = false;
    for (const PreTokenizerRow &row : kPreTokenizers)
    {
        if (pre == row.name)
        {
            plan.preTokenizer = row.preTokenizer;
            known = true;
        }
    }
    if (!known)
        return refusal("pre_tokenizer_unknown", "tokenizer.ggml.pre %s is not one ring 0 runs: smollm", pre.c_str());
    if (keys.find("tokenizer.ggml.add_bos_token")->boolean)
        return refusal("begin_of_text_unsupported",
                       "tokenizer.ggml.add_bos_token: ring 0 puts no begin-of-text token before a prompt");
    const MetadataValue *const spacePrefix = keys.find("tokenizer.ggml.add_space_prefix");
    if (spacePrefix != nullptr && spacePrefix->boolean)
        return refusal("space_prefix_unsupported",
                       "tokenizer.ggml.add_space_prefix: ring 0 puts no space before a prompt");
    return std::nullopt;
}

/**
 * @brief The bytes token @p id of type @p type stands for, or its refusal.
 */
[[nodiscard]] std::optional<Refusal> tokenBytes(std::uint32_t id, const std::string &text, std::int32_t type,
                                                std::string &bytes)
{
    if (type == static_cast<std::int32_t>(kControlToken))
        bytes = text;
    else if (type == static_cast<std::int32_t>(kNormalToken))
    {
        const std::optional<std::string> decoded = byteLevelBytes(text);
        if (!decoded.has_value())
            return refusal("token_not_byte_level", "token %u is not written in GPT-2's byte-level alphabet", id);
        bytes = *decoded;
    }
    else
        return refusal("token_type_unsupported", "token %u is of type %d: ring 0 takes NORMAL (1) and CONTROL (3)", id,
                       type);
    if (bytes.empty())
        return refusal("token_empty", "token %u stands for no byte", id);
    if (bytes.size() > infer::kMaxTokenBytes)
        return refusal("token_too_long", "token %u is %zu bytes, past the %u a token may be", id, bytes.size(),
                       infer::kMaxTokenBytes);
    return std::nullopt;
}

/**
 * @brief Decodes every token into the vocabulary's text and entries.
 */
[[nodiscard]] std::optional<Refusal> planVocabulary(const Keys &keys, Plan &plan)
{
    const std::vector<std::string> &tokens = keys.find("tokenizer.ggml.tokens")->texts;
    const std::vector<std::int32_t> &types = keys.find("tokenizer.ggml.token_type")->signedWords;
    if (types.size() != tokens.size())
        return refusal("vocabulary_size_disagrees", "tokenizer.ggml.token_type holds %zu types for %zu tokens",
                       types.size(), tokens.size());
    const MetadataValue *const size = keys.find(keys.prefix + "vocab_size");
    if (size != nullptr && size->unsignedWord != tokens.size())
        return refusal("vocabulary_size_disagrees", "%svocab_size %u is not the %zu tokens'", keys.prefix.c_str(),
                       size->unsignedWord, tokens.size());

    std::map<std::string, std::uint32_t> firstWithBytes;
    for (std::uint32_t id = 0u; id < tokens.size(); ++id)
    {
        std::string bytes;
        if (std::optional<Refusal> refused = tokenBytes(id, tokens[id], types[id], bytes))
            return refused;
        const auto [held, inserted] = firstWithBytes.emplace(bytes, id);
        if (!inserted)
            return refusal("token_repeated", "token %u has the bytes of token %u", id, held->second);
        plan.vocabularyEntries.push_back(infer::VocabEntry{static_cast<std::uint32_t>(plan.vocabularyText.size()),
                                                           static_cast<std::uint32_t>(bytes.size())});
        plan.vocabularyText += bytes;
    }
    plan.shape.vocabSize = static_cast<std::uint32_t>(tokens.size());
    return std::nullopt;
}

/**
 * @brief Resolves every merge to its two tokens' ids and their concatenation's, by the tokens' own text.
 */
[[nodiscard]] std::optional<Refusal> planMerges(const Keys &keys, Plan &plan)
{
    const MetadataValue *const merges = keys.find("tokenizer.ggml.merges");
    if (merges == nullptr || merges->texts.empty())
        return refusal("merges_missing",
                       "tokenizer.ggml.merges holds no merge: a byte-level BPE vocabulary needs them");

    const std::vector<std::string> &tokens = keys.find("tokenizer.ggml.tokens")->texts;
    std::map<std::string, std::uint32_t> idOf;
    for (std::uint32_t id = 0u; id < tokens.size(); ++id)
        idOf.emplace(tokens[id], id);
    for (std::uint32_t rank = 0u; rank < merges->texts.size(); ++rank)
    {
        const std::string &merge = merges->texts[rank];
        const std::size_t space = merge.find(' ');
        if (space == std::string::npos || merge.find(' ', space + 1u) != std::string::npos)
            return refusal("merge_unresolved", "merge %u, \"%s\", is not two tokens apart by one space", rank,
                           merge.c_str());
        const std::string left = merge.substr(0u, space);
        const std::string right = merge.substr(space + 1u);
        const auto leftId = idOf.find(left);
        const auto rightId = idOf.find(right);
        const auto resultId = idOf.find(left + right);
        if (leftId == idOf.end() || rightId == idOf.end() || resultId == idOf.end())
            return refusal("merge_unresolved", "merge %u, \"%s\", names a token, or makes one, the vocabulary lacks",
                           rank, merge.c_str());
        plan.merges.push_back(infer::BytePairMerge{leftId->second, rightId->second, resultId->second});
    }
    return std::nullopt;
}

/**
 * @brief Bytes an arena holds for the indexes of @p plan's vocabulary and merges.
 */
[[nodiscard]] std::size_t indexArenaBytes(const Plan &plan)
{
    return 16u * (plan.vocabularyEntries.size() + 4u * plan.merges.size()) + kArenaSlack;
}

/**
 * @brief Indexes the merges as ring 0 will, to name the reason it would refuse them.
 */
[[nodiscard]] std::optional<Refusal> checkMergesIndex(const Plan &plan)
{
    infer::TensorArena arena{indexArenaBytes(plan)};
    infer::Vocab vocab;
    infer::BytePairMerges merges;

    if (!vocab.borrow(arena, plan.vocabularyText.data(), static_cast<core::u32>(plan.vocabularyText.size()),
                      plan.vocabularyEntries.data(), plan.shape.vocabSize))
        return refusal("merges_refused", "the vocabulary's index does not fit");
    const infer::BytePairMergesVerdict verdict =
        merges.borrow(arena, plan.merges.data(), static_cast<core::u32>(plan.merges.size()), vocab);
    if (verdict != infer::BytePairMergesVerdict::Indexed)
        return refusal("merges_refused", "the merges are refused as %s", mergesVerdictName(verdict));
    return std::nullopt;
}

/**
 * @brief Decides the chat format, and refuses an end of a turn ring 0 would not stop on.
 */
[[nodiscard]] std::optional<Refusal> planTurns(const Keys &keys, Plan &plan)
{
    const MetadataValue *const chatTemplate = keys.find("tokenizer.chat_template");
    if (chatTemplate == nullptr)
        return std::nullopt;
    if (chatTemplate->text.find(kChatMlOpening) == std::string::npos)
        return refusal("chat_template_unknown",
                       "tokenizer.chat_template is not ChatML, which llama.cpp b3775 knows by <|im_start|>");
    plan.chatFormat = infer::ChatFormat::ChatMl;

    infer::TensorArena arena{indexArenaBytes(plan)};
    infer::Vocab vocab;
    infer::ChatTokens tokens;
    if (!vocab.borrow(arena, plan.vocabularyText.data(), static_cast<core::u32>(plan.vocabularyText.size()),
                      plan.vocabularyEntries.data(), plan.shape.vocabSize) ||
        !infer::findChatTokens(plan.chatFormat, vocab, tokens))
        return refusal("chat_tokens_missing",
                       "a ChatML template over a vocabulary without <|im_start|> and <|im_end|>");

    for (const char *key :
         {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"})
    {
        const MetadataValue *const end = keys.find(key);
        if (end != nullptr && !tokens.endsTurn(end->unsignedWord))
            return refusal("end_of_turn_disagrees",
                           "%s %u is not a token ring 0 ends a turn on: <|im_end|> (%u) or <|endoftext|>", key,
                           end->unsignedWord, tokens.turnEnd);
    }
    return std::nullopt;
}

/**
 * @brief Refuses a shape and a tokenizer whose image would pass 32 bits.
 */
[[nodiscard]] std::optional<Refusal> checkImageSize(const Plan &plan, const ConversionOptions &options)
{
    const std::uint64_t bytes =
        infer::layOutModelImage(plan.shape,
                                infer::ModelTokenizerExtents{static_cast<core::u32>(plan.vocabularyText.size()),
                                                             static_cast<core::u32>(plan.merges.size())},
                                options.alignmentLog2, nullptr, 0u);
    if (bytes == 0u || bytes > kLargestImageBytes)
        return refusal("image_too_large", "the image would take %llu bytes, past the 2^32 - 1 ring 0 counts",
                       static_cast<unsigned long long>(bytes));
    return std::nullopt;
}

/**
 * @enum Slot
 * @brief What a tensor of the image is: a matrix of 8-bit blocks, or a vector of Q16.16 gains.
 */
enum class Slot : std::uint8_t {
    Matrix, ///< Blocks of 32.
    Norm,   ///< Gains.
};

/**
 * @struct ExpectedTensor
 * @brief A tensor the image takes: its name, its slot and its extents.
 */
struct ExpectedTensor {
    std::string name;          ///< As the GGUF names it.
    Slot slot{Slot::Norm};     ///< Matrix or norm.
    std::uint32_t rows{1u};    ///< Rows; 1 for a norm.
    std::uint32_t columns{0u}; ///< Columns, or gains.
    bool paired{false};        ///< A query or key matrix, whose rows the rotation pairs.
};

/**
 * @brief The tensors of the image, in the order its segments hold them: the embedding, each layer's attention, its
 *        two norms first, the final norm, then each layer's expert.
 */
[[nodiscard]] std::vector<ExpectedTensor> expectedTensors(const infer::ModelConfig &shape)
{
    std::vector<ExpectedTensor> tensors;
    tensors.push_back({"token_embd.weight", Slot::Matrix, shape.vocabSize, shape.dim, false});
    for (std::uint32_t l = 0u; l < shape.layers; ++l)
    {
        const std::string block = "blk." + std::to_string(l) + ".";
        tensors.push_back({block + "attn_norm.weight", Slot::Norm, 1u, shape.dim, false});
        tensors.push_back({block + "ffn_norm.weight", Slot::Norm, 1u, shape.dim, false});
        tensors.push_back({block + "attn_q.weight", Slot::Matrix, shape.dim, shape.dim, true});
        tensors.push_back({block + "attn_k.weight", Slot::Matrix, shape.keyValueWidth(), shape.dim, true});
        tensors.push_back({block + "attn_v.weight", Slot::Matrix, shape.keyValueWidth(), shape.dim, false});
        tensors.push_back({block + "attn_output.weight", Slot::Matrix, shape.dim, shape.dim, false});
    }
    tensors.push_back({"output_norm.weight", Slot::Norm, 1u, shape.dim, false});
    for (std::uint32_t l = 0u; l < shape.layers; ++l)
    {
        const std::string block = "blk." + std::to_string(l) + ".";
        tensors.push_back({block + "ffn_gate.weight", Slot::Matrix, shape.ffnHidden, shape.dim, false});
        tensors.push_back({block + "ffn_up.weight", Slot::Matrix, shape.ffnHidden, shape.dim, false});
        tensors.push_back({block + "ffn_down.weight", Slot::Matrix, shape.dim, shape.ffnHidden, false});
    }
    return tensors;
}

/**
 * @brief Refuses a tensor the image has no slot for, naming why.
 */
[[nodiscard]] std::optional<Refusal> checkEveryTensorKnown(const SourceModel &source,
                                                           const std::vector<ExpectedTensor> &expected)
{
    std::set<std::string> known;
    for (const ExpectedTensor &tensor : expected)
        known.insert(tensor.name);
    for (const auto &[name, tensor] : source.tensors)
    {
        if (known.count(name) != 0u)
            continue;
        if (name == "output.weight")
            return refusal("output_untied", "output.weight: the image's output head is its embedding, tied");
        if (name == "rope_freqs.weight")
            return refusal("rotary_scaled", "rope_freqs.weight scales the rotation, which ring 0 turns unscaled");
        for (const char *part : kBiasedParts)
        {
            const std::string suffix = std::string(".") + part + ".bias";
            if (name.rfind("blk.", 0u) == 0u && name.size() > suffix.size() &&
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
                return refusal("bias_unsupported", "%s: a v1 image holds no bias", name.c_str());
        }
        return refusal("tensor_unknown", "%s is a tensor the image has no slot for", name.c_str());
    }
    return std::nullopt;
}

/**
 * @brief The bytes @p tensor's @p values take in its type; 0 for a type the converter does not read.
 */
[[nodiscard]] std::uint64_t bytesOf(TensorType type, std::uint64_t values)
{
    switch (type)
    {
    case TensorType::Float32: return 4u * values;
    case TensorType::Float16:
    case TensorType::BrainFloat16: return 2u * values;
    case TensorType::QuantisedEightBit: return values / infer::kQuantBlockLanes * kEightBitBlockBytes;
    case TensorType::Other: break;
    }
    return 0u;
}

/**
 * @brief Refuses a tensor of the image that is missing, of a type or extents it does not take, or whose bytes end past
 *        what the file holds.
 */
[[nodiscard]] std::optional<Refusal> checkTensor(const SourceModel &source, const ExpectedTensor &expected)
{
    const auto found = source.tensors.find(expected.name);
    if (found == source.tensors.end())
        return refusal("tensor_missing", "%s is missing", expected.name.c_str());
    const SourceTensor &tensor = found->second;
    const bool floating = tensor.type == TensorType::Float32 || tensor.type == TensorType::Float16 ||
                          tensor.type == TensorType::BrainFloat16;
    const bool eightBit = tensor.type == TensorType::QuantisedEightBit && expected.slot == Slot::Matrix;
    if (!floating && !eightBit)
        return refusal("tensor_type_unsupported",
                       "%s is of type %s: a matrix takes F32, F16, BF16 or Q8_0, a norm F32, "
                       "F16 or BF16",
                       expected.name.c_str(), tensor.typeName.c_str());
    const std::vector<std::uint64_t> wanted = expected.slot == Slot::Matrix ?
                                                  std::vector<std::uint64_t>{expected.columns, expected.rows} :
                                                  std::vector<std::uint64_t>{expected.columns};
    if (tensor.extents != wanted || (eightBit && expected.columns % infer::kQuantBlockLanes != 0u))
        return refusal("tensor_shape_wrong", "%s is not %u by %u, as the shape gives it", expected.name.c_str(),
                       expected.columns, expected.rows);
    const std::uint64_t needed = bytesOf(tensor.type, static_cast<std::uint64_t>(expected.rows) * expected.columns);
    if (tensor.bytes == nullptr || needed > tensor.byteCount)
        return refusal("tensor_outside_file", "%s needs %llu bytes, past the %llu the file holds for it",
                       expected.name.c_str(), static_cast<unsigned long long>(needed),
                       static_cast<unsigned long long>(tensor.byteCount));
    return std::nullopt;
}

/**
 * @brief The refusal of value @p index of tensor @p name, which roundToFixed refused as @p rounding.
 */
[[nodiscard]] Refusal weightRefusal(const std::string &name, std::uint64_t index, double value, FixedRounding rounding)
{
    if (rounding == FixedRounding::NotFinite)
        return refusal("weight_not_finite", "%s holds %g at %llu", name.c_str(), value,
                       static_cast<unsigned long long>(index));
    return refusal("weight_past_fixed32", "%s holds %g at %llu, past Q16.16's 32767.99998", name.c_str(), value,
                   static_cast<unsigned long long>(index));
}

/**
 * @struct Weights
 * @brief The image's weights, held while the model is assembled and written.
 */
struct Weights {
    std::map<std::string, std::vector<infer::QuantBlock>> matrices; ///< Each matrix's blocks, by its name.
    std::map<std::string, std::vector<math::Fixed32>> norms;        ///< Each norm's gains, by its name.
};

/**
 * @brief Rounds a norm's gains to Q16.16.
 */
[[nodiscard]] std::optional<Refusal> takeNorm(const SourceTensor &tensor, const ExpectedTensor &expected,
                                              Weights &weights)
{
    std::vector<math::Fixed32> &gains = weights.norms[expected.name];
    gains.resize(expected.columns);
    for (std::uint32_t i = 0u; i < expected.columns; ++i)
    {
        std::int32_t raw = 0;
        const double value = sourceValue(tensor, i);
        const FixedRounding rounding = roundToFixed(value, raw);
        if (rounding != FixedRounding::Rounded)
            return weightRefusal(expected.name, i, value, rounding);
        gains[i] = math::Fixed32::fromRaw(raw);
    }
    return std::nullopt;
}

/**
 * @brief Quantises source row @p sourceRow of a float matrix into @p blocks, as ring 0 quantises its activations.
 */
[[nodiscard]] std::optional<Refusal> takeFloatRow(const SourceTensor &tensor, const ExpectedTensor &expected,
                                                  std::uint32_t sourceRow, std::vector<math::Fixed32> &scratch,
                                                  infer::QuantBlock *blocks)
{
    for (std::uint32_t column = 0u; column < expected.columns; ++column)
    {
        const std::uint64_t index = static_cast<std::uint64_t>(sourceRow) * expected.columns + column;
        std::int32_t raw = 0;
        const double value = sourceValue(tensor, index);
        const FixedRounding rounding = roundToFixed(value, raw);
        if (rounding != FixedRounding::Rounded)
            return weightRefusal(expected.name, index, value, rounding);
        scratch[column] = math::Fixed32::fromRaw(raw);
    }
    infer::quantiseRow(scratch.data(), expected.columns, blocks);
    return std::nullopt;
}

/**
 * @brief Takes source row @p sourceRow of a Q8_0 matrix into @p blocks, its lanes kept.
 */
[[nodiscard]] std::optional<Refusal> takeEightBitRow(const SourceTensor &tensor, const ExpectedTensor &expected,
                                                     std::uint32_t sourceRow, infer::QuantBlock *blocks)
{
    const std::uint32_t perRow = infer::quantBlockCount(expected.columns);
    for (std::uint32_t b = 0u; b < perRow; ++b)
    {
        const std::uint64_t block = static_cast<std::uint64_t>(sourceRow) * perRow + b;
        const BlockTaking taking = takeEightBitBlock(tensor.bytes + block * kEightBitBlockBytes, blocks[b]);
        if (taking == BlockTaking::ScaleRefused)
            return refusal("q8_0_scale_refused", "%s's block %llu has a negative, NaN or infinite step",
                           expected.name.c_str(), static_cast<unsigned long long>(block));
        if (taking == BlockTaking::PastRange)
            return refusal("weight_past_fixed32", "%s's block %llu has a step whose magnitude is past Q16.16",
                           expected.name.c_str(), static_cast<unsigned long long>(block));
        if (taking == BlockTaking::LanePastRange)
            return refusal("lane_past_range", "%s's block %llu has a lane of -128", expected.name.c_str(),
                           static_cast<unsigned long long>(block));
    }
    return std::nullopt;
}

/**
 * @brief Requantises a matrix into blocks, its query and key rows read through the pairing.
 */
[[nodiscard]] std::optional<Refusal> takeMatrix(const SourceTensor &tensor, const ExpectedTensor &expected,
                                                const Plan &plan, Weights &weights)
{
    const std::uint32_t perRow = infer::quantBlockCount(expected.columns);
    std::vector<infer::QuantBlock> &blocks = weights.matrices[expected.name];
    blocks.resize(static_cast<std::size_t>(expected.rows) * perRow);
    std::vector<math::Fixed32> scratch(expected.columns);
    const RotaryPairing pairing = expected.paired ? plan.pairing : RotaryPairing::Adjacent;

    for (std::uint32_t row = 0u; row < expected.rows; ++row)
    {
        const std::uint32_t sourceRow = sourceRowOf(row, plan.shape.headDim(), pairing);
        infer::QuantBlock *const out = blocks.data() + static_cast<std::size_t>(row) * perRow;
        std::optional<Refusal> refused = tensor.type == TensorType::QuantisedEightBit ?
                                             takeEightBitRow(tensor, expected, sourceRow, out) :
                                             takeFloatRow(tensor, expected, sourceRow, scratch, out);
        if (refused)
            return refused;
    }
    return std::nullopt;
}

/**
 * @brief A view of matrix @p name of @p weights, @p rows by @p columns.
 */
[[nodiscard]] infer::QuantMatrixView matrixOf(const Weights &weights, const std::string &name, std::uint32_t rows,
                                              std::uint32_t columns)
{
    return infer::QuantMatrixView{weights.matrices.at(name).data(), rows, columns};
}

/**
 * @brief A view of norm @p name of @p weights.
 */
[[nodiscard]] infer::ConstVectorView normOf(const Weights &weights, const std::string &name)
{
    const std::vector<math::Fixed32> &gains = weights.norms.at(name);
    return infer::ConstVectorView{gains.data(), static_cast<core::u32>(gains.size())};
}

/**
 * @brief The layers of @p weights, as ring 0 holds them.
 */
[[nodiscard]] std::vector<infer::LayerWeights> layersOf(const Weights &weights, const infer::ModelConfig &shape)
{
    std::vector<infer::LayerWeights> layers(shape.layers);
    for (std::uint32_t l = 0u; l < shape.layers; ++l)
    {
        const std::string block = "blk." + std::to_string(l) + ".";
        infer::LayerWeights &layer = layers[l];
        layer.attentionNorm = normOf(weights, block + "attn_norm.weight");
        layer.feedForwardNorm = normOf(weights, block + "ffn_norm.weight");
        layer.query = matrixOf(weights, block + "attn_q.weight", shape.dim, shape.dim);
        layer.key = matrixOf(weights, block + "attn_k.weight", shape.keyValueWidth(), shape.dim);
        layer.value = matrixOf(weights, block + "attn_v.weight", shape.keyValueWidth(), shape.dim);
        layer.output = matrixOf(weights, block + "attn_output.weight", shape.dim, shape.dim);
        layer.gate = matrixOf(weights, block + "ffn_gate.weight", shape.ffnHidden, shape.dim);
        layer.up = matrixOf(weights, block + "ffn_up.weight", shape.ffnHidden, shape.dim);
        layer.down = matrixOf(weights, block + "ffn_down.weight", shape.dim, shape.ffnHidden);
    }
    return layers;
}

/**
 * @brief The slot of @p model that holds tensor @p expected: a matrix, or a norm's gains.
 */
[[nodiscard]] infer::QuantMatrixView matrixSlot(const infer::Model &model, const std::string &name)
{
    if (name == "token_embd.weight")
        return model.embedding();
    const std::size_t dot = name.find('.', 4u);
    const infer::LayerWeights &layer = model.layer(static_cast<std::uint32_t>(std::stoul(name.substr(4u, dot - 4u))));
    const std::string part = name.substr(dot + 1u);
    if (part == "attn_q.weight")
        return layer.query;
    if (part == "attn_k.weight")
        return layer.key;
    if (part == "attn_v.weight")
        return layer.value;
    if (part == "attn_output.weight")
        return layer.output;
    if (part == "ffn_gate.weight")
        return layer.gate;
    if (part == "ffn_up.weight")
        return layer.up;
    return layer.down;
}

/**
 * @brief The norm slot of @p model that holds tensor @p name.
 */
[[nodiscard]] infer::ConstVectorView normSlot(const infer::Model &model, const std::string &name)
{
    if (name == "output_norm.weight")
        return model.finalNorm();
    const std::size_t dot = name.find('.', 4u);
    const infer::LayerWeights &layer = model.layer(static_cast<std::uint32_t>(std::stoul(name.substr(4u, dot - 4u))));
    return name.substr(dot + 1u) == "attn_norm.weight" ? layer.attentionNorm : layer.feedForwardNorm;
}

/**
 * @brief How far the slot of @p image that holds @p expected lies from the source tensor, value by value.
 */
[[nodiscard]] ErrorStatistics slotError(const infer::Model &image, const SourceTensor &tensor,
                                        const ExpectedTensor &expected, const Plan &plan)
{
    ErrorStatistics error;
    if (expected.slot == Slot::Norm)
    {
        const infer::ConstVectorView gains = normSlot(image, expected.name);
        for (std::uint32_t i = 0u; i < expected.columns; ++i)
            error.add(static_cast<double>(gains.at(i).raw()) / 65536.0, sourceValue(tensor, i), i);
        return error;
    }
    const infer::QuantMatrixView matrix = matrixSlot(image, expected.name);
    const RotaryPairing pairing = expected.paired ? plan.pairing : RotaryPairing::Adjacent;
    for (std::uint32_t row = 0u; row < expected.rows; ++row)
    {
        const std::uint32_t sourceRow = sourceRowOf(row, plan.shape.headDim(), pairing);
        const infer::QuantBlock *const blocks = matrix.row(row);
        for (std::uint32_t column = 0u; column < expected.columns; ++column)
        {
            const std::uint32_t block = column / infer::kQuantBlockLanes;
            const double source =
                sourceValue(tensor, static_cast<std::uint64_t>(sourceRow) * expected.columns + column);
            error.add(laneValue(blocks[block], column % infer::kQuantBlockLanes), source,
                      static_cast<std::uint64_t>(row) * matrix.blocksPerRow() + block);
        }
    }
    return error;
}

/**
 * @brief A refused conversion.
 */
[[nodiscard]] Conversion refused(Refusal why)
{
    Conversion conversion;
    conversion.refusal = std::move(why);
    return conversion;
}

/**
 * @brief Decides everything the keys decide, in the order the refusals are checked.
 */
[[nodiscard]] std::optional<Refusal> planFromKeys(Keys &keys, const ConversionOptions &options, Plan &plan)
{
    if (std::optional<Refusal> why = findArchitecture(keys))
        return why;
    if (std::optional<Refusal> why = checkEveryKey(keys))
        return why;
    if (std::optional<Refusal> why = checkRequiredKeys(keys))
        return why;
    if (std::optional<Refusal> why = planShape(keys, options, plan))
        return why;
    if (std::optional<Refusal> why = planNormAndRotation(keys, plan))
        return why;
    if (std::optional<Refusal> why = planTokenizer(keys, plan))
        return why;
    if (std::optional<Refusal> why = planVocabulary(keys, plan))
        return why;
    if (std::optional<Refusal> why = planMerges(keys, plan))
        return why;
    if (std::optional<Refusal> why = checkMergesIndex(plan))
        return why;
    if (std::optional<Refusal> why = planTurns(keys, plan))
        return why;
    return checkImageSize(plan, options);
}

/**
 * @brief Takes every tensor of the image from @p source.
 */
[[nodiscard]] std::optional<Refusal> takeTensors(const SourceModel &source, const std::vector<ExpectedTensor> &expected,
                                                 const Plan &plan, Weights &weights)
{
    if (std::optional<Refusal> why = checkEveryTensorKnown(source, expected))
        return why;
    for (const ExpectedTensor &tensor : expected)
    {
        if (std::optional<Refusal> why = checkTensor(source, tensor))
            return why;
    }
    for (const ExpectedTensor &tensor : expected)
    {
        const SourceTensor &held = source.tensors.at(tensor.name);
        std::optional<Refusal> why =
            tensor.slot == Slot::Norm ? takeNorm(held, tensor, weights) : takeMatrix(held, tensor, plan, weights);
        if (why)
            return why;
    }
    return std::nullopt;
}

/**
 * @brief The parts ring 0's model is assembled from.
 */
[[nodiscard]] infer::ModelParts partsOf(const Plan &plan, const Weights &weights,
                                        const std::vector<infer::LayerWeights> &layers)
{
    infer::ModelParts parts;
    parts.shape = plan.shape;
    parts.vocabularyText = plan.vocabularyText.data();
    parts.vocabularyTextBytes = static_cast<core::u32>(plan.vocabularyText.size());
    parts.vocabularyEntries = plan.vocabularyEntries.data();
    parts.merges = plan.merges.data();
    parts.mergeCount = static_cast<core::u32>(plan.merges.size());
    parts.preTokenizer = plan.preTokenizer;
    parts.chatFormat = plan.chatFormat;
    parts.embedding = matrixOf(weights, "token_embd.weight", plan.shape.vocabSize, plan.shape.dim);
    parts.layers = layers.data();
    parts.finalNorm = normOf(weights, "output_norm.weight");
    parts.rotary = infer::RotaryFrequencies{plan.rotary.data(), static_cast<core::u32>(plan.rotary.size())};
    return parts;
}

} // namespace

Conversion convertModel(const SourceModel &source, const ConversionOptions &options)
{
    Keys keys{source.metadata, nullptr, {}};
    Plan plan;
    if (std::optional<Refusal> why = planFromKeys(keys, options, plan))
        return refused(*why);
    const std::vector<ExpectedTensor> expected = expectedTensors(plan.shape);
    Weights weights;
    if (std::optional<Refusal> why = takeTensors(source, expected, plan, weights))
        return refused(*why);

    const std::vector<infer::LayerWeights> layers = layersOf(weights, plan.shape);
    infer::TensorArena assemblyArena{indexArenaBytes(plan)};
    infer::Model assembled;
    const infer::ModelReadReport assembly = assembled.assemble(assemblyArena, partsOf(plan, weights, layers));
    if (assembly.verdict != infer::ModelReadVerdict::Read)
        return refused(refusal("image_self_check_failed", "the parts were refused as %s, entry %u",
                               infer::modelReadVerdictName(assembly.verdict), assembly.segment));

    Conversion conversion;
    conversion.image.resize(assembled.blobBytes(options.alignmentLog2));
    if (conversion.image.empty() ||
        assembled.writeBlob(conversion.image.data(), static_cast<core::u32>(conversion.image.size()),
                            options.alignmentLog2) != conversion.image.size())
        return refused(refusal("image_self_check_failed", "writeBlob wrote no image of the assembled model"));

    infer::TensorArena viewArena{indexArenaBytes(plan) + plan.shape.layers * sizeof(infer::LayerWeights)};
    infer::Model image;
    const infer::ModelReadReport read =
        image.view(viewArena, conversion.image.data(), static_cast<core::u32>(conversion.image.size()));
    if (read.verdict != infer::ModelReadVerdict::Read || image.fold(0x811C9DC5u) != assembled.fold(0x811C9DC5u))
        return refused(refusal("image_self_check_failed", "the image written reads back as %s, or folds otherwise",
                               infer::modelReadVerdictName(read.verdict)));

    for (const ExpectedTensor &tensor : expected)
    {
        const SourceTensor &held = source.tensors.at(tensor.name);
        TensorReport report{tensor.name, held.typeName, tensor.rows, tensor.columns,
                            slotError(image, held, tensor, plan)};
        const double error = report.error.relativeRootMeanSquare();
        if (!(error <= kRequantisationBound))
            return refused(refusal("requantisation_past_bound",
                                   "%s lies %.3e from its source, relative root mean square, past %.0e",
                                   tensor.name.c_str(), error, kRequantisationBound));
        conversion.tensors.push_back(std::move(report));
    }
    conversion.architecture = keys.architecture->name;
    conversion.shape = plan.shape;
    conversion.preTokenizer = plan.preTokenizer;
    conversion.chatFormat = plan.chatFormat;
    conversion.mergeCount = static_cast<std::uint32_t>(plan.merges.size());
    conversion.vocabularyTextBytes = static_cast<std::uint32_t>(plan.vocabularyText.size());
    return conversion;
}

} // namespace lpl::convert
