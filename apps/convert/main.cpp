#include "infer/fixtures/smollm2_byte_pairs.hpp"

#include <lpl/convert/Agreement.hpp>
#include <lpl/convert/Converter.hpp>
#include <lpl/convert/ModelFile.hpp>
#include <lpl/infer/Tokenizer.hpp>

#include <openssl/evp.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kDone = 0;
constexpr int kRefused = 1;
constexpr int kUsage = 2;
constexpr std::uint32_t kDefaultThreads = 4u;
constexpr std::uint32_t kLongRepeats = 3u; ///< The long sequence: the prompts' ids this many times over.
constexpr double kControlBase = 10000.0;   ///< The control table's base, LLaMA's, where SmolLM2's is 100000.
constexpr std::size_t kTokenizerArenaBytes = 64u << 20;
constexpr std::uint32_t kTurnRoom = 1024u;

constexpr const char *kHelp =
    R"(lpl-convert: turns a GGUF model into a .lplmind image of version 1, and compares the two.

Usage:
  lpl-convert convert <model.gguf> <image.lplmind> --context <positions> [--alignment-log2 <12..22>]
  lpl-convert agree <model.gguf> <image.lplmind> [--threads <count>]
  lpl-convert --help

convert
  Reads the GGUF through ggml's gguf API and refuses by name what ring 0's transformer does not execute: a key, a
  tensor, a type, a token, a template or a value it does not know. Writes the image through lpl::infer's writeBlob,
  reads it back, and refuses it when a slot lies past a relative root mean square of 2e-2 from its source tensor.
  --context is required: it sizes the image's cache, 46080 bytes a position for SmolLM2-135M, and may not pass the
  GGUF's own. The records, one per line:
    input path=<file> bytes=<count> sha256=<hex>
    tensor name=<tensor> type=<type> rows=<count> columns=<count> relative_rms=<error> largest_error=<error>
           worst_block=<block>
    image path=<file> bytes=<count> sha256=<hex> architecture=<name> layers=<count> dim=<count> heads=<count>
          key_value_heads=<count> feed_forward=<count> context=<count> vocabulary=<count> merges=<count>
          epsilon_units=<count> pre_tokenizer=<value> chat_format=<value>
    conversion seconds=<seconds> peak_resident_kib=<kibibytes>
  or, when refused, one record: refused name=<name> detail=<what it refuses>

agree
  Runs the image in lpl::infer and the GGUF in llama.cpp b3775 on the same ids: SmolLM2-135M-Instruct's 18 prompts and
  6 chat turns of tests/infer/fixtures/smollm2_byte_pairs.hpp, and the 18 prompts' ids three times over, 1248
  positions. A position is judged when llama.cpp's first logit leads its second by at least 1.0. Then runs two
  controls assembled from the image's own parts, its keys and values swapped and its rotary table of base 10000,
  which must each disagree at a judged position. The records:
    tokens kind=<prompt|turn> index=<i> ours=<match|differ> llama_cpp=<match|differ>
    sequence kind=<prompt|turn|long> index=<i> positions=<n> judged=<n> disagreements=<n> agreements=<n>
             last_judged=<0|1> last_agrees=<0|1> largest_logit_difference=<x> reference_residual_peak=<x>
             image_residual_peak=<x> image_saturations=<n> forward_matches=<0|1>
    total ... ; control name=<name> judged=<n> disagreements=<n> largest_logit_difference=<x>
    verdict agrees=<0|1> controls_fail=<0|1>

Exit status: 0 converted, or agreeing with both controls failing; 1 refused, or not; 2 a usage error.
)";

/**
 * @brief The SHA-256 of the @p size bytes at @p bytes, in hexadecimal.
 */
[[nodiscard]] std::string sha256Of(const std::uint8_t *bytes, std::size_t size)
{
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0u;
    EVP_Digest(bytes, size, digest, &length, EVP_sha256(), nullptr);
    std::string hex;
    char pair[3];
    for (unsigned int i = 0u; i < length; ++i)
    {
        std::snprintf(pair, sizeof(pair), "%02x", digest[i]);
        hex += pair;
    }
    return hex;
}

/**
 * @brief This process's peak resident memory, in KiB.
 */
[[nodiscard]] long peakResidentKibibytes()
{
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

/**
 * @brief The value of option @p name among @p arguments, or nullptr when it is not given.
 */
[[nodiscard]] const char *optionOf(const std::vector<std::string> &arguments, const char *name)
{
    for (std::size_t i = 0u; i + 1u < arguments.size(); ++i)
    {
        if (arguments[i] == name)
            return arguments[i + 1u].c_str();
    }
    return nullptr;
}

/**
 * @brief @p text as a positive count, or 0 when it is not one.
 */
[[nodiscard]] std::uint32_t countOf(const char *text)
{
    if (text == nullptr)
        return 0u;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return end != text && *end == '\0' && value <= 0xFFFFFFFFul ? static_cast<std::uint32_t>(value) : 0u;
}

/**
 * @brief Prints @p refusal as its record, and returns the status of a refusal.
 */
int printRefusal(const lpl::convert::Refusal &refusal)
{
    std::printf("refused name=%s detail=%s\n", refusal.name.c_str(), refusal.detail.c_str());
    return kRefused;
}

/**
 * @brief Writes @p image as the file at @p path.
 */
[[nodiscard]] bool writeImage(const std::string &path, const std::vector<std::uint8_t> &image)
{
    std::FILE *const file = std::fopen(path.c_str(), "wb");
    if (file == nullptr)
        return false;
    const bool written = std::fwrite(image.data(), 1u, image.size(), file) == image.size();
    return std::fclose(file) == 0 && written;
}

/**
 * @brief Reads the file at @p path whole into @p bytes, whose storage is aligned for Model::view.
 */
[[nodiscard]] bool readImage(const std::string &path, std::vector<std::uint8_t> &bytes)
{
    std::FILE *const file = std::fopen(path.c_str(), "rb");
    if (file == nullptr)
        return false;
    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    bytes.resize(size > 0 ? static_cast<std::size_t>(size) : 0u);
    const bool read = size > 0 && std::fread(bytes.data(), 1u, bytes.size(), file) == bytes.size();
    std::fclose(file);
    return read;
}

/**
 * @brief Prints the report of each slot of @p conversion.
 */
void printTensors(const lpl::convert::Conversion &conversion)
{
    for (const lpl::convert::TensorReport &tensor : conversion.tensors)
        std::printf(
            "tensor name=%s type=%s rows=%llu columns=%llu relative_rms=%.4e largest_error=%.4e worst_block=%llu\n",
            tensor.name.c_str(), tensor.type.c_str(), static_cast<unsigned long long>(tensor.rows),
            static_cast<unsigned long long>(tensor.columns), tensor.error.relativeRootMeanSquare(),
            tensor.error.largestError, static_cast<unsigned long long>(tensor.error.worstBlock));
}

/**
 * @brief Runs `convert`.
 */
int convert(const std::vector<std::string> &arguments)
{
    if (arguments.size() < 3u || optionOf(arguments, "--context") == nullptr)
    {
        std::fputs("lpl-convert convert: needs <model.gguf> <image.lplmind> --context <positions>; see --help\n",
                   stderr);
        return kUsage;
    }
    const auto started = std::chrono::steady_clock::now();
    lpl::convert::ConversionOptions options;
    options.contextLength = countOf(optionOf(arguments, "--context"));
    if (const char *alignment = optionOf(arguments, "--alignment-log2"))
        options.alignmentLog2 = countOf(alignment);

    lpl::convert::ModelFile file;
    if (std::optional<lpl::convert::Refusal> refusal = file.open(arguments[1]))
        return printRefusal(*refusal);
    std::printf("input path=%s bytes=%zu sha256=%s\n", arguments[1].c_str(), file.size(),
                sha256Of(file.bytes(), file.size()).c_str());
    const lpl::convert::Conversion conversion = lpl::convert::convertModel(file.source(), options);
    if (conversion.refusal.has_value())
        return printRefusal(*conversion.refusal);
    if (!writeImage(arguments[2], conversion.image))
        return printRefusal(lpl::convert::Refusal{"image_unwritable", arguments[2] + ": the image cannot be written"});

    printTensors(conversion);
    const lpl::infer::ModelConfig &shape = conversion.shape;
    std::printf("image path=%s bytes=%zu sha256=%s architecture=%s layers=%u dim=%u heads=%u key_value_heads=%u "
                "feed_forward=%u context=%u vocabulary=%u merges=%u epsilon_units=%u pre_tokenizer=%u chat_format=%u\n",
                arguments[2].c_str(), conversion.image.size(),
                sha256Of(conversion.image.data(), conversion.image.size()).c_str(), conversion.architecture.c_str(),
                shape.layers, shape.dim, shape.heads, shape.keyValueHeads, shape.ffnHidden, shape.contextLength,
                shape.vocabSize, conversion.mergeCount, shape.normEpsilon,
                static_cast<unsigned>(conversion.preTokenizer), static_cast<unsigned>(conversion.chatFormat));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf("conversion seconds=%.2f peak_resident_kib=%ld\n", seconds, peakResidentKibibytes());
    return kDone;
}

/**
 * @struct Sequence
 * @brief One sequence both engines run: its ids, and what it is.
 */
struct Sequence {
    const char *kind;                  ///< prompt, turn or long.
    std::uint32_t index;               ///< Its place among its kind.
    std::vector<std::uint32_t> tokens; ///< The ids both engines are fed.
};

/**
 * @brief The fixture's ids from @p first, @p count of them, out of @p ids.
 */
[[nodiscard]] std::vector<std::uint32_t> idsOf(const lpl::core::u32 *ids, std::uint32_t first, std::uint32_t count)
{
    return std::vector<std::uint32_t>(ids + first, ids + first + count);
}

/**
 * @brief The sequences of SmolLM2's fixture: each prompt's ids, each turn's, and the prompts' ids three times over.
 */
[[nodiscard]] std::vector<Sequence> fixtureSequences()
{
    std::vector<Sequence> sequences;
    std::vector<std::uint32_t> prompts;
    for (std::uint32_t i = 0u; i < smollm2_byte_pairs::kPromptCount; ++i)
    {
        const smollm2_byte_pairs::Prompt &prompt = smollm2_byte_pairs::kPrompts[i];
        sequences.push_back({"prompt", i, idsOf(smollm2_byte_pairs::kReferenceIds, prompt.firstId, prompt.idCount)});
        prompts.insert(prompts.end(), sequences.back().tokens.begin(), sequences.back().tokens.end());
    }
    for (std::uint32_t i = 0u; i < smollm2_byte_pairs::kChatPairCount; ++i)
    {
        const smollm2_byte_pairs::ChatPair &pair = smollm2_byte_pairs::kChatPairs[i];
        sequences.push_back({"turn", i, idsOf(smollm2_byte_pairs::kChatTurnIds, pair.firstTurnId, pair.turnIdCount)});
    }
    std::vector<std::uint32_t> repeated;
    for (std::uint32_t r = 0u; r < kLongRepeats; ++r)
        repeated.insert(repeated.end(), prompts.begin(), prompts.end());
    sequences.push_back({"long", 0u, repeated});
    return sequences;
}

/**
 * @brief SmolLM2's chat template rendered for chat pair @p pair, as llama.cpp is given it to split.
 */
[[nodiscard]] std::string renderedTurn(const smollm2_byte_pairs::ChatPair &pair)
{
    const std::string system(smollm2_byte_pairs::kChatText + pair.systemOffset, pair.systemBytes);
    const std::string user(smollm2_byte_pairs::kChatText + pair.userOffset, pair.userBytes);
    return "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user +
           "<|im_end|>\n<|im_start|>assistant\n";
}

/**
 * @brief Whether lpl::infer, bound to @p image's tokenizer, and llama.cpp give the fixture's ids, printing one record
 *        per prompt and turn.
 * @return Whether lpl::infer gave the fixture's ids every time; llama.cpp's are printed, not judged.
 */
[[nodiscard]] bool checkTokens(const lpl::infer::Model &image, lpl::convert::ReferenceModel &reference)
{
    lpl::infer::TensorArena arena{kTokenizerArenaBytes};
    lpl::infer::Tokenizer tokenizer;
    if (!tokenizer.bind(arena, image.vocab(), image.merges(), image.preTokenizer()))
        return false;
    bool every = true;
    std::vector<lpl::core::u32> ids(kTurnRoom);
    for (std::uint32_t i = 0u; i < smollm2_byte_pairs::kPromptCount; ++i)
    {
        const smollm2_byte_pairs::Prompt &prompt = smollm2_byte_pairs::kPrompts[i];
        const std::vector<std::uint32_t> expected =
            idsOf(smollm2_byte_pairs::kReferenceIds, prompt.firstId, prompt.idCount);
        lpl::core::u32 skipped = 0u;
        const lpl::core::u32 count = tokenizer.encode(smollm2_byte_pairs::kPromptText + prompt.textOffset,
                                                      prompt.textBytes, ids.data(), kTurnRoom, skipped);
        const bool ours = std::vector<std::uint32_t>(ids.begin(), ids.begin() + count) == expected;
        const std::string text(smollm2_byte_pairs::kPromptText + prompt.textOffset, prompt.textBytes);
        const bool theirs = reference.tokenize(text, false) == expected;
        std::printf("tokens kind=prompt index=%u ours=%s llama_cpp=%s\n", i, ours ? "match" : "differ",
                    theirs ? "match" : "differ");
        every = every && ours;
    }
    for (std::uint32_t i = 0u; i < smollm2_byte_pairs::kChatPairCount; ++i)
    {
        const smollm2_byte_pairs::ChatPair &pair = smollm2_byte_pairs::kChatPairs[i];
        lpl::infer::TensorArena scratch{kTokenizerArenaBytes / 8u};
        const lpl::infer::ChatTurnReport report = lpl::infer::encodeChatTurn(
            scratch, tokenizer, image.chatTokens(), smollm2_byte_pairs::kChatText + pair.systemOffset, pair.systemBytes,
            smollm2_byte_pairs::kChatText + pair.userOffset, pair.userBytes, ids.data(), kTurnRoom);
        const bool ours = report.verdict == lpl::infer::ChatTurnVerdict::Encoded &&
                          std::vector<std::uint32_t>(ids.begin(), ids.begin() + report.tokens) ==
                              idsOf(smollm2_byte_pairs::kChatTurnIds, pair.firstTurnId, pair.turnIdCount);
        const bool theirs = reference.tokenize(renderedTurn(pair), true) ==
                            idsOf(smollm2_byte_pairs::kTemplateIds, pair.firstTemplateId, pair.templateIdCount);
        std::printf("tokens kind=turn index=%u ours=%s llama_cpp=%s\n", i, ours ? "match" : "differ",
                    theirs ? "match" : "differ");
        every = every && ours;
    }
    return every;
}

/**
 * @struct Totals
 * @brief The sum of what the sequences found.
 */
struct Totals {
    std::uint32_t sequences{0u};        ///< Sequences run.
    std::uint64_t positions{0u};        ///< Positions run.
    std::uint64_t judged{0u};           ///< Positions judged.
    std::uint64_t disagreements{0u};    ///< Judged positions that disagree.
    std::uint64_t agreements{0u};       ///< Positions, judged or not, that agree.
    double largestLogitDifference{0.0}; ///< Over every position.
    double referenceResidualPeak{0.0};  ///< llama.cpp's.
    double imageResidualPeak{0.0};      ///< lpl::infer's.
    std::uint64_t imageSaturations{0u}; ///< lpl::infer's.
    bool ran{true};                     ///< Every sequence ran.
    bool forwardMatches{true};          ///< Every sequence's layer-by-layer logits were Transformer::forward's.
};

/**
 * @brief Adds @p agreement to @p totals.
 */
void add(Totals &totals, const lpl::convert::SequenceAgreement &agreement)
{
    ++totals.sequences;
    totals.positions += agreement.positions.size();
    totals.judged += agreement.judged();
    totals.disagreements += agreement.judgedDisagreements();
    totals.agreements += agreement.agreements();
    for (const lpl::convert::PositionAgreement &position : agreement.positions)
        totals.largestLogitDifference = std::max(totals.largestLogitDifference, position.largestLogitDifference);
    totals.referenceResidualPeak = std::max(totals.referenceResidualPeak, agreement.referenceResidualPeak);
    totals.imageResidualPeak = std::max(totals.imageResidualPeak, agreement.imageResidualPeak);
    totals.imageSaturations += agreement.imageSaturations;
    totals.forwardMatches = totals.forwardMatches && agreement.imageMatchesForward;
}

/**
 * @brief Prints the record of @p sequence's @p agreement.
 */
void printSequence(const Sequence &sequence, const lpl::convert::SequenceAgreement &agreement)
{
    const lpl::convert::PositionAgreement &last = agreement.positions.back();
    double largest = 0.0;
    for (const lpl::convert::PositionAgreement &position : agreement.positions)
        largest = std::max(largest, position.largestLogitDifference);
    std::printf("sequence kind=%s index=%u positions=%zu judged=%u disagreements=%u agreements=%u last_judged=%d "
                "last_agrees=%d largest_logit_difference=%.4f reference_residual_peak=%.1f image_residual_peak=%.1f "
                "image_saturations=%llu forward_matches=%d\n",
                sequence.kind, sequence.index, agreement.positions.size(), agreement.judged(),
                agreement.judgedDisagreements(), agreement.agreements(),
                last.referenceMargin >= lpl::convert::kAgreementMargin ? 1 : 0,
                last.imageToken == last.referenceToken ? 1 : 0, largest, agreement.referenceResidualPeak,
                agreement.imageResidualPeak, static_cast<unsigned long long>(agreement.imageSaturations),
                agreement.imageMatchesForward ? 1 : 0);
    std::fflush(stdout);
}

/**
 * @brief Runs every sequence through @p model and @p reference, printing a record for each when @p print.
 */
[[nodiscard]] Totals runSequences(lpl::convert::ReferenceModel &reference, const lpl::infer::Model &model,
                                  const std::vector<Sequence> &sequences, bool print)
{
    Totals totals;
    for (const Sequence &sequence : sequences)
    {
        lpl::convert::SequenceAgreement agreement;
        if (!lpl::convert::compareSequence(reference, model, sequence.tokens, agreement))
        {
            totals.ran = false;
            std::printf("sequence kind=%s index=%u ran=0\n", sequence.kind, sequence.index);
            continue;
        }
        add(totals, agreement);
        if (print)
            printSequence(sequence, agreement);
    }
    return totals;
}

/**
 * @brief Runs control @p control of @p image on the short sequences, printing its record.
 * @return Whether it disagreed at a judged position, as a control must.
 */
[[nodiscard]] bool runControl(lpl::convert::ReferenceModel &reference, const lpl::infer::Model &image,
                              lpl::convert::Control control, const char *name, const std::vector<Sequence> &sequences)
{
    lpl::convert::ControlModel made;
    if (!lpl::convert::assembleControl(image, control, kControlBase, made))
    {
        std::printf("control name=%s assembled=0\n", name);
        return false;
    }
    std::vector<Sequence> shorter(sequences.begin(), sequences.end() - 1);
    const Totals totals = runSequences(reference, made.model, shorter, false);
    std::printf("control name=%s judged=%llu disagreements=%llu largest_logit_difference=%.4f\n", name,
                static_cast<unsigned long long>(totals.judged), static_cast<unsigned long long>(totals.disagreements),
                totals.largestLogitDifference);
    return totals.ran && totals.disagreements > 0u;
}

/**
 * @brief Runs `agree`.
 */
int agree(const std::vector<std::string> &arguments)
{
    if (arguments.size() < 3u)
    {
        std::fputs("lpl-convert agree: needs <model.gguf> <image.lplmind>; see --help\n", stderr);
        return kUsage;
    }
    const std::uint32_t threads =
        optionOf(arguments, "--threads") != nullptr ? countOf(optionOf(arguments, "--threads")) : kDefaultThreads;
    lpl::convert::ReferenceModel reference;
    if (std::optional<lpl::convert::Refusal> refusal = reference.open(arguments[1], threads == 0u ? 1u : threads))
        return printRefusal(*refusal);
    std::vector<std::uint8_t> bytes;
    if (!readImage(arguments[2], bytes))
        return printRefusal(lpl::convert::Refusal{"file_unreadable", arguments[2] + ": the image cannot be read"});
    lpl::infer::TensorArena arena{kTokenizerArenaBytes};
    lpl::infer::Model image;
    const lpl::infer::ModelReadReport read = image.view(arena, bytes.data(), static_cast<lpl::core::u32>(bytes.size()));
    if (read.verdict != lpl::infer::ModelReadVerdict::Read)
        return printRefusal(lpl::convert::Refusal{"image_unreadable", lpl::infer::modelReadVerdictName(read.verdict)});
    if (image.config().vocabSize != reference.vocabularySize() || image.chatTokens().turnEnd != 2u)
        return printRefusal(
            lpl::convert::Refusal{"sequences_not_this_model", "the sequences are SmolLM2-135M-Instruct's ids"});

    std::printf("settings threads=%u margin=%.1f keys=f32 values=f16 decode=one_per_sequence\n", reference.threads(),
                lpl::convert::kAgreementMargin);
    const bool tokens = checkTokens(image, reference);
    const std::vector<Sequence> sequences = fixtureSequences();
    const Totals totals = runSequences(reference, image, sequences, true);
    std::printf("total sequences=%u positions=%llu judged=%llu disagreements=%llu agreements=%llu "
                "largest_logit_difference=%.4f reference_residual_peak=%.1f image_residual_peak=%.1f "
                "image_saturations=%llu forward_matches=%d tokens_match=%d\n",
                totals.sequences, static_cast<unsigned long long>(totals.positions),
                static_cast<unsigned long long>(totals.judged), static_cast<unsigned long long>(totals.disagreements),
                static_cast<unsigned long long>(totals.agreements), totals.largestLogitDifference,
                totals.referenceResidualPeak, totals.imageResidualPeak,
                static_cast<unsigned long long>(totals.imageSaturations), totals.forwardMatches ? 1 : 0,
                tokens ? 1 : 0);
    const bool swapped =
        runControl(reference, image, lpl::convert::Control::KeyValueSwapped, "key_value_swapped", sequences);
    const bool rotated =
        runControl(reference, image, lpl::convert::Control::RotaryBase, "rotary_base_10000", sequences);
    const bool agrees =
        totals.ran && totals.forwardMatches && tokens && totals.judged > 0u && totals.disagreements == 0u;
    std::printf("verdict agrees=%d controls_fail=%d\n", agrees ? 1 : 0, swapped && rotated ? 1 : 0);
    return agrees && swapped && rotated ? kDone : kRefused;
}

} // namespace

int main(int argc, char **argv)
{
    const std::vector<std::string> arguments(argv + 1, argv + argc);
    if (arguments.empty() || arguments[0] == "--help" || arguments[0] == "-h")
    {
        std::fputs(kHelp, arguments.empty() ? stderr : stdout);
        return arguments.empty() ? kUsage : kDone;
    }
    if (arguments[0] == "convert")
        return convert(arguments);
    if (arguments[0] == "agree")
        return agree(arguments);
    std::fprintf(stderr, "lpl-convert: no command %s; see --help\n", arguments[0].c_str());
    return kUsage;
}
