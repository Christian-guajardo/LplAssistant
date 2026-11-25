#include "infer/fixtures/smollm2_byte_pairs.hpp"

#include <lpl/convert/Agreement.hpp>
#include <lpl/convert/Converter.hpp>
#include <lpl/convert/ModelFile.hpp>
#include <lpl/infer/Tokenizer.hpp>

#include <openssl/evp.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
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
  lpl-convert agree <model.gguf> <image.lplmind> [--threads <count>] [--positions]
  lpl-convert --help

Both commands read the GGUF through ggml b3775's gguf parser, which stops the process on some malformed files rather
than refusing them: give them a GGUF of a chain you trust, such as tools/gguf_reference.py's for SmolLM2.

convert
  Refuses by name what ring 0's transformer does not execute: a key, a tensor, a type, a token, a template or a value
  it does not know, a key written twice. Writes the image through lpl::infer's writeBlob, reads it back, and refuses
  it when a slot lies past a relative root mean square of 2e-2 from its source tensor. The names of the refusals are
  those convert/include/lpl/convert/Converter.hpp and ModelFile.hpp list. --context is required: it sizes the image's
  cache, 46080 bytes a position for SmolLM2-135M, and may not pass the GGUF's own. The records, one per line:
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
  positions, which the image's context must hold. A position is judged when llama.cpp's first logit leads its second
  by at least 1.0. Then runs, on the 24 short sequences, two controls assembled from the image's own parts, its keys
  and values swapped and its rotary table of base 10000, which must each disagree at a judged position and lie
  further from llama.cpp than the image. The records, one per line:
    settings threads=<count> margin=<logits> keys=f32 values=f16 decode=one_per_sequence
    tokens kind=<prompt|turn> index=<i> ours=<match|differ> llama_cpp=<match|differ>
    sequence kind=<prompt|turn|long> index=<i> positions=<n> judged=<n> disagreements=<n> agreements=<n>
             last_judged=<0|1> last_agrees=<0|1> largest_logit_difference=<x> reference_residual_peak=<x>
             image_residual_peak=<x> image_saturations=<n> forward_matches=<0|1>
      or, for a sequence an engine could not run: sequence kind=<kind> index=<i> ran=0
    position kind=<kind> index=<i> at=<position> image_token=<id> reference_token=<id> margin=<logits>
             largest_logit_difference=<x>, with --positions only
    total sequences=<n> positions=<n> judged=<n> disagreements=<n> agreements=<n> largest_logit_difference=<x>
          reference_residual_peak=<x> image_residual_peak=<x> image_saturations=<n> forward_matches=<0|1>
          tokens_match=<0|1>
    control name=<key_value_swapped|rotary_base_10000> judged=<n> disagreements=<n> largest_logit_difference=<x>
      or, when its parts cannot be assembled: control name=<name> assembled=0
    verdict agrees=<0|1> controls_fail=<0|1>
  or, when refused before any sequence, one record: refused name=<name> detail=<what it refuses>

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
 * @brief Whether @p arguments hold the flag @p name.
 */
[[nodiscard]] bool flagOf(const std::vector<std::string> &arguments, const char *name)
{
    return std::find(arguments.begin(), arguments.end(), name) != arguments.end();
}

/**
 * @brief @p text as a count of 32 bits, or nothing when it is not one.
 */
[[nodiscard]] std::optional<std::uint32_t> countOf(const char *text)
{
    char *end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || errno != 0 || value > 0xFFFFFFFFul || text[0] == '-')
        return std::nullopt;
    return static_cast<std::uint32_t>(value);
}

/**
 * @brief The count option @p name gives, @p fallback when it is not given; prints the usage error and gives nothing
 *        when it is given but is not a count.
 */
[[nodiscard]] std::optional<std::uint32_t> countOption(const std::vector<std::string> &arguments, const char *name,
                                                       std::uint32_t fallback)
{
    const char *const text = optionOf(arguments, name);
    if (text == nullptr)
        return fallback;
    const std::optional<std::uint32_t> count = countOf(text);
    if (!count.has_value())
        std::fprintf(stderr, "lpl-convert: %s %s is not a count; see --help\n", name, text);
    return count;
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
 * @brief Reads the image at @p path whole into @p bytes, whose storage is aligned for Model::view.
 * @return Nothing when read; `file_missing` when no file is there, `file_unreadable` when it cannot be read.
 */
[[nodiscard]] std::optional<lpl::convert::Refusal> readImage(const std::string &path, std::vector<std::uint8_t> &bytes)
{
    struct stat status {};
    if (stat(path.c_str(), &status) != 0)
        return lpl::convert::Refusal{errno == ENOENT ? "file_missing" : "file_unreadable",
                                     path + ": " + std::strerror(errno)};
    std::FILE *const file = std::fopen(path.c_str(), "rb");
    if (!S_ISREG(status.st_mode) || status.st_size == 0 || file == nullptr)
    {
        if (file != nullptr)
            std::fclose(file);
        return lpl::convert::Refusal{"file_unreadable", path + ": not a regular file of at least one byte"};
    }
    bytes.resize(static_cast<std::size_t>(status.st_size));
    const bool read = std::fread(bytes.data(), 1u, bytes.size(), file) == bytes.size();
    std::fclose(file);
    if (!read)
        return lpl::convert::Refusal{"file_unreadable", path + ": it cannot be read whole"};
    return std::nullopt;
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
    const char *const context = optionOf(arguments, "--context");
    if (arguments.size() < 3u || context == nullptr)
    {
        std::fputs("lpl-convert convert: needs <model.gguf> <image.lplmind> --context <positions>; see --help\n",
                   stderr);
        return kUsage;
    }
    const std::optional<std::uint32_t> positions = countOption(arguments, "--context", 0u);
    const std::optional<std::uint32_t> alignment =
        countOption(arguments, "--alignment-log2", lpl::infer::kModelImageDefaultAlignmentLog2);
    if (!positions.has_value() || !alignment.has_value())
        return kUsage;

    const auto started = std::chrono::steady_clock::now();
    lpl::convert::ConversionOptions options;
    options.contextLength = *positions;
    options.alignmentLog2 = *alignment;
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
 * @brief The short sequences of SmolLM2's fixture: each prompt's ids, then each turn's.
 */
[[nodiscard]] std::vector<Sequence> shortSequences()
{
    std::vector<Sequence> sequences;
    for (std::uint32_t i = 0u; i < smollm2_byte_pairs::kPromptCount; ++i)
    {
        const smollm2_byte_pairs::Prompt &prompt = smollm2_byte_pairs::kPrompts[i];
        sequences.push_back({"prompt", i, idsOf(smollm2_byte_pairs::kReferenceIds, prompt.firstId, prompt.idCount)});
    }
    for (std::uint32_t i = 0u; i < smollm2_byte_pairs::kChatPairCount; ++i)
    {
        const smollm2_byte_pairs::ChatPair &pair = smollm2_byte_pairs::kChatPairs[i];
        sequences.push_back({"turn", i, idsOf(smollm2_byte_pairs::kChatTurnIds, pair.firstTurnId, pair.turnIdCount)});
    }
    return sequences;
}

/**
 * @brief The long sequence: the prompts' ids, one prompt after the other, kLongRepeats times over.
 */
[[nodiscard]] Sequence longSequence(const std::vector<Sequence> &shorter)
{
    Sequence repeated{"long", 0u, {}};
    for (std::uint32_t r = 0u; r < kLongRepeats; ++r)
    {
        for (const Sequence &sequence : shorter)
        {
            if (std::strcmp(sequence.kind, "prompt") == 0)
                repeated.tokens.insert(repeated.tokens.end(), sequence.tokens.begin(), sequence.tokens.end());
        }
    }
    return repeated;
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
 * @brief The largest |difference| of a logit over @p agreement's positions.
 */
[[nodiscard]] double largestDifferenceOf(const lpl::convert::SequenceAgreement &agreement)
{
    double largest = 0.0;
    for (const lpl::convert::PositionAgreement &position : agreement.positions)
        largest = std::max(largest, position.largestLogitDifference);
    return largest;
}

/**
 * @brief Prints the record of @p sequence's @p agreement, and one per position when @p positions.
 */
void printSequence(const Sequence &sequence, const lpl::convert::SequenceAgreement &agreement, bool positions)
{
    const lpl::convert::PositionAgreement &last = agreement.positions.back();
    std::printf("sequence kind=%s index=%u positions=%zu judged=%u disagreements=%u agreements=%u last_judged=%d "
                "last_agrees=%d largest_logit_difference=%.4f reference_residual_peak=%.1f image_residual_peak=%.1f "
                "image_saturations=%llu forward_matches=%d\n",
                sequence.kind, sequence.index, agreement.positions.size(), agreement.judged(),
                agreement.judgedDisagreements(), agreement.agreements(),
                last.referenceMargin >= lpl::convert::kAgreementMargin ? 1 : 0,
                last.imageToken == last.referenceToken ? 1 : 0, largestDifferenceOf(agreement),
                agreement.referenceResidualPeak, agreement.imageResidualPeak,
                static_cast<unsigned long long>(agreement.imageSaturations), agreement.imageMatchesForward ? 1 : 0);
    for (std::size_t at = 0u; positions && at < agreement.positions.size(); ++at)
    {
        const lpl::convert::PositionAgreement &position = agreement.positions[at];
        std::printf("position kind=%s index=%u at=%zu image_token=%u reference_token=%u margin=%.4f "
                    "largest_logit_difference=%.4f\n",
                    sequence.kind, sequence.index, at, position.imageToken, position.referenceToken,
                    position.referenceMargin, position.largestLogitDifference);
    }
    std::fflush(stdout);
}

/**
 * @brief Runs every sequence through @p model and @p reference, printing a record for each when @p print.
 * @return One comparison per sequence; one without positions for a sequence an engine could not run.
 */
[[nodiscard]] std::vector<lpl::convert::SequenceAgreement> runSequences(lpl::convert::ReferenceModel &reference,
                                                                        const lpl::infer::Model &model,
                                                                        const std::vector<Sequence> &sequences,
                                                                        bool print, bool positions)
{
    std::vector<lpl::convert::SequenceAgreement> agreements;
    for (const Sequence &sequence : sequences)
    {
        lpl::convert::SequenceAgreement agreement;
        if (!lpl::convert::compareSequence(reference, model, sequence.tokens, agreement))
        {
            agreement.positions.clear();
            std::printf("sequence kind=%s index=%u ran=0\n", sequence.kind, sequence.index);
        }
        else if (print)
            printSequence(sequence, agreement, positions);
        agreements.push_back(std::move(agreement));
    }
    return agreements;
}

/**
 * @brief Prints the total of @p agreements.
 */
void printTotal(const std::vector<lpl::convert::SequenceAgreement> &agreements, bool tokens)
{
    std::uint64_t positions = 0u, judged = 0u, disagreements = 0u, matching = 0u, saturations = 0u;
    double largest = 0.0, referencePeak = 0.0, imagePeak = 0.0;
    bool forward = true;
    for (const lpl::convert::SequenceAgreement &agreement : agreements)
    {
        positions += agreement.positions.size();
        judged += agreement.judged();
        disagreements += agreement.judgedDisagreements();
        matching += agreement.agreements();
        saturations += agreement.imageSaturations;
        largest = std::max(largest, largestDifferenceOf(agreement));
        referencePeak = std::max(referencePeak, agreement.referenceResidualPeak);
        imagePeak = std::max(imagePeak, agreement.imageResidualPeak);
        forward = forward && agreement.imageMatchesForward;
    }
    std::printf("total sequences=%zu positions=%llu judged=%llu disagreements=%llu agreements=%llu "
                "largest_logit_difference=%.4f reference_residual_peak=%.1f image_residual_peak=%.1f "
                "image_saturations=%llu forward_matches=%d tokens_match=%d\n",
                agreements.size(), static_cast<unsigned long long>(positions), static_cast<unsigned long long>(judged),
                static_cast<unsigned long long>(disagreements), static_cast<unsigned long long>(matching), largest,
                referencePeak, imagePeak, static_cast<unsigned long long>(saturations), forward ? 1 : 0,
                tokens ? 1 : 0);
}

/**
 * @brief Runs control @p control of @p image on the short sequences, printing its record.
 * @return Whether it fails as a control must, against @p imageShort, the image's comparisons on the same sequences.
 */
[[nodiscard]] bool runControl(lpl::convert::ReferenceModel &reference, const lpl::infer::Model &image,
                              lpl::convert::Control control, const char *name, const std::vector<Sequence> &shorter,
                              const std::vector<lpl::convert::SequenceAgreement> &imageShort)
{
    lpl::convert::ControlModel made;
    if (!lpl::convert::assembleControl(image, control, kControlBase, made))
    {
        std::printf("control name=%s assembled=0\n", name);
        return false;
    }
    const std::vector<lpl::convert::SequenceAgreement> agreements =
        runSequences(reference, made.model, shorter, false, false);
    std::uint64_t judged = 0u, disagreements = 0u;
    double largest = 0.0;
    for (const lpl::convert::SequenceAgreement &agreement : agreements)
    {
        judged += agreement.judged();
        disagreements += agreement.judgedDisagreements();
        largest = std::max(largest, largestDifferenceOf(agreement));
    }
    std::printf("control name=%s judged=%llu disagreements=%llu largest_logit_difference=%.4f\n", name,
                static_cast<unsigned long long>(judged), static_cast<unsigned long long>(disagreements), largest);
    return lpl::convert::controlFails(agreements, imageShort);
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
    const std::optional<std::uint32_t> threads = countOption(arguments, "--threads", kDefaultThreads);
    if (!threads.has_value() || *threads == 0u)
    {
        if (threads.has_value())
            std::fputs("lpl-convert: --threads 0 runs nothing; see --help\n", stderr);
        return kUsage;
    }
    std::vector<std::uint8_t> bytes;
    if (std::optional<lpl::convert::Refusal> refusal = readImage(arguments[2], bytes))
        return printRefusal(*refusal);
    lpl::convert::ReferenceModel reference;
    if (std::optional<lpl::convert::Refusal> refusal = reference.open(arguments[1], *threads))
        return printRefusal(*refusal);
    lpl::infer::TensorArena arena{kTokenizerArenaBytes};
    lpl::infer::Model image;
    const lpl::infer::ModelReadReport read = image.view(arena, bytes.data(), static_cast<lpl::core::u32>(bytes.size()));
    if (read.verdict != lpl::infer::ModelReadVerdict::Read)
        return printRefusal(lpl::convert::Refusal{"image_unreadable", lpl::infer::modelReadVerdictName(read.verdict)});
    if (image.config().vocabSize != reference.vocabularySize() || image.chatTokens().turnEnd != 2u)
        return printRefusal(
            lpl::convert::Refusal{"sequences_not_this_model", "the sequences are SmolLM2-135M-Instruct's ids"});
    const std::vector<Sequence> shorter = shortSequences();
    std::vector<Sequence> sequences = shorter;
    sequences.push_back(longSequence(shorter));
    if (image.config().contextLength < sequences.back().tokens.size())
        return printRefusal(lpl::convert::Refusal{"image_context_too_short",
                                                  "the image holds " + std::to_string(image.config().contextLength) +
                                                      " positions, and the long sequence " +
                                                      std::to_string(sequences.back().tokens.size())});

    std::printf("settings threads=%u margin=%.1f keys=f32 values=f16 decode=one_per_sequence\n", reference.threads(),
                lpl::convert::kAgreementMargin);
    const bool tokens = checkTokens(image, reference);
    const std::vector<lpl::convert::SequenceAgreement> agreements =
        runSequences(reference, image, sequences, true, flagOf(arguments, "--positions"));
    printTotal(agreements, tokens);
    const std::vector<lpl::convert::SequenceAgreement> imageShort(agreements.begin(), agreements.end() - 1);
    const bool swapped =
        runControl(reference, image, lpl::convert::Control::KeyValueSwapped, "key_value_swapped", shorter, imageShort);
    const bool rotated =
        runControl(reference, image, lpl::convert::Control::RotaryBase, "rotary_base_10000", shorter, imageShort);
    const bool agrees = tokens && lpl::convert::agreesAtEveryJudgedPosition(agreements);
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
