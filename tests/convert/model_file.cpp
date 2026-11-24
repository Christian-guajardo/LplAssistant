#include "synthetic_model.hpp"

#include <lpl/convert/Converter.hpp>
#include <lpl/convert/ModelFile.hpp>
#include <lpl/testing/Test.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>

LPL_TEST_SUITE(convert_model_file);

namespace {

constexpr std::uint32_t kContext = 32u; ///< The context the tests ask for.

/**
 * @brief Whether @p read holds every key of @p written, of the same kind and value, and no other.
 */
[[nodiscard]] bool sameKeys(const lpl::convert::SourceModel &read, const lpl::convert::SourceModel &written)
{
    if (read.metadata.size() != written.metadata.size())
        return false;
    for (const auto &[key, value] : written.metadata)
    {
        const auto found = read.metadata.find(key);
        if (found == read.metadata.end())
            return false;
        const lpl::convert::MetadataValue &other = found->second;
        if (other.kind != value.kind || other.unsignedWord != value.unsignedWord || other.real != value.real ||
            other.boolean != value.boolean || other.text != value.text || other.texts != value.texts ||
            other.signedWords != value.signedWords)
            return false;
    }
    return true;
}

/**
 * @brief Whether @p read holds every tensor of @p written, of the same type, extents and bytes, and no other.
 */
[[nodiscard]] bool sameTensors(const lpl::convert::SourceModel &read, const lpl::convert::SourceModel &written)
{
    if (read.tensors.size() != written.tensors.size())
        return false;
    for (const auto &[name, tensor] : written.tensors)
    {
        const auto found = read.tensors.find(name);
        if (found == read.tensors.end())
            return false;
        const lpl::convert::SourceTensor &other = found->second;
        if (other.type != tensor.type || other.typeName != tensor.typeName || other.extents != tensor.extents ||
            other.byteCount < tensor.byteCount || std::memcmp(other.bytes, tensor.bytes, tensor.byteCount) != 0)
            return false;
    }
    return true;
}

/**
 * @brief The refusal's name opening the file at @p path gives, or "read".
 */
[[nodiscard]] std::string outcomeOfOpening(const std::string &path)
{
    lpl::convert::ModelFile file;
    const std::optional<lpl::convert::Refusal> refusal = file.open(path);

    return refusal.has_value() ? refusal->name : "read";
}

/**
 * @brief Writes @p bytes as the file at @p path.
 */
void writeBytes(const std::string &path, const std::string &bytes)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
}

/**
 * @brief Checks that the synthetic model of @p weights, written then read, is the same model and converts the same.
 */
void checkRoundTrip(lpl::testing::Test &test, lpl::convert::tests::SyntheticWeights weights, const std::string &name)
{
    const lpl::convert::tests::SyntheticModel model(weights);
    const std::string path = lpl::convert::tests::temporaryPath(name);
    model.write(path);

    lpl::convert::ModelFile file;
    const std::optional<lpl::convert::Refusal> refusal = file.open(path);
    test.check(!refusal.has_value() && file.size() == std::filesystem::file_size(path),
               "the file written through ggml's gguf API maps whole");
    test.check(sameKeys(file.source(), model.source()), "every key reads back of its kind and value, and no other");
    test.check(sameTensors(file.source(), model.source()),
               "every tensor reads back of its type and extents, its bytes the ones written, and no other");

    lpl::convert::ConversionOptions options;
    options.contextLength = kContext;
    const lpl::convert::Conversion fromFile = lpl::convert::convertModel(file.source(), options);
    const lpl::convert::Conversion fromMemory = lpl::convert::convertModel(model.source(), options);
    test.check(!fromFile.refusal.has_value() && !fromFile.image.empty() && fromFile.image == fromMemory.image,
               "and the model read converts to the image the model in memory converts to");
    std::filesystem::remove(path);
}

} // namespace

/**
 * @brief A model of F32 matrices, written as a GGUF file and read back, is the same model.
 */
LPL_TEST(a_written_f32_model_reads_back_as_it_was_written)
{
    checkRoundTrip(test, lpl::convert::tests::SyntheticWeights::Float32, "round-trip-f32.gguf");
}

/**
 * @brief And so is one of Q8_0 matrices, whose blocks the file stores as ggml quantised them.
 */
LPL_TEST(a_written_q8_0_model_reads_back_as_it_was_written)
{
    checkRoundTrip(test, lpl::convert::tests::SyntheticWeights::QuantisedEightBit, "round-trip-q8-0.gguf");
}

/**
 * @brief No file is file_missing, a file ggml cannot parse is file_unreadable, and a file cut before a tensor's
 *        bytes is tensor_outside_file: three messages for three causes.
 */
LPL_TEST(a_missing_an_unreadable_and_a_cut_file_are_refused_apart)
{
    const std::string missing = lpl::convert::tests::temporaryPath("missing.gguf");
    const std::string text = lpl::convert::tests::temporaryPath("text.gguf");
    const std::string empty = lpl::convert::tests::temporaryPath("empty.gguf");
    const std::string cut = lpl::convert::tests::temporaryPath("cut.gguf");
    const std::string directory = lpl::convert::tests::temporaryPath("directory.gguf");
    std::filesystem::remove(missing);
    writeBytes(text, "this is not a GGUF file, though its name says it is");
    writeBytes(empty, "");
    std::filesystem::create_directories(directory);

    test.check(outcomeOfOpening(missing) == "file_missing", "no file is file_missing");
    test.check(outcomeOfOpening(text) == "file_unreadable", "a text file is file_unreadable");
    test.check(outcomeOfOpening(empty) == "file_unreadable", "and so is an empty one");
    test.check(outcomeOfOpening(directory) == "file_unreadable", "and a directory");

    const lpl::convert::tests::SyntheticModel model(lpl::convert::tests::SyntheticWeights::Float32);
    model.write(cut);
    const std::uintmax_t whole = std::filesystem::file_size(cut);
    std::filesystem::resize_file(cut, whole / 2u);
    test.check(outcomeOfOpening(cut) == "tensor_outside_file",
               "a file cut in half, before the last tensors' offsets, is tensor_outside_file");

    model.write(cut);
    std::filesystem::resize_file(cut, whole - 1u);
    lpl::convert::ModelFile file;
    const std::optional<lpl::convert::Refusal> opened = file.open(cut);
    lpl::convert::ConversionOptions options;
    options.contextLength = kContext;
    const lpl::convert::Conversion conversion = lpl::convert::convertModel(file.source(), options);
    test.check(!opened.has_value() && conversion.refusal.has_value() &&
                   conversion.refusal->name == "tensor_outside_file",
               "while one cut a byte short opens, and its last tensor, a byte short, is refused by the converter");
    for (const std::string &path : {text, empty, cut})
        std::filesystem::remove(path);
    std::filesystem::remove(directory);
}
