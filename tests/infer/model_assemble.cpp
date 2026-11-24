#include "chat_vocabulary.hpp"
#include "model_image_writer.hpp"

#include <lpl/infer/ModelImage.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(model_assemble);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kTurnTokenCount = 3u;
constexpr lpl::core::u32 kVocabularyEntry = 1u;      ///< One-based entry of a parity image's vocabulary.
constexpr lpl::core::u32 kEmbeddingEntry = 2u;       ///< Of its embedding.
constexpr lpl::core::u32 kFirstAttentionEntry = 3u;  ///< Of its first layer's attention.
constexpr lpl::core::u32 kSecondAttentionEntry = 4u; ///< Of its second layer's attention.
constexpr lpl::core::u32 kFinalNormEntry = 5u;       ///< Of its final norm.
constexpr lpl::core::u32 kRotaryEntry = 6u;          ///< Of its rotary table.
constexpr lpl::core::u32 kFirstExpertEntry = 7u;     ///< Of its first layer's expert.
constexpr lpl::core::u32 kSecondExpertEntry = 8u;    ///< Of its second layer's expert.
constexpr lpl::core::u32 kMergesEntry = 9u;          ///< Of its merges.
constexpr lpl::core::u32 kUnknownWord = 9u;          ///< A pre-tokenizer and a chat format no reader knows.
constexpr lpl::core::u32 kTinyArenaBytes = 64u;      ///< Too small for the vocabulary's index.

/** The three tokens of SmolLM2's turns, appended to the parity vocabulary. */
constexpr const char *kTurnTokens[kTurnTokenCount] = {"<|endoftext|>", "<|im_start|>", "<|im_end|>"};

/**
 * @struct Seeded
 * @brief A seeded ChatML model over the parity vocabulary and its turn tokens, one merge, and the parts it is made of,
 *        as a caller holds them.
 */
struct Seeded {
    lpl::infer::Vocab vocab{};                 ///< The parity vocabulary and the three turn tokens.
    lpl::infer::BytePairMerge merge{};         ///< `l` and `d` make `ld`.
    lpl::infer::BytePairMerges merges{};       ///< That merge, indexed.
    lpl::infer::Model model{};                 ///< Derived from the parity seed.
    lpl::infer::LayerWeights *layers{nullptr}; ///< A copy of its layers, in the arena.
    lpl::infer::ModelParts parts{};            ///< Its parts.
};

/**
 * @brief The token whose text is the terminated @p text, in @p vocab.
 */
[[nodiscard]] lpl::core::u32 tokenOf(const lpl::infer::Vocab &vocab, const char *text)
{
    lpl::core::u32 length = 0u;

    while (text[length] != '\0')
        ++length;
    return vocab.find(text, length);
}

/**
 * @brief Seeds the model of @ref Seeded in @p arena and lays its parts out as its caller would hold them.
 */
[[nodiscard]] bool seed(lpl::infer::TensorArena &arena, Seeded &out)
{
    if (!lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens, kTurnTokenCount, out.vocab))
        return false;

    out.merge = lpl::infer::BytePairMerge{tokenOf(out.vocab, "l"), tokenOf(out.vocab, "d"), tokenOf(out.vocab, "ld")};
    if (out.merges.borrow(arena, &out.merge, 1u, out.vocab) != lpl::infer::BytePairMergesVerdict::Indexed)
        return false;

    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();
    shape.vocabSize = out.vocab.size();
    if (!out.model.synthesise(arena, shape, out.vocab, out.merges,
                              lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel, lpl::infer::ChatFormat::ChatMl,
                              lpl::infer::parityModelSeed()))
        return false;

    out.layers = arena.claim<lpl::infer::LayerWeights>(shape.layers);
    if (out.layers == nullptr)
        return false;
    for (lpl::core::u32 l = 0u; l < shape.layers; ++l)
        out.layers[l] = out.model.layer(l);

    out.parts.shape = out.model.config();
    out.parts.vocabularyText = out.vocab.blob();
    out.parts.vocabularyTextBytes = out.vocab.blobBytes();
    out.parts.vocabularyEntries = out.vocab.entries();
    out.parts.merges = &out.merge;
    out.parts.mergeCount = 1u;
    out.parts.preTokenizer = out.model.preTokenizer();
    out.parts.chatFormat = out.model.chatTokens().format;
    out.parts.embedding = out.model.embedding();
    out.parts.layers = out.layers;
    out.parts.finalNorm = out.model.finalNorm();
    out.parts.rotary = out.model.rotaryFrequencies();
    return true;
}

/**
 * @brief What assembling @p parts in @p arena reports, with a model that was ready before, and whether a refusal left
 *        it not ready.
 */
[[nodiscard]] lpl::infer::ModelReadReport assembled(lpl::infer::TensorArena &arena, const Seeded &seeded,
                                                    const lpl::infer::ModelParts &parts, bool &notReadyWhenRefused)
{
    lpl::infer::Model model;

    notReadyWhenRefused = model.assemble(arena, seeded.parts).verdict == lpl::infer::ModelReadVerdict::Read;
    const lpl::infer::ModelReadReport report = model.assemble(arena, parts);
    if (report.verdict != lpl::infer::ModelReadVerdict::Read)
        notReadyWhenRefused = notReadyWhenRefused && !model.ready();
    return report;
}

/**
 * @brief Whether assembling @p parts is refused with @p verdict naming the one-based @p entry, a model that was ready
 *        left not ready.
 */
[[nodiscard]] bool refusedAs(lpl::infer::TensorArena &arena, const Seeded &seeded, const lpl::infer::ModelParts &parts,
                             lpl::infer::ModelReadVerdict verdict, lpl::core::u32 entry)
{
    bool notReady = false;
    const lpl::infer::ModelReadReport report = assembled(arena, seeded, parts, notReady);

    return report.verdict == verdict && report.segment == entry && notReady;
}

/**
 * @brief Whether assembling @p parts is refused as the header of an image would be, with @p image.
 */
[[nodiscard]] bool refusedAsTheHeader(lpl::infer::TensorArena &arena, const Seeded &seeded,
                                      const lpl::infer::ModelParts &parts, lpl::infer::ModelImageVerdict image)
{
    bool notReady = false;
    const lpl::infer::ModelReadReport report = assembled(arena, seeded, parts, notReady);

    return report.verdict == lpl::infer::ModelReadVerdict::ImageRefused && report.image == image &&
           report.segment == 0u && notReady;
}

/**
 * @brief Whether @p parts, with layer @p layer replaced by @p replaced, is refused as misshapen naming @p entry.
 */
[[nodiscard]] bool layerRefusedAs(lpl::infer::TensorArena &arena, const Seeded &seeded, lpl::core::u32 layer,
                                  const lpl::infer::LayerWeights &replaced, lpl::core::u32 entry)
{
    lpl::infer::LayerWeights *const layers = arena.claim<lpl::infer::LayerWeights>(seeded.parts.shape.layers);
    if (layers == nullptr)
        return false;

    for (lpl::core::u32 l = 0u; l < seeded.parts.shape.layers; ++l)
        layers[l] = seeded.layers[l];
    layers[layer] = replaced;
    lpl::infer::ModelParts parts = seeded.parts;
    parts.layers = layers;
    return refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, entry);
}

/**
 * @brief Whether the terminated texts @p lhs and @p rhs are the same.
 */
[[nodiscard]] bool sameText(const char *lhs, const char *rhs)
{
    while (*lhs != '\0' && *lhs == *rhs)
    {
        ++lhs;
        ++rhs;
    }
    return *lhs == *rhs;
}

/**
 * @brief Whether the @p size bytes of @p lhs and @p rhs are the same.
 */
[[nodiscard]] bool sameBytes(const lpl::core::u8 *lhs, const lpl::core::u8 *rhs, lpl::core::u32 size)
{
    for (lpl::core::u32 i = 0u; i < size; ++i)
    {
        if (lhs[i] != rhs[i])
            return false;
    }
    return true;
}

} // namespace

/**
 * @brief A model assembled from a seeded model's parts writes the seeded model's bytes, with its tokenizer and its
 *        turns, and borrows each part where it lies.
 */
LPL_TEST(a_model_assembled_from_a_seeded_model_s_parts_writes_its_bytes)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    Seeded seeded;
    lpl::infer::Model model;
    lpl::core::u32 seededSize = 0u;
    lpl::core::u32 assembledSize = 0u;

    test.check(seed(arena, seeded), "the seeded model builds");
    const lpl::infer::ModelReadReport report = model.assemble(arena, seeded.parts);
    test.check(report.verdict == lpl::infer::ModelReadVerdict::Read && report.segment == 0u && model.ready(),
               "its parts assemble");
    if (!model.ready())
        return;

    const lpl::core::u8 *const fromSeed = lpl::infer::tests::writeModelImage(
        arena, seeded.model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, seededSize);
    const lpl::core::u8 *const fromParts = lpl::infer::tests::writeModelImage(
        arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, assembledSize);
    test.check(fromSeed != nullptr && fromParts != nullptr && seededSize == assembledSize &&
                   sameBytes(fromSeed, fromParts, seededSize),
               "and write the seeded model's image, byte for byte");
    test.check(model.fold(kFnv1aOffsetBasis) == seeded.model.fold(kFnv1aOffsetBasis), "its weights fold the same");
    test.check(model.merges().size() == 1u && model.merges().rank(seeded.merge.left, seeded.merge.right) == 0u &&
                   model.preTokenizer() == lpl::infer::PreTokenizer::IndividualDigitsThenByteLevel,
               "it encodes by the merge and the pre-tokenizer it was given");
    test.check(model.chatTokens().format == lpl::infer::ChatFormat::ChatMl &&
                   model.chatTokens().turnStart == tokenOf(seeded.vocab, "<|im_start|>") &&
                   model.chatTokens().turnEnd == tokenOf(seeded.vocab, "<|im_end|>") &&
                   model.chatTokens().endOfText == tokenOf(seeded.vocab, "<|endoftext|>"),
               "and finds the tokens of its turns by their text");
    test.check(model.embedding().blocks == seeded.parts.embedding.blocks &&
                   model.finalNorm().values == seeded.parts.finalNorm.values &&
                   model.rotaryFrequencies().turnsPerPosition == seeded.parts.rotary.turnsPerPosition &&
                   model.vocab().blob() == seeded.parts.vocabularyText &&
                   model.vocab().entries() == seeded.parts.vocabularyEntries &&
                   model.merges().merges() == seeded.parts.merges && &model.layer(1u) == &seeded.layers[1],
               "each part is borrowed where it lies, none copied");
    test.measure("image_bytes", assembledSize);
}

/**
 * @brief A shape, a vocabulary text, a pre-tokenizer or a chat format that an image's header could not hold is refused
 *        as reading that header refuses it.
 */
LPL_TEST(parts_an_image_header_could_not_hold_are_refused_as_reading_it_would)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    Seeded seeded;

    test.check(seed(arena, seeded), "the seeded model builds");
    lpl::infer::ModelParts headless = seeded.parts;
    headless.shape.heads = 0u;
    lpl::infer::ModelParts ungrouped = seeded.parts;
    ungrouped.shape.keyValueHeads = 3u;
    lpl::infer::ModelParts textless = seeded.parts;
    textless.vocabularyTextBytes = 0u;
    lpl::infer::ModelParts unknownSplit = seeded.parts;
    unknownSplit.preTokenizer = static_cast<lpl::infer::PreTokenizer>(kUnknownWord);
    lpl::infer::ModelParts unknownTurns = seeded.parts;
    unknownTurns.chatFormat = static_cast<lpl::infer::ChatFormat>(kUnknownWord);

    test.check(refusedAsTheHeader(arena, seeded, headless, lpl::infer::ModelImageVerdict::BadShape),
               "a shape without heads is refused as bad_shape, and a ready model is left not ready");
    test.check(refusedAsTheHeader(arena, seeded, ungrouped, lpl::infer::ModelImageVerdict::BadShape),
               "and so are key/value heads that do not divide the heads");
    test.check(refusedAsTheHeader(arena, seeded, textless, lpl::infer::ModelImageVerdict::BadShape),
               "and a vocabulary without text");
    test.check(refusedAsTheHeader(arena, seeded, unknownSplit, lpl::infer::ModelImageVerdict::UnknownPreTokenizer),
               "a pre-tokenizer no reader knows is refused as unknown_pre_tokenizer");
    test.check(refusedAsTheHeader(arena, seeded, unknownTurns, lpl::infer::ModelImageVerdict::UnknownChatFormat),
               "a chat format no reader knows is refused as unknown_chat_format");
}

/**
 * @brief A missing part, or a part whose extents are not the shape's, is refused as misshapen, naming the entry it
 *        would take in the model's image.
 */
LPL_TEST(a_part_that_is_missing_or_not_of_the_shape_is_refused_naming_its_entry)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    Seeded seeded;

    test.check(seed(arena, seeded), "the seeded model builds");
    const lpl::infer::ModelConfig &shape = seeded.parts.shape;

    lpl::infer::ModelParts parts = seeded.parts;
    parts.vocabularyText = nullptr;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, kVocabularyEntry),
               "a vocabulary without its text is refused as parts_misshapen, naming entry 1");
    parts = seeded.parts;
    parts.vocabularyEntries = nullptr;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, kVocabularyEntry),
               "and one without its entries");

    parts = seeded.parts;
    parts.embedding.rows = shape.vocabSize - 1u;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, kEmbeddingEntry),
               "an embedding a row short names entry 2");
    parts = seeded.parts;
    parts.embedding.columns = shape.dim + 1u;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, kEmbeddingEntry),
               "and one a column wide");
    parts = seeded.parts;
    parts.embedding.blocks = nullptr;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, kEmbeddingEntry),
               "and one without its blocks");

    parts = seeded.parts;
    parts.layers = nullptr;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, kFirstAttentionEntry),
               "no layers name the first layer's attention, entry 3");

    lpl::infer::LayerWeights layer = seeded.layers[0];
    layer.query.rows = shape.dim + 1u;
    test.check(layerRefusedAs(arena, seeded, 0u, layer, kFirstAttentionEntry), "a query a row long names entry 3");
    layer = seeded.layers[1];
    layer.key.rows = shape.keyValueWidth() + 1u;
    test.check(layerRefusedAs(arena, seeded, 1u, layer, kSecondAttentionEntry),
               "the second layer's key a row long names entry 4");
    layer = seeded.layers[0];
    layer.value.columns = shape.dim - 1u;
    test.check(layerRefusedAs(arena, seeded, 0u, layer, kFirstAttentionEntry), "a value a column short names entry 3");
    layer = seeded.layers[1];
    layer.output.columns = shape.dim + 1u;
    test.check(layerRefusedAs(arena, seeded, 1u, layer, kSecondAttentionEntry),
               "an output a column wide names entry 4");
    layer = seeded.layers[1];
    layer.output.blocks = nullptr;
    test.check(layerRefusedAs(arena, seeded, 1u, layer, kSecondAttentionEntry), "and one without its blocks");
    layer = seeded.layers[0];
    layer.attentionNorm.count = shape.dim - 1u;
    test.check(layerRefusedAs(arena, seeded, 0u, layer, kFirstAttentionEntry),
               "an attention norm a gain short names entry 3");
    layer = seeded.layers[1];
    layer.feedForwardNorm.values = nullptr;
    test.check(layerRefusedAs(arena, seeded, 1u, layer, kSecondAttentionEntry),
               "a feed-forward norm without its gains names the attention it is written in, entry 4");

    parts = seeded.parts;
    parts.finalNorm.count = shape.dim + 1u;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, kFinalNormEntry),
               "a final norm a gain long names entry 5");
    parts = seeded.parts;
    parts.rotary.pairs = shape.rotaryPairs() + 1u;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, kRotaryEntry),
               "a rotary table a pair long names entry 6");
    parts = seeded.parts;
    parts.rotary.turnsPerPosition = nullptr;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, kRotaryEntry),
               "and one without its words");

    layer = seeded.layers[0];
    layer.gate.rows = shape.ffnHidden - 1u;
    test.check(layerRefusedAs(arena, seeded, 0u, layer, kFirstExpertEntry), "a gate a row short names entry 7");
    layer = seeded.layers[1];
    layer.up.columns = shape.dim + 1u;
    test.check(layerRefusedAs(arena, seeded, 1u, layer, kSecondExpertEntry), "an up a column wide names entry 8");
    layer = seeded.layers[1];
    layer.down.columns = shape.ffnHidden - 1u;
    test.check(layerRefusedAs(arena, seeded, 1u, layer, kSecondExpertEntry), "a down a column short names entry 8");
    layer = seeded.layers[0];
    layer.down.rows = shape.dim + 1u;
    test.check(layerRefusedAs(arena, seeded, 0u, layer, kFirstExpertEntry), "and one a row long names entry 7");

    parts = seeded.parts;
    parts.merges = nullptr;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::PartsMisshapen, kMergesEntry),
               "merges counted but missing name entry 9");
    parts.mergeCount = 0u;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::Read, 0u),
               "while no merge at all assembles, segmented by longest match");
}

/**
 * @brief A vocabulary entry reading would refuse, a vocabulary without the tokens of its format, merges that do not
 * hold and an arena too small are each refused by name.
 */
LPL_TEST(a_vocabulary_merges_or_arena_that_does_not_hold_is_refused_by_name)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    Seeded seeded;
    lpl::infer::Vocab parity;

    test.check(seed(arena, seeded), "the seeded model builds");
    const lpl::core::u32 count = seeded.parts.shape.vocabSize;
    lpl::infer::VocabEntry *const pastTheText = arena.claim<lpl::infer::VocabEntry>(count);
    lpl::infer::VocabEntry *const empty = arena.claim<lpl::infer::VocabEntry>(count);
    test.check(pastTheText != nullptr && empty != nullptr, "the arena holds two copies of the entries");
    if (pastTheText == nullptr || empty == nullptr)
        return;
    for (lpl::core::u32 i = 0u; i < count; ++i)
    {
        pastTheText[i] = seeded.parts.vocabularyEntries[i];
        empty[i] = seeded.parts.vocabularyEntries[i];
    }
    pastTheText[count - 1u].offset = seeded.parts.vocabularyTextBytes;
    empty[0].length = 0u;

    lpl::infer::ModelParts parts = seeded.parts;
    parts.vocabularyEntries = pastTheText;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::VocabularyRefused, kVocabularyEntry),
               "a token past the vocabulary's text is refused as vocabulary_refused, naming entry 1");
    parts.vocabularyEntries = empty;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::VocabularyRefused, kVocabularyEntry),
               "and so is an empty token");

    test.check(lpl::infer::buildParityVocab(arena, parity), "the parity vocabulary builds");
    parts = seeded.parts;
    parts.shape.vocabSize = parity.size();
    parts.vocabularyText = parity.blob();
    parts.vocabularyTextBytes = parity.blobBytes();
    parts.vocabularyEntries = parity.entries();
    parts.embedding.rows = parity.size();
    parts.mergeCount = 0u;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::ChatTokensMissing, kVocabularyEntry),
               "a ChatML model whose vocabulary lacks the tokens of its turns is refused as chat_tokens_missing");
    parts.chatFormat = lpl::infer::ChatFormat::Plain;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::Read, 0u),
               "while the same vocabulary assembles for a Plain model");

    const lpl::infer::BytePairMerge notTheConcatenation{seeded.merge.left, seeded.merge.right,
                                                        tokenOf(seeded.vocab, "gen")};
    const lpl::infer::BytePairMerge pastTheVocabulary{seeded.merge.left, seeded.parts.shape.vocabSize,
                                                      seeded.merge.result};
    parts = seeded.parts;
    parts.merges = &notTheConcatenation;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::MergesRefused, kMergesEntry),
               "a merge whose result is not its pair's bytes is refused as merges_refused, naming entry 9");
    parts.merges = &pastTheVocabulary;
    test.check(refusedAs(arena, seeded, parts, lpl::infer::ModelReadVerdict::MergesRefused, kMergesEntry),
               "and so is one that names a token past the vocabulary");

    lpl::infer::TensorArena tiny{kTinyArenaBytes};
    lpl::infer::Model model;
    const lpl::infer::ModelReadReport report = model.assemble(tiny, seeded.parts);
    test.check(report.verdict == lpl::infer::ModelReadVerdict::ArenaExhausted && report.segment == kVocabularyEntry &&
                   !model.ready(),
               "an arena that cannot hold the vocabulary's index is refused as arena_exhausted, naming entry 1");
    test.check(
        sameText(lpl::infer::modelReadVerdictName(lpl::infer::ModelReadVerdict::PartsMisshapen), "parts_misshapen"),
        "the new verdict is named parts_misshapen");
}
