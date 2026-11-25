#include "chat_vocabulary.hpp"
#include "model_image_writer.hpp"

#include <lpl/infer/ModelImage.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(model_image);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;
constexpr lpl::core::u32 kImageVersion = 1u;
constexpr lpl::core::u32 kRetiredImageBytes = 256u;
constexpr lpl::core::u32 kParityVocabulary = 121u;
constexpr lpl::core::u32 kMadeUpVocabularyText = 300u;
constexpr lpl::core::u32 kMadeUpMerges = 5u;
constexpr lpl::core::u32 kParitySegments = 9u;
constexpr lpl::core::u32 kFinalNormEntry = 5u;          ///< One-based entry of the parity image's final norm.
constexpr lpl::core::u32 kRotaryEntry = 6u;             ///< One-based entry of its rotary table.
constexpr lpl::core::u32 kFirstExpertEntry = 7u;        ///< One-based entry of its first expert.
constexpr lpl::core::u32 kSecondExpertEntry = 8u;       ///< One-based entry of its second expert.
constexpr lpl::core::u32 kMergesEntry = 9u;             ///< One-based entry of its merges, empty.
constexpr lpl::core::u32 kParityPairs = 6u;             ///< Pairs of the parity model's heads of 12 channels.
constexpr lpl::core::u32 kMadeUpTurnStep = 0x9E3779B9u; ///< Pair i of a made-up table turns by i + 1 times this.
constexpr lpl::core::u32 kTableRoom = 16u;
constexpr lpl::core::u32 kEntryRole = 0u;
constexpr lpl::core::u32 kEntryOffset = 16u;
constexpr lpl::core::u32 kEntryLength = 24u;
constexpr lpl::core::u32 kEntrySignature = 32u;
constexpr lpl::core::u32 kEntryReserved = 36u;
constexpr lpl::core::u32 kVocabularyEntryLength = 4u;
constexpr lpl::core::u32 kTinyArenaBytes = 16u;
constexpr lpl::core::u32 kWordBytes = 4u;
constexpr lpl::core::u32 kArenasForThreeShiftedCopies = 4u;
constexpr lpl::core::u32 kTokenPastTheText = 2u;
constexpr lpl::core::u32 kHeaderVersionWord = 4u;
constexpr lpl::core::u32 kHeaderBytesWord = 8u;
constexpr lpl::core::u32 kHeaderAlignmentWord = 12u;
constexpr lpl::core::u32 kHeaderFlagsWord = 16u;
constexpr lpl::core::u32 kHeaderCountWord = 20u;
constexpr lpl::core::u32 kHeaderImageBytesWord = 24u;
constexpr lpl::core::u32 kHeaderHeadsWord = 40u;
constexpr lpl::core::u32 kHeaderMergeCountWord = 64u;
constexpr lpl::core::u32 kHeaderPreTokenizerWord = 68u;
constexpr lpl::core::u32 kHeaderKeyValueHeadsWord = 72u;
constexpr lpl::core::u32 kMergeBytes = 12u;
constexpr lpl::core::u32 kNormEpsilon = 42950u;       ///< 1e-5 in units of 2^-32, rounded.
constexpr lpl::core::u32 kGroupedHeads = 6u;          ///< Query heads of the grouped model: three per key/value head.
constexpr lpl::core::u32 kGroupedKeyValueHeads = 2u;  ///< Its key and value heads.
constexpr lpl::core::u32 kGroupedAttentionEntry = 3u; ///< One-based entry of the grouped image's first attention.
constexpr lpl::core::usize kChatArenaBytes = 1024u * 1024u;
constexpr lpl::core::u32 kChatMlWord = 1u; ///< The word of the ChatML format.
constexpr lpl::core::u32 kTurnTokenCount = 3u;

/** The three tokens of SmolLM2's turns, appended to the parity vocabulary at 121, 122 and 123. */
constexpr const char *kTurnTokens[kTurnTokenCount] = {"<|endoftext|>", "<|im_start|>", "<|im_end|>"};

/**
 * @brief The parity model's shape with the canonical vocabulary's size.
 */
[[nodiscard]] lpl::infer::ModelConfig parityShape()
{
    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();

    shape.vocabSize = kParityVocabulary;
    return shape;
}

/**
 * @brief The parity model's shape with 6 heads of 8 channels reading 2 key/value heads.
 */
[[nodiscard]] lpl::infer::ModelConfig groupedShape()
{
    lpl::infer::ModelConfig shape = parityShape();

    shape.heads = kGroupedHeads;
    shape.keyValueHeads = kGroupedKeyValueHeads;
    return shape;
}

/**
 * @brief Builds the canonical vocabulary and the parity model in @p arena.
 */
[[nodiscard]] bool synthesiseParityModel(lpl::infer::TensorArena &arena, lpl::infer::Vocab &vocab,
                                         lpl::infer::Model &model)
{
    if (!lpl::infer::buildParityVocab(arena, vocab))
        return false;

    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();

    shape.vocabSize = vocab.size();
    return model.synthesise(arena, shape, vocab, lpl::infer::parityModelSeed());
}

/**
 * @brief Builds the canonical vocabulary and the model of groupedShape() in @p arena.
 */
[[nodiscard]] bool synthesiseGroupedModel(lpl::infer::TensorArena &arena, lpl::infer::Vocab &vocab,
                                          lpl::infer::Model &model)
{
    if (!lpl::infer::buildParityVocab(arena, vocab))
        return false;

    lpl::infer::ModelConfig shape = groupedShape();

    shape.vocabSize = vocab.size();
    return model.synthesise(arena, shape, vocab, lpl::infer::parityModelSeed());
}

/**
 * @brief Whether @p address lies inside the @p size bytes at @p image.
 */
[[nodiscard]] bool inside(const void *address, const lpl::core::u8 *image, lpl::core::u32 size)
{
    const lpl::core::u8 *const byte = static_cast<const lpl::core::u8 *>(address);

    return byte >= image && byte < image + size;
}

/**
 * @brief Whether every weight of @p model, and its vocabulary's text and entries, lie inside the @p size bytes at
 *        @p image.
 */
[[nodiscard]] bool everyWeightInside(const lpl::infer::Model &model, const lpl::core::u8 *image, lpl::core::u32 size)
{
    bool all = inside(model.embedding().blocks, image, size) && inside(model.finalNorm().values, image, size) &&
               inside(model.vocab().blob(), image, size) && inside(model.vocab().entries(), image, size);

    for (lpl::core::u32 l = 0u; all && l < model.config().layers; ++l)
    {
        const lpl::infer::LayerWeights &layer = model.layer(l);
        all = inside(layer.query.blocks, image, size) && inside(layer.key.blocks, image, size) &&
              inside(layer.value.blocks, image, size) && inside(layer.output.blocks, image, size) &&
              inside(layer.gate.blocks, image, size) && inside(layer.up.blocks, image, size) &&
              inside(layer.down.blocks, image, size) && inside(layer.attentionNorm.values, image, size) &&
              inside(layer.feedForwardNorm.values, image, size);
    }
    return all;
}

/**
 * @brief The FNV-1a fold of @p length bytes at @p offset of @p bytes, worked apart from the module's.
 */
[[nodiscard]] lpl::core::u32 foldBytes(const lpl::core::u8 *bytes, lpl::core::u64 offset, lpl::core::u64 length)
{
    lpl::core::u32 hash = kFnv1aOffsetBasis;

    for (lpl::core::u64 at = offset; at < offset + length; ++at)
    {
        hash ^= bytes[at];
        hash *= kFnv1aPrime;
    }
    return hash;
}

/**
 * @brief Whether every byte of @p bytes from @p from to @p to is zero, and the span not empty.
 */
[[nodiscard]] bool zeroBetween(const lpl::core::u8 *bytes, lpl::core::u64 from, lpl::core::u64 to)
{
    bool zero = from < to;

    for (lpl::core::u64 at = from; zero && at < to; ++at)
        zero = bytes[at] == 0u;
    return zero;
}

/**
 * @brief Writes @p value little-endian at @p offset of @p bytes.
 */
void putWordAt(lpl::core::u8 *bytes, lpl::core::u32 offset, lpl::core::u32 value)
{
    for (lpl::core::u32 i = 0u; i < 4u; ++i)
        bytes[offset + i] = static_cast<lpl::core::u8>((value >> (8u * i)) & 0xFFu);
}

/**
 * @brief Reads the little-endian word at @p offset of @p bytes.
 */
[[nodiscard]] lpl::core::u32 wordAt(const lpl::core::u8 *bytes, lpl::core::u32 offset)
{
    lpl::core::u32 value = 0u;

    for (lpl::core::u32 i = 0u; i < 4u; ++i)
        value |= static_cast<lpl::core::u32>(bytes[offset + i]) << (8u * i);
    return value;
}

/**
 * @brief Where the one-based table entry @p entry starts.
 */
[[nodiscard]] lpl::core::u32 entryAt(lpl::core::u32 entry)
{
    return lpl::infer::kModelImageHeaderBytes + (entry - 1u) * lpl::infer::kModelSegmentEntryBytes;
}

/**
 * @brief Whether @p tokens are a ChatML turn's, with the three tokens appended to the parity vocabulary.
 */
[[nodiscard]] bool hasTheAppendedTurnTokens(const lpl::infer::ChatTokens &tokens)
{
    return tokens.format == lpl::infer::ChatFormat::ChatMl && tokens.endOfText == 121u && tokens.turnStart == 122u &&
           tokens.turnEnd == 123u;
}

/**
 * @brief Whether @p lhs and @p rhs are the same null-terminated text.
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
 * @brief What checking @p image finds with its word at @p offset set to @p value; the word is put back after.
 */
[[nodiscard]] lpl::infer::ModelImageInspection inspectedWith(lpl::core::u8 *image, lpl::core::u32 size,
                                                             lpl::core::u32 offset, lpl::core::u32 value)
{
    lpl::infer::ModelSegment segments[kTableRoom];
    const lpl::core::u32 kept = wordAt(image, offset);

    putWordAt(image, offset, value);
    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    putWordAt(image, offset, kept);
    return inspection;
}

/**
 * @brief Whether @p image, its word at @p offset set to @p value, is refused with @p verdict naming @p segment; the
 *        word is put back after.
 */
[[nodiscard]] bool refusedWith(lpl::core::u8 *image, lpl::core::u32 size, lpl::core::u32 offset, lpl::core::u32 value,
                               lpl::infer::ModelImageVerdict verdict, lpl::core::u32 segment)
{
    const lpl::infer::ModelImageInspection inspection = inspectedWith(image, size, offset, value);

    return inspection.verdict == verdict && inspection.segment == segment;
}

/**
 * @brief Whether @p image, its word at @p offset set to @p value, is refused at its header by the name @p verdict; the
 *        word is put back after.
 */
[[nodiscard]] bool refusedByName(lpl::core::u8 *image, lpl::core::u32 size, lpl::core::u32 offset, lpl::core::u32 value,
                                 const char *verdict)
{
    const lpl::infer::ModelImageInspection inspection = inspectedWith(image, size, offset, value);

    return sameText(lpl::infer::modelImageVerdictName(inspection.verdict), verdict) && inspection.segment == 0u;
}

/**
 * @brief Whether @p segment has @p role, @p residency, @p layer and @p expert.
 */
[[nodiscard]] bool isSegment(const lpl::infer::ModelSegment &segment, lpl::infer::ModelSegmentRole role,
                             lpl::infer::ModelSegmentResidency residency, lpl::core::u32 layer, lpl::core::u32 expert)
{
    return segment.role == role && segment.residency == residency && segment.layer == layer && segment.expert == expert;
}

/**
 * @brief Whether @p left and @p right hold the same tokens: each entry's offset and length, and the text.
 */
[[nodiscard]] bool sameVocabulary(const lpl::infer::Vocab &left, const lpl::infer::Vocab &right)
{
    bool same = left.size() == right.size() && left.blobBytes() == right.blobBytes();

    for (lpl::core::u32 i = 0u; same && i < left.size(); ++i)
        same = left.entries()[i].offset == right.entries()[i].offset &&
               left.entries()[i].length == right.entries()[i].length;
    for (lpl::core::u32 i = 0u; same && i < left.blobBytes(); ++i)
        same = left.blob()[i] == right.blob()[i];
    return same;
}

} // namespace

/**
 * @brief A dense model's image holds its residents after the table, 64-byte aligned, the rotary table last among them,
 *        then one expert per layer on a 4 KiB boundary, then the merges, 64-byte aligned.
 *
 * @details Worked by hand for the parity shape (dim 48, 2 layers, 4 heads, feed-forward 96) with 121 tokens, 300
 *          bytes of text and 5 merges, a quantised block 36 bytes: the table ends at 80 + 9 * 40 = 440; the vocabulary
 *          (8 * 121 + 300 = 1268) at 448; the embedding (121 rows of 2 blocks, 8712) at 1728; each attention (2 norms
 *          of 48 words and 4 matrices of 48 rows of 2 blocks, 14208) at 10496 and 24704; the final norm (192) at 38912,
 *          ending at 39104; the rotary table (6 pairs of a head of 12 channels, a word each, 24) at 39104, ending at
 *          39128; each expert (gate and up of 96 rows of 2 blocks, down of 48 rows of 3, 19008) at 40960 and 61440,
 *          ending at 80448; the merges (5 of 12 bytes, 60) at 80448, the image ending at 80508.
 */
LPL_TEST(a_dense_model_lays_out_its_residents_then_one_expert_per_layer)
{
    lpl::infer::ModelSegment segments[kTableRoom];
    const lpl::core::u64 bytes = lpl::infer::layOutModelImage(
        parityShape(), lpl::infer::ModelTokenizerExtents{kMadeUpVocabularyText, kMadeUpMerges},
        lpl::infer::kModelImageDefaultAlignmentLog2, segments, kTableRoom);
    const lpl::infer::ModelSegmentResidency resident = lpl::infer::ModelSegmentResidency::Resident;
    const lpl::infer::ModelSegmentResidency shared = lpl::infer::ModelSegmentResidency::SharedExpert;

    test.check(bytes == 80508u, "the image is 80508 bytes");
    test.check(
        isSegment(segments[0], lpl::infer::ModelSegmentRole::Vocabulary, resident, 0u, 0u) &&
            isSegment(segments[1], lpl::infer::ModelSegmentRole::Embedding, resident, 0u, 0u) &&
            isSegment(segments[2], lpl::infer::ModelSegmentRole::Attention, resident, 1u, 0u) &&
            isSegment(segments[3], lpl::infer::ModelSegmentRole::Attention, resident, 2u, 0u) &&
            isSegment(segments[4], lpl::infer::ModelSegmentRole::FinalNorm, resident, 0u, 0u) &&
            isSegment(segments[5], lpl::infer::ModelSegmentRole::Rotary, resident, 0u, 0u) &&
            isSegment(segments[6], lpl::infer::ModelSegmentRole::FeedForward, shared, 1u, 1u) &&
            isSegment(segments[7], lpl::infer::ModelSegmentRole::FeedForward, shared, 2u, 1u) &&
            isSegment(segments[8], lpl::infer::ModelSegmentRole::Merges, resident, 0u, 0u),
        "the residents in order, the rotary table after the final norm, each layer's one shared expert, then the "
        "merges");
    test.check(segments[0].offset == 448u && segments[1].offset == 1728u && segments[2].offset == 10496u &&
                   segments[3].offset == 24704u && segments[4].offset == 38912u && segments[5].offset == 39104u,
               "each resident on the next 64 bytes after the one before, the first after the table");
    test.check(segments[8].offset == 80448u && segments[8].length == kMadeUpMerges * kMergeBytes,
               "the merges right after the last expert, 12 bytes each");
    test.check(segments[0].length == 1268u && segments[1].length == 8712u && segments[2].length == 14208u &&
                   segments[3].length == 14208u && segments[4].length == 192u && segments[5].length == 24u &&
                   segments[6].length == 19008u && segments[7].length == 19008u,
               "each length the shape gives its role, a word per pair for the rotary table");
    test.check(segments[6].offset == 40960u && segments[7].offset == 61440u, "each expert on the next 4 KiB boundary");
    test.measure("image_bytes", static_cast<lpl::core::u32>(bytes));
}

/**
 * @brief A model of 6 heads reading 2 key/value heads lays out each attention with key and value matrices 16 rows
 *        high, one row per channel of a key/value head: 9600 bytes where the parity model's 4 heads of their own take
 *        14208.
 *
 * @details Worked by hand for the parity shape with 6 heads of 8 channels and 2 key/value heads, 121 tokens, 300 bytes
 *          of text and 5 merges: the table ends at 80 + 9 * 40 = 440; the vocabulary (1268) at 448; the embedding
 *          (8712) at 1728; each attention (2 norms of 48 words, 384; query and output of 48 rows of 2 blocks, 3456
 *          each; key and value of 16 rows of 2 blocks, 1152 each: 9600) at 10496 and 20096; the final norm (192) at
 *          29696; the rotary table (4 pairs of a head of 8 channels, 16) at 29888, ending at 29904; each expert (19008)
 *          at 32768 and 53248, ending at 72256; the merges (60) at 72256, the image ending at 72316.
 */
LPL_TEST(a_grouped_model_lays_out_its_key_and_value_matrices_one_key_value_head_wide)
{
    lpl::infer::ModelSegment segments[kTableRoom];
    const lpl::core::u64 bytes = lpl::infer::layOutModelImage(
        groupedShape(), lpl::infer::ModelTokenizerExtents{kMadeUpVocabularyText, kMadeUpMerges},
        lpl::infer::kModelImageDefaultAlignmentLog2, segments, kTableRoom);

    test.check(bytes == 72316u, "the image is 72316 bytes");
    test.check(segments[2].length == 9600u && segments[3].length == 9600u,
               "each attention is 9600 bytes, its key and value matrices 16 rows high");
    test.check(segments[5].length == 16u, "the rotary table holds a word for each of a head's 4 pairs");
    test.check(segments[0].offset == 448u && segments[1].offset == 1728u && segments[2].offset == 10496u &&
                   segments[3].offset == 20096u && segments[4].offset == 29696u && segments[5].offset == 29888u,
               "each resident on the next 64 bytes after the one before");
    test.check(segments[6].offset == 32768u && segments[7].offset == 53248u && segments[8].offset == 72256u,
               "each expert on the next 4 KiB boundary, then the merges");
    test.measure("grouped_image_bytes", static_cast<lpl::core::u32>(bytes));
}

/**
 * @brief An alignment of 2^21 puts each expert on a 2 MiB boundary; an alignment below 2^12 or above 2^22, a shape
 *        refused, no vocabulary text or too little room for the table lays nothing out.
 */
LPL_TEST(an_alignment_is_kept_from_4_kib_to_4_mib_and_refused_past_them)
{
    lpl::infer::ModelSegment segments[kTableRoom];
    lpl::infer::ModelConfig badHeads = parityShape();

    badHeads.heads = 5u;
    badHeads.keyValueHeads = 5u;
    const lpl::infer::ModelTokenizerExtents text{kMadeUpVocabularyText, 0u};
    const lpl::core::u64 at2Mib = lpl::infer::layOutModelImage(parityShape(), text, 21u, segments, kTableRoom);

    test.check(at2Mib == 4194304u + 19008u && segments[6].offset == 2097152u && segments[7].offset == 4194304u,
               "at 2^21, the experts at 2 MiB and 4 MiB");
    test.check(lpl::infer::layOutModelImage(parityShape(), text, 11u, segments, kTableRoom) == 0u &&
                   lpl::infer::layOutModelImage(parityShape(), text, 23u, segments, kTableRoom) == 0u,
               "2^11 and 2^23 are refused");
    test.check(lpl::infer::layOutModelImage(badHeads, text, 12u, segments, kTableRoom) == 0u &&
                   lpl::infer::layOutModelImage(parityShape(), lpl::infer::ModelTokenizerExtents{0u, 0u}, 12u, segments,
                                                kTableRoom) == 0u,
               "so are 5 heads over a width of 48, and no vocabulary text");
    test.check(lpl::infer::layOutModelImage(parityShape(), text, 12u, segments, kParitySegments - 1u) == 0u,
               "and a table of 9 entries into room for 8");
}

/**
 * @brief The parity model's image, version 1, is accepted, each segment's signature holds, and it reads back
 *        to the same weights.
 */
LPL_TEST(a_written_image_is_accepted_and_reads_back_to_the_same_weights)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Model reopened;
    lpl::infer::ModelSegment segments[kTableRoom];
    lpl::core::u32 size = 0u;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr && lpl::infer::modelImageVersion(image, size) == kImageVersion,
               "it is written as version 1");
    if (image == nullptr)
        return;

    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    bool everySegmentSigned = inspection.verdict == lpl::infer::ModelImageVerdict::Accepted;
    for (lpl::core::u32 index = 0u; everySegmentSigned && index < inspection.header.segmentCount; ++index)
        everySegmentSigned = lpl::infer::modelSegmentSignature(image, segments[index]) == segments[index].signature;

    test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted && inspection.segment == 0u,
               "its header and table are accepted");
    test.check(inspection.header.alignmentLog2 == 12u && inspection.header.segmentCount == kParitySegments &&
                   inspection.header.imageBytes == size,
               "at 4 KiB, with 8 segments, as long as written");
    test.check(everySegmentSigned, "and each segment's bytes give its signature");
    const lpl::infer::ModelSegment &finalNorm = segments[kFinalNormEntry - 1u];
    const lpl::infer::ModelSegment &rotary = segments[kRotaryEntry - 1u];
    test.check(finalNorm.signature == foldBytes(image, finalNorm.offset, finalNorm.length),
               "the final norm's signature is the FNV-1a of its bytes, folded apart");
    test.check(zeroBetween(image, rotary.offset + rotary.length, segments[kFirstExpertEntry - 1u].offset),
               "the padding before the first expert is zero");
    test.check(reopened.open(arena, image, size) && reopened.fold(kFnv1aOffsetBasis) == model.fold(kFnv1aOffsetBasis),
               "it reads back to the same weights, bit for bit");
    test.check(sameVocabulary(reopened.vocab(), model.vocab()), "and the same tokens, which the weights do not fold");
    test.measure("image_bytes", size);
    test.measure("first_expert_offset", static_cast<lpl::core::u32>(segments[kFirstExpertEntry - 1u].offset));
}

/**
 * @brief An image carries its model's norm epsilon, and every reading gives it back.
 */
LPL_TEST(an_image_carries_its_model_s_norm_epsilon_and_reads_back_with_it)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Model reopened;
    lpl::infer::Model viewed;
    lpl::core::u32 size = 0u;

    test.check(lpl::infer::buildParityVocab(arena, vocab), "the canonical vocabulary builds");
    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();
    shape.vocabSize = vocab.size();
    shape.normEpsilon = kNormEpsilon;
    test.check(model.synthesise(arena, shape, vocab, lpl::infer::parityModelSeed()) &&
                   model.config().normEpsilon == kNormEpsilon,
               "a model of epsilon 1e-5 builds");

    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and its image is written");
    if (image == nullptr)
        return;

    const lpl::infer::ModelImageInspection inspection = lpl::infer::inspectModelImage(image, size, nullptr, 0u);
    test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                   inspection.header.shape.normEpsilon == kNormEpsilon,
               "its header gives the epsilon");
    test.check(reopened.open(arena, image, size) && reopened.config().normEpsilon == kNormEpsilon,
               "a copy reads it back");
    test.check(viewed.view(arena, image, size).verdict == lpl::infer::ModelReadVerdict::Read &&
                   viewed.config().normEpsilon == kNormEpsilon,
               "and so does a reading in place");
}

/**
 * @brief An image of a model of 6 heads for 2 key/value heads gives them in its header, and reads back to the same
 *        weights, copied and in place, still grouped.
 */
LPL_TEST(an_image_carries_its_key_value_heads_and_reads_back_grouped)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Model reopened;
    lpl::infer::Model viewed;
    lpl::core::u32 size = 0u;

    test.check(synthesiseGroupedModel(arena, vocab, model) && model.config().keyValueHeads == kGroupedKeyValueHeads,
               "a model of 6 heads for 2 key/value heads builds");
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and its image is written");
    if (image == nullptr)
        return;

    const lpl::infer::ModelImageInspection inspection = lpl::infer::inspectModelImage(image, size, nullptr, 0u);
    test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                   inspection.header.shape.keyValueHeads == kGroupedKeyValueHeads &&
                   wordAt(image, kHeaderKeyValueHeadsWord) == kGroupedKeyValueHeads,
               "its header gives 2 key/value heads, in the word at byte 72");
    test.check(reopened.open(arena, image, size) && reopened.config().keyValueHeads == kGroupedKeyValueHeads &&
                   reopened.fold(kFnv1aOffsetBasis) == model.fold(kFnv1aOffsetBasis),
               "a copy reads back the same weights, grouped");
    test.check(viewed.view(arena, image, size).verdict == lpl::infer::ModelReadVerdict::Read &&
                   viewed.config().keyValueHeads == kGroupedKeyValueHeads &&
                   viewed.fold(kFnv1aOffsetBasis) == model.fold(kFnv1aOffsetBasis),
               "and so does a reading in place");
}

/**
 * @brief A header giving no key/value heads, or 4 for 6 heads, is refused as a bad shape. Giving 6 for 6 makes a shape
 *        that holds, so the grouped image's first attention is then refused for its length.
 */
LPL_TEST(key_value_heads_that_do_not_divide_the_heads_are_refused_as_a_bad_shape)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::ModelSegment segments[kTableRoom];
    lpl::infer::ModelConfig fourForSix = groupedShape();
    lpl::core::u32 size = 0u;

    fourForSix.keyValueHeads = 4u;
    test.check(synthesiseGroupedModel(arena, vocab, model), "a model of 6 heads for 2 key/value heads builds");
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and is written");
    if (image == nullptr)
        return;

    test.check(refusedWith(image, size, kHeaderKeyValueHeadsWord, 0u, lpl::infer::ModelImageVerdict::BadShape, 0u) &&
                   refusedWith(image, size, kHeaderKeyValueHeadsWord, 4u, lpl::infer::ModelImageVerdict::BadShape, 0u),
               "no key/value heads, or 4 for 6 heads, is a bad shape");
    test.check(refusedWith(image, size, kHeaderKeyValueHeadsWord, kGroupedHeads,
                           lpl::infer::ModelImageVerdict::SegmentWrongLength, kGroupedAttentionEntry),
               "6 for 6 is a shape, whose attention of 14208 bytes is not the 9600 written");
    test.check(lpl::infer::layOutModelImage(fourForSix,
                                            lpl::infer::ModelTokenizerExtents{kMadeUpVocabularyText, kMadeUpMerges},
                                            lpl::infer::kModelImageDefaultAlignmentLog2, segments, kTableRoom) == 0u,
               "and nothing is laid out for 4 key/value heads over 6 heads");
}

/**
 * @brief Written at an alignment of 2^16, the image keeps each expert on a 64 KiB boundary and still reads back.
 */
LPL_TEST(an_image_written_at_64_kib_reads_back_with_its_experts_on_that_boundary)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Model reopened;
    lpl::infer::ModelSegment segments[kTableRoom];
    lpl::core::u32 size = 0u;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    lpl::core::u8 *const image = lpl::infer::tests::writeModelImage(arena, model, 16u, 0u, size);
    test.check(image != nullptr, "it is written at 2^16");
    if (image == nullptr)
        return;

    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                   inspection.header.alignmentLog2 == 16u && segments[kFirstExpertEntry - 1u].offset % 65536u == 0u &&
                   segments[kSecondExpertEntry - 1u].offset % 65536u == 0u,
               "accepted, each expert on a 64 KiB boundary");
    test.check(reopened.open(arena, image, size) && reopened.fold(kFnv1aOffsetBasis) == model.fold(kFnv1aOffsetBasis),
               "and it reads back to the same weights");
    test.measure("image_bytes", size);
}

/**
 * @brief Every change to the header or the table that the layout does not allow is refused, by name and, for an
 *        entry, by its one-based index.
 */
LPL_TEST(every_corruption_of_the_header_or_the_table_is_refused_by_name)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::ModelSegment segments[kTableRoom];
    lpl::core::u32 size = 0u;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and is written");
    if (image == nullptr)
        return;

    const lpl::core::u32 firstExpertOffset = wordAt(image, entryAt(kFirstExpertEntry) + kEntryOffset);
    const lpl::core::u32 embeddingOffset = wordAt(image, entryAt(2u) + kEntryOffset);
    const lpl::core::u32 lastExpertOffset = wordAt(image, entryAt(kSecondExpertEntry) + kEntryOffset);
    const lpl::core::u32 lastExpertLength = wordAt(image, entryAt(kSecondExpertEntry) + kEntryLength);
    const lpl::core::u32 rotaryLength = wordAt(image, entryAt(kRotaryEntry) + kEntryLength);

    test.check(
        lpl::infer::inspectModelImage(image, lpl::infer::kModelImageHeaderBytes - 1u, segments, kTableRoom).verdict ==
            lpl::infer::ModelImageVerdict::TooShort,
        "an image one byte shorter than its header");
    test.check(refusedWith(image, size, 0u, 0x4D4C504Du, lpl::infer::ModelImageVerdict::BadMagic, 0u),
               "a magic not LPLM");
    test.check(refusedWith(image, size, kHeaderVersionWord, 3u, lpl::infer::ModelImageVerdict::UnknownVersion, 0u),
               "a version 3");
    test.check(refusedWith(image, size, kHeaderBytesWord, 60u, lpl::infer::ModelImageVerdict::BadHeader, 0u) &&
                   refusedWith(image, size, kHeaderFlagsWord, 1u, lpl::infer::ModelImageVerdict::BadHeader, 0u),
               "header bytes of 60, a flag set");
    test.check(refusedWith(image, size, kHeaderBytesWord, 64u, lpl::infer::ModelImageVerdict::BadHeader, 0u),
               "header bytes of 64, as LplAssistant 0.6.2 wrote, without a merge count or a pre-tokenizer");
    test.check(refusedWith(image, size, kHeaderBytesWord, 72u, lpl::infer::ModelImageVerdict::BadHeader, 0u),
               "header bytes of 72, without a count of key/value heads");
    test.check(refusedWith(image, size, kHeaderBytesWord, 76u, lpl::infer::ModelImageVerdict::BadHeader, 0u),
               "header bytes of 76, without a chat format");
    test.check(refusedWith(image, size, kHeaderAlignmentWord, 11u, lpl::infer::ModelImageVerdict::BadAlignment, 0u) &&
                   refusedWith(image, size, kHeaderAlignmentWord, 23u, lpl::infer::ModelImageVerdict::BadAlignment, 0u),
               "an alignment of 2^11 or 2^23");
    test.check(refusedWith(image, size, kHeaderHeadsWord, 5u, lpl::infer::ModelImageVerdict::BadShape, 0u),
               "5 heads over a width of 48");
    test.check(
        refusedWith(image, size, kHeaderImageBytesWord, size + 1u, lpl::infer::ModelImageVerdict::BadImageBytes, 0u),
        "a header claiming one byte more than the image");
    test.check(refusedWith(image, size, kHeaderCountWord, 7u, lpl::infer::ModelImageVerdict::BadSegmentCount, 0u) &&
                   refusedWith(image, size, kHeaderCountWord, kTableRoom + 1u,
                               lpl::infer::ModelImageVerdict::BadSegmentCount, 0u),
               "7 segments for 2 layers, or more than the room given");
    test.check(
        refusedWith(image, size, entryAt(1u) + kEntryRole, 2u, lpl::infer::ModelImageVerdict::SegmentUnexpected, 1u) &&
            refusedWith(image, size, entryAt(3u) + kEntryReserved, 1u, lpl::infer::ModelImageVerdict::SegmentUnexpected,
                        3u),
        "an embedding where the vocabulary goes, an entry's reserved word set");
    test.check(refusedWith(image, size, entryAt(kRotaryEntry) + kEntryRole,
                           static_cast<lpl::core::u32>(lpl::infer::ModelSegmentRole::FinalNorm),
                           lpl::infer::ModelImageVerdict::SegmentUnexpected, kRotaryEntry),
               "a second final norm where the rotary table goes");
    test.check(refusedWith(image, size, entryAt(kRotaryEntry) + kEntryLength, rotaryLength - kWordBytes,
                           lpl::infer::ModelImageVerdict::SegmentWrongLength, kRotaryEntry),
               "a rotary table one pair shorter than the heads' pairs");
    test.check(refusedWith(image, size, entryAt(kSecondExpertEntry) + kEntryLength, lastExpertLength + 1u,
                           lpl::infer::ModelImageVerdict::SegmentWrongLength, kSecondExpertEntry),
               "an expert one byte longer than its shape");
    test.check(refusedWith(image, size, entryAt(kFirstExpertEntry) + kEntryOffset, firstExpertOffset + 64u,
                           lpl::infer::ModelImageVerdict::SegmentMisaligned, kFirstExpertEntry),
               "an expert 64 bytes off its boundary");
    test.check(refusedWith(image, size, entryAt(2u) + kEntryOffset, embeddingOffset - 64u,
                           lpl::infer::ModelImageVerdict::SegmentOverlaps, 2u),
               "an embedding starting inside the vocabulary");
    test.check(refusedWith(image, size, entryAt(kSecondExpertEntry) + kEntryOffset, lastExpertOffset + 4096u,
                           lpl::infer::ModelImageVerdict::SegmentOutside, kSecondExpertEntry),
               "the last expert one page further, past the image");
    test.check(refusedWith(image, size, entryAt(kSecondExpertEntry) + kEntryOffset, ((size / 4096u) + 2u) * 4096u,
                           lpl::infer::ModelImageVerdict::SegmentOutside, kSecondExpertEntry),
               "the last expert starting past the image's end");
    test.check(lpl::infer::inspectModelImage(image, size, segments, kParitySegments - 1u).verdict ==
                   lpl::infer::ModelImageVerdict::BadSegmentCount,
               "a table of 9 entries into room for 8");
    test.check(
        refusedWith(image, size, kHeaderPreTokenizerWord, 2u, lpl::infer::ModelImageVerdict::UnknownPreTokenizer, 0u),
        "a pre-tokenizer this reader does not know");
    test.check(refusedWith(image, size, entryAt(kMergesEntry) + kEntryLength, kMergeBytes,
                           lpl::infer::ModelImageVerdict::SegmentWrongLength, kMergesEntry) &&
                   refusedWith(image, size, kHeaderMergeCountWord, 1u,
                               lpl::infer::ModelImageVerdict::SegmentWrongLength, kMergesEntry),
               "a merge where the header counts none, or one counted where the table holds none");
    test.check(refusedWith(image, size, entryAt(kMergesEntry) + kEntryRole,
                           static_cast<lpl::core::u32>(lpl::infer::ModelSegmentRole::Rotary),
                           lpl::infer::ModelImageVerdict::SegmentUnexpected, kMergesEntry),
               "a second rotary table where the merges go");
    test.check(refusedWith(image, 100u, kHeaderImageBytesWord, 100u, lpl::infer::ModelImageVerdict::TooShort, 0u),
               "100 bytes that say 100 but cannot hold the table");
    test.check(lpl::infer::inspectModelImage(image, size, segments, kTableRoom).verdict ==
                   lpl::infer::ModelImageVerdict::Accepted,
               "and the image, each word put back, is accepted again");
}

/**
 * @brief A byte changed inside an expert leaves the table accepted, since the table is checked on its own, but fails
 *        that expert's signature, and the image does not open.
 */
LPL_TEST(a_changed_byte_fails_its_segment_s_signature_and_the_image_does_not_open)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Model reopened;
    lpl::infer::ModelSegment segments[kTableRoom];
    lpl::core::u32 size = 0u;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and is written");
    if (image == nullptr)
        return;

    const lpl::core::u32 inside = wordAt(image, entryAt(kSecondExpertEntry) + kEntryOffset) + 1000u;
    image[inside] = static_cast<lpl::core::u8>(image[inside] ^ 0x01u);
    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    const lpl::infer::ModelSegment &first = segments[kFirstExpertEntry - 1u];
    const lpl::infer::ModelSegment &second = segments[kSecondExpertEntry - 1u];

    test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted, "the table is still accepted");
    test.check(lpl::infer::modelSegmentSignature(image, second) != second.signature &&
                   lpl::infer::modelSegmentSignature(image, first) == first.signature,
               "the second expert's signature fails, the first's holds");
    test.check(!reopened.open(arena, image, size), "the image does not open");
    const lpl::infer::ModelReadReport copied = reopened.copy(arena, image, size);
    test.check(copied.verdict == lpl::infer::ModelReadVerdict::SignatureMismatch &&
                   copied.segment == kSecondExpertEntry,
               "and its copy says why: entry 8's signature");
    image[inside] = static_cast<lpl::core::u8>(image[inside] ^ 0x01u);
    test.check(reopened.open(arena, image, size), "and opens once the byte is put back");
}

/**
 * @brief The writer puts the model's rotary table in its segment, a little-endian word per pair, and both readings take
 *        it from there: a made-up table put in its place, its signature put right, is the one a copy and a reading in
 *        place give back, the latter where it lies, and the weights' fold tells that model from the one written.
 */
LPL_TEST(an_image_carries_its_rotary_table_through_a_copy_and_a_reading_in_place)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Model copied;
    lpl::infer::Model viewed;
    lpl::infer::ModelSegment segments[kTableRoom];
    lpl::core::u32 size = 0u;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and is written on a word boundary");
    if (image == nullptr)
        return;

    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    const lpl::infer::ModelSegment rotary = segments[kRotaryEntry - 1u];
    const lpl::infer::RotaryFrequencies written = model.rotaryFrequencies();
    const lpl::core::u32 at = static_cast<lpl::core::u32>(rotary.offset);
    test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                   rotary.role == lpl::infer::ModelSegmentRole::Rotary && rotary.length == kParityPairs * kWordBytes,
               "the image is accepted, its sixth entry a rotary table of a word per pair");
    if (rotary.role != lpl::infer::ModelSegmentRole::Rotary || written.pairs != kParityPairs)
        return;

    bool everyWritten = true;
    for (lpl::core::u32 pair = 0u; everyWritten && pair < kParityPairs; ++pair)
        everyWritten = wordAt(image, at + pair * kWordBytes) == written.turnsPerPosition[pair];
    test.check(everyWritten, "the writer puts each pair's turns in the segment, little-endian");

    for (lpl::core::u32 pair = 0u; pair < kParityPairs; ++pair)
        putWordAt(image, at + pair * kWordBytes, kMadeUpTurnStep * (pair + 1u));
    putWordAt(image, entryAt(kRotaryEntry) + kEntrySignature, foldBytes(image, rotary.offset, rotary.length));

    const bool copyRead = copied.open(arena, image, size);
    const bool viewRead = viewed.view(arena, image, size).verdict == lpl::infer::ModelReadVerdict::Read;
    const lpl::infer::RotaryFrequencies fromCopy = copied.rotaryFrequencies();
    const lpl::infer::RotaryFrequencies fromView = viewed.rotaryFrequencies();
    bool everyRead = copyRead && viewRead && fromCopy.pairs == kParityPairs && fromView.pairs == kParityPairs;
    for (lpl::core::u32 pair = 0u; everyRead && pair < kParityPairs; ++pair)
        everyRead = fromCopy.turnsPerPosition[pair] == kMadeUpTurnStep * (pair + 1u) &&
                    fromView.turnsPerPosition[pair] == kMadeUpTurnStep * (pair + 1u);
    test.check(everyRead, "a copy and a reading in place give back the table the image holds, not the seed's");
    test.check(viewRead && inside(fromView.turnsPerPosition, image, size), "the reading in place where it lies");
    test.check(copyRead && !inside(fromCopy.turnsPerPosition, image, size), "the copy in the arena");
    test.check(copyRead && copied.fold(kFnv1aOffsetBasis) != model.fold(kFnv1aOffsetBasis),
               "and the weights' fold covers the table: the model read differs from the one written by it alone");
}

/**
 * @brief A byte changed in the rotary table leaves the table of segments accepted, but fails the rotary table's
 *        signature: neither a copy nor a reading in place takes the image, and both name entry 6.
 */
LPL_TEST(a_changed_byte_of_the_rotary_table_fails_its_signature)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Model copied;
    lpl::infer::Model viewed;
    lpl::infer::ModelSegment segments[kTableRoom];
    lpl::core::u32 size = 0u;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and is written on a word boundary");
    if (image == nullptr)
        return;

    const lpl::infer::ModelImageInspection before = lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    const lpl::infer::ModelSegment rotary = segments[kRotaryEntry - 1u];
    test.check(before.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                   rotary.role == lpl::infer::ModelSegmentRole::Rotary,
               "its sixth entry is the rotary table");
    if (rotary.role != lpl::infer::ModelSegmentRole::Rotary)
        return;

    const lpl::core::u32 changed = static_cast<lpl::core::u32>(rotary.offset) + kWordBytes + 1u;
    image[changed] = static_cast<lpl::core::u8>(image[changed] ^ 0x01u);
    const lpl::infer::ModelImageInspection after = lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    const bool rotaryFails = lpl::infer::modelSegmentSignature(image, rotary) != rotary.signature;
    const lpl::infer::ModelReadReport copy = copied.copy(arena, image, size);
    const lpl::infer::ModelReadReport view = viewed.view(arena, image, size);
    image[changed] = static_cast<lpl::core::u8>(image[changed] ^ 0x01u);

    test.check(after.verdict == lpl::infer::ModelImageVerdict::Accepted && rotaryFails,
               "the table of segments is still accepted, the rotary table's signature fails");
    test.check(copy.verdict == lpl::infer::ModelReadVerdict::SignatureMismatch && copy.segment == kRotaryEntry &&
                   !copied.ready(),
               "a copy refuses entry 6's signature");
    test.check(view.verdict == lpl::infer::ModelReadVerdict::SignatureMismatch && view.segment == kRotaryEntry &&
                   !viewed.ready(),
               "and so does a reading in place");
}

/**
 * @brief A model of one channel per head has no pair to turn: its rotary table is empty, and it still builds, writes
 *        an image whose rotary segment is empty, and reads back from it by a copy and in place.
 */
LPL_TEST(a_model_of_one_channel_per_head_has_an_empty_rotary_table_and_still_reads_back)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Model copied;
    lpl::infer::Model viewed;
    lpl::infer::ModelSegment segments[kTableRoom];
    lpl::core::u32 size = 0u;

    test.check(lpl::infer::buildParityVocab(arena, vocab), "the canonical vocabulary builds");
    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();
    shape.vocabSize = vocab.size();
    shape.heads = shape.dim;
    shape.keyValueHeads = shape.dim;
    test.check(model.synthesise(arena, shape, vocab, lpl::infer::parityModelSeed()) &&
                   model.rotaryFrequencies().pairs == 0u,
               "a model of 48 heads of one channel builds, its table empty");

    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and its image is written");
    if (image == nullptr)
        return;

    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                   segments[kRotaryEntry - 1u].role == lpl::infer::ModelSegmentRole::Rotary &&
                   segments[kRotaryEntry - 1u].length == 0u,
               "accepted, its rotary segment empty");
    test.check(copied.open(arena, image, size) && copied.rotaryFrequencies().pairs == 0u &&
                   copied.fold(kFnv1aOffsetBasis) == model.fold(kFnv1aOffsetBasis),
               "a copy reads it back");
    test.check(viewed.view(arena, image, size).verdict == lpl::infer::ModelReadVerdict::Read &&
                   viewed.rotaryFrequencies().pairs == 0u &&
                   viewed.fold(kFnv1aOffsetBasis) == model.fold(kFnv1aOffsetBasis),
               "and so does a reading in place");
}

/**
 * @brief Only version 1 is read: the parity image with its version word at 0, at 2 as LplAssistant 0.4.0 and 0.5.0
 *        wrote the same layout, or at 3 is refused at its header as an unknown version, and a version 2 does not open.
 */
LPL_TEST(an_image_of_another_version_is_refused_by_name)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Model reopened;
    lpl::core::u32 size = 0u;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and is written");
    if (image == nullptr)
        return;

    test.check(refusedByName(image, size, kHeaderVersionWord, 0u, "unknown_version") &&
                   refusedByName(image, size, kHeaderVersionWord, 2u, "unknown_version") &&
                   refusedByName(image, size, kHeaderVersionWord, 3u, "unknown_version"),
               "a version 0, 2 or 3 is refused as unknown_version");
    putWordAt(image, kHeaderVersionWord, 2u);
    test.check(!reopened.open(arena, image, size), "and a version 2 does not open");
}

/**
 * @brief An image in the layout LplAssistant 0.3.0 wrote (magic, version 1, the shape, the vocabulary's text bytes,
 *        then each section after the other) says version 1 too, and is refused all the same: its width stands where
 *        the header gives its own size and its heads where the flags must be 0, so its header is bad whatever its
 *        shape, and it is neither copied nor read in place.
 */
LPL_TEST(an_image_in_the_retired_layout_is_refused_as_a_bad_header)
{
    alignas(lpl::infer::kModelViewAlignment) lpl::core::u8 retired[kRetiredImageBytes]{};
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Model model;
    const lpl::infer::ModelConfig shape = parityShape();
    const lpl::core::u32 retiredHeader[] = {
        lpl::infer::kModelMagic, kImageVersion,       shape.dim,       shape.layers,         shape.heads,
        shape.ffnHidden,         shape.contextLength, shape.vocabSize, kMadeUpVocabularyText};
    lpl::core::u32 at = 0u;

    for (const lpl::core::u32 word : retiredHeader)
    {
        putWordAt(retired, at, word);
        at += kWordBytes;
    }
    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(retired, kRetiredImageBytes, nullptr, 0u);
    test.check(inspection.verdict == lpl::infer::ModelImageVerdict::BadHeader && inspection.segment == 0u,
               "it is refused as a bad header");

    const lpl::infer::ModelReadReport copied = model.copy(arena, retired, kRetiredImageBytes);
    const lpl::infer::ModelReadReport viewed = model.view(arena, retired, kRetiredImageBytes);
    test.check(copied.verdict == lpl::infer::ModelReadVerdict::ImageRefused &&
                   copied.image == lpl::infer::ModelImageVerdict::BadHeader &&
                   viewed.verdict == lpl::infer::ModelReadVerdict::ImageRefused &&
                   viewed.image == lpl::infer::ModelImageVerdict::BadHeader && !model.ready(),
               "and it is neither copied nor read in place");
}

/**
 * @brief The parity model's image, its first byte on a word boundary, is read where it lies: the same weights and
 *        tokens, every weight and the vocabulary inside the image, and the arena grows by at least every segment's
 *        bytes less than when the same image is copied.
 */
LPL_TEST(an_image_on_a_word_boundary_is_read_where_it_lies)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Model viewed;
    lpl::infer::Model copied;
    lpl::infer::ModelSegment segments[kTableRoom];
    lpl::core::u32 size = 0u;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    const lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and is written on a word boundary");
    if (image == nullptr)
        return;

    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    lpl::core::u64 segmentBytes = 0u;
    for (lpl::core::u32 index = 0u;
         inspection.verdict == lpl::infer::ModelImageVerdict::Accepted && index < inspection.header.segmentCount;
         ++index)
        segmentBytes += segments[index].length;

    const lpl::core::usize beforeView = arena.used();
    const lpl::infer::ModelReadReport report = viewed.view(arena, image, size);
    const lpl::core::usize viewGrowth = arena.used() - beforeView;
    const lpl::core::usize beforeOpen = arena.used();
    const bool opened = copied.open(arena, image, size);
    const lpl::core::usize openGrowth = arena.used() - beforeOpen;

    test.check(report.verdict == lpl::infer::ModelReadVerdict::Read && report.segment == 0u && viewed.ready(),
               "the image is read");
    test.check(viewed.fold(kFnv1aOffsetBasis) == model.fold(kFnv1aOffsetBasis), "to the same weights, bit for bit");
    test.check(sameVocabulary(viewed.vocab(), model.vocab()), "and the same tokens");
    test.check(everyWeightInside(viewed, image, size), "every matrix, norm and token lies in the image");
    test.check(!everyWeightInside(copied, image, size), "where a copied model's weights do not");
    test.check(opened && segmentBytes > 0u && openGrowth >= viewGrowth + segmentBytes,
               "reading in place claims at least every segment's bytes less than copying");
    test.measure("segment_bytes", static_cast<lpl::core::u32>(segmentBytes));
}

/**
 * @brief An image one, two or three bytes past a word boundary is not read in place, by name, and the model is not
 *        ready; Model::open reads the same bytes to the same weights.
 */
LPL_TEST(an_image_off_a_word_boundary_is_refused_in_place_and_copied_by_open)
{
    lpl::infer::TensorArena arena{kArenasForThreeShiftedCopies * lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    for (lpl::core::u32 shift = 1u; shift < kWordBytes; ++shift)
    {
        lpl::infer::Model viewed;
        lpl::infer::Model copied;
        lpl::core::u32 size = 0u;
        const lpl::core::u8 *const image =
            lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, shift, size);
        test.check(image != nullptr, "the image is written off a word boundary");
        if (image == nullptr)
            return;

        const lpl::infer::ModelReadReport report = viewed.view(arena, image, size);
        test.check(report.verdict == lpl::infer::ModelReadVerdict::Misaligned && !viewed.ready(),
                   "is refused in place as misaligned");
        test.check(copied.open(arena, image, size) && copied.fold(kFnv1aOffsetBasis) == model.fold(kFnv1aOffsetBasis),
                   "and open copies it to the same weights");
    }
}

/**
 * @brief The judgement of a base: a word boundary on a little-endian host is read in place; a byte past it is
 *        misaligned; a big-endian host reads nothing in place, aligned or not.
 */
LPL_TEST(a_base_is_judged_by_its_alignment_and_the_host_s_byte_order)
{
    alignas(4) const lpl::core::u8 bytes[8]{};

    test.check(lpl::infer::judgeModelView(bytes, true) == lpl::infer::ModelReadVerdict::Read,
               "a word boundary on a little-endian host");
    test.check(lpl::infer::judgeModelView(bytes + 1, true) == lpl::infer::ModelReadVerdict::Misaligned &&
                   lpl::infer::judgeModelView(bytes + 2, true) == lpl::infer::ModelReadVerdict::Misaligned &&
                   lpl::infer::judgeModelView(bytes + 3, true) == lpl::infer::ModelReadVerdict::Misaligned,
               "one, two or three bytes past it");
    test.check(lpl::infer::judgeModelView(bytes, false) == lpl::infer::ModelReadVerdict::BigEndianHost &&
                   lpl::infer::judgeModelView(bytes + 1, false) == lpl::infer::ModelReadVerdict::BigEndianHost,
               "a big-endian host, aligned or not");
}

/**
 * @brief Read in place, a flipped lane of the second expert fails its signature by its entry, a bad magic is the
 *        table's refusal by its name, a token past the text whose signature was put right is the vocabulary's
 *        refusal, and an arena too small for the layers' views is exhausted; the model is ready after none of them.
 */
LPL_TEST(every_refusal_of_a_reading_in_place_names_what_it_refuses)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::TensorArena tiny{kTinyArenaBytes};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::core::u32 size = 0u;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    test.check(image != nullptr, "and is written on a word boundary");
    if (image == nullptr)
        return;

    const lpl::core::u32 inExpert = wordAt(image, entryAt(kSecondExpertEntry) + kEntryOffset) + 1000u;
    image[inExpert] = static_cast<lpl::core::u8>(image[inExpert] ^ 0x01u);
    lpl::infer::Model flipped;
    const lpl::infer::ModelReadReport mismatch = flipped.view(arena, image, size);
    image[inExpert] = static_cast<lpl::core::u8>(image[inExpert] ^ 0x01u);
    test.check(mismatch.verdict == lpl::infer::ModelReadVerdict::SignatureMismatch &&
                   mismatch.segment == kSecondExpertEntry && !flipped.ready(),
               "a flipped lane of the second expert fails entry 8's signature");

    const lpl::core::u32 magic = wordAt(image, 0u);
    putWordAt(image, 0u, 0x4D4C504Du);
    lpl::infer::Model unmarked;
    const lpl::infer::ModelReadReport refused = unmarked.view(arena, image, size);
    putWordAt(image, 0u, magic);
    test.check(refused.verdict == lpl::infer::ModelReadVerdict::ImageRefused &&
                   refused.image == lpl::infer::ModelImageVerdict::BadMagic && !unmarked.ready(),
               "a magic not LPLM is the table's refusal, bad_magic");

    const lpl::core::u32 vocabularyOffset = wordAt(image, entryAt(1u) + kEntryOffset);
    const lpl::core::u32 vocabularyLength = wordAt(image, entryAt(1u) + kEntryLength);
    const lpl::core::u32 firstTokenOffset = wordAt(image, vocabularyOffset);
    const lpl::core::u32 firstTokenLength = wordAt(image, vocabularyOffset + kVocabularyEntryLength);
    const lpl::core::u32 signature = wordAt(image, entryAt(1u) + kEntrySignature);
    putWordAt(image, vocabularyOffset, model.vocab().blobBytes() - 1u);
    putWordAt(image, vocabularyOffset + kVocabularyEntryLength, kTokenPastTheText);
    putWordAt(image, entryAt(1u) + kEntrySignature, foldBytes(image, vocabularyOffset, vocabularyLength));
    lpl::infer::Model untokened;
    const lpl::infer::ModelReadReport vocabulary = untokened.view(arena, image, size);
    putWordAt(image, vocabularyOffset, firstTokenOffset);
    putWordAt(image, vocabularyOffset + kVocabularyEntryLength, firstTokenLength);
    putWordAt(image, entryAt(1u) + kEntrySignature, signature);
    test.check(
        vocabulary.verdict == lpl::infer::ModelReadVerdict::VocabularyRefused && vocabulary.segment == 1u &&
            !untokened.ready(),
        "a two-byte token starting on the text's last byte, its signature put right, is the vocabulary's refusal");

    lpl::infer::Model cramped;
    test.check(cramped.view(tiny, image, size).verdict == lpl::infer::ModelReadVerdict::ArenaExhausted &&
                   !cramped.ready(),
               "16 bytes of arena are exhausted by the layers' views");

    lpl::infer::TensorArena tight{sizeof(lpl::infer::LayerWeights) * model.config().layers};
    lpl::infer::Model unindexed;
    const lpl::infer::ModelReadReport index = unindexed.view(tight, image, size);
    test.check(index.verdict == lpl::infer::ModelReadVerdict::ArenaExhausted && index.segment == 1u &&
                   !unindexed.ready(),
               "an arena that holds the layers' views but not the vocabulary's index is exhausted by entry 1");

    lpl::infer::Model restored;
    test.check(restored.view(arena, image, size).verdict == lpl::infer::ModelReadVerdict::Read,
               "and the image, each byte put back, is read again");
    image[inExpert] = static_cast<lpl::core::u8>(image[inExpert] ^ 0x01u);
    test.check(restored.view(arena, image, size).verdict == lpl::infer::ModelReadVerdict::SignatureMismatch &&
                   !restored.ready(),
               "a model that read it, refused another image, is not ready");
    image[inExpert] = static_cast<lpl::core::u8>(image[inExpert] ^ 0x01u);
}

/**
 * @brief Each verdict of a reading has its own name, and a value outside them is "unknown".
 */
LPL_TEST(every_verdict_of_a_reading_has_its_name)
{
    const auto name = [](lpl::infer::ModelReadVerdict verdict) { return lpl::infer::modelReadVerdictName(verdict); };

    test.check(sameText(name(lpl::infer::ModelReadVerdict::Read), "read") &&
                   sameText(name(lpl::infer::ModelReadVerdict::Misaligned), "misaligned") &&
                   sameText(name(lpl::infer::ModelReadVerdict::BigEndianHost), "big_endian_host") &&
                   sameText(name(lpl::infer::ModelReadVerdict::ImageRefused), "image_refused") &&
                   sameText(name(lpl::infer::ModelReadVerdict::SignatureMismatch), "signature_mismatch") &&
                   sameText(name(lpl::infer::ModelReadVerdict::VocabularyRefused), "vocabulary_refused") &&
                   sameText(name(lpl::infer::ModelReadVerdict::ArenaExhausted), "arena_exhausted") &&
                   sameText(name(lpl::infer::ModelReadVerdict::MergesRefused), "merges_refused") &&
                   sameText(name(lpl::infer::ModelReadVerdict::ChatTokensMissing), "chat_tokens_missing"),
               "each verdict its name");
    test.check(sameText(name(static_cast<lpl::infer::ModelReadVerdict>(99u)), "unknown"), "and an unknown value");
}

/**
 * @brief A model's chat format goes through its image: the header's word at byte 76 says ChatML, and a copy and a
 *        reading in place find the tokens of its turns again in the vocabulary; the parity model's image says Plain.
 */
LPL_TEST(an_image_carries_its_chat_format_through_a_copy_and_a_reading_in_place)
{
    lpl::infer::TensorArena arena{kChatArenaBytes};
    lpl::infer::Vocab chatVocab;
    lpl::infer::Vocab parityVocab;
    lpl::infer::Model chat;
    lpl::infer::Model parity;
    lpl::core::u32 chatSize = 0u;
    lpl::core::u32 paritySize = 0u;

    test.check(lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens, kTurnTokenCount, chatVocab) &&
                   lpl::infer::tests::synthesiseParityModelOver(arena, chatVocab, lpl::infer::ChatFormat::ChatMl,
                                                                lpl::infer::parityModelConfig().contextLength, chat) &&
                   synthesiseParityModel(arena, parityVocab, parity),
               "a ChatML model over the parity vocabulary and three turn tokens builds, and so does the parity model");
    lpl::core::u8 *const chatImage =
        lpl::infer::tests::writeModelImage(arena, chat, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, chatSize);
    lpl::core::u8 *const parityImage =
        lpl::infer::tests::writeModelImage(arena, parity, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, paritySize);
    if (!test.check(chatImage != nullptr && parityImage != nullptr, "and both are written"))
        return;

    const lpl::infer::ModelImageInspection chatInspection =
        lpl::infer::inspectModelImage(chatImage, chatSize, nullptr, 0u);
    const lpl::infer::ModelImageInspection parityInspection =
        lpl::infer::inspectModelImage(parityImage, paritySize, nullptr, 0u);
    lpl::infer::Model copied;
    lpl::infer::Model viewed;
    lpl::infer::Model parityViewed;
    const bool copiedRead = copied.copy(arena, chatImage, chatSize).verdict == lpl::infer::ModelReadVerdict::Read;
    const bool viewedRead = viewed.view(arena, chatImage, chatSize).verdict == lpl::infer::ModelReadVerdict::Read;

    test.check(chatInspection.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                   chatInspection.header.chatFormat == lpl::infer::ChatFormat::ChatMl &&
                   wordAt(chatImage, lpl::infer::tests::kHeaderChatFormatByte) == kChatMlWord,
               "the ChatML image's header holds 1 at byte 76");
    test.check(copiedRead && hasTheAppendedTurnTokens(copied.chatTokens()),
               "copied, the model has a ChatML turn: <|im_start|> 122, <|im_end|> 123, <|endoftext|> 121");
    test.check(viewedRead && hasTheAppendedTurnTokens(viewed.chatTokens()), "and read in place, it has them too");
    test.check(parityInspection.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                   parityInspection.header.chatFormat == lpl::infer::ChatFormat::Plain &&
                   wordAt(parityImage, lpl::infer::tests::kHeaderChatFormatByte) == 0u &&
                   parityViewed.view(arena, parityImage, paritySize).verdict == lpl::infer::ModelReadVerdict::Read &&
                   parityViewed.chatTokens().format == lpl::infer::ChatFormat::Plain &&
                   parityViewed.chatTokens().turnEnd == lpl::infer::kNoToken,
               "the parity model's image says Plain, and the model read from it has no turn");
    test.measure("chat_image_bytes", chatSize);
}

/**
 * @brief A chat format this reader does not know is refused at the header by its name, unknown_chat_format, and the
 *        model is not read.
 */
LPL_TEST(a_chat_format_this_reader_does_not_know_is_refused_by_name)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::core::u32 size = 0u;

    test.check(synthesiseParityModel(arena, vocab, model), "the parity model builds");
    lpl::core::u8 *const image =
        lpl::infer::tests::writeModelImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, size);
    if (!test.check(image != nullptr, "and is written"))
        return;

    test.check(refusedByName(image, size, lpl::infer::tests::kHeaderChatFormatByte, 2u, "unknown_chat_format") &&
                   refusedWith(image, size, lpl::infer::tests::kHeaderChatFormatByte, 0xFFFFFFFFu,
                               lpl::infer::ModelImageVerdict::UnknownChatFormat, 0u),
               "a chat format of 2, or of 0xFFFFFFFF, is refused at the header as unknown_chat_format");

    lpl::infer::tests::putChatFormatWord(image, 2u);
    lpl::infer::Model refused;
    const lpl::infer::ModelReadReport report = refused.view(arena, image, size);
    lpl::infer::tests::putChatFormatWord(image, 0u);
    test.check(report.verdict == lpl::infer::ModelReadVerdict::ImageRefused &&
                   report.image == lpl::infer::ModelImageVerdict::UnknownChatFormat && !refused.ready(),
               "and the model is not read");
}

/**
 * @brief An image whose format needs a token its vocabulary lacks is refused by name, chat_tokens_missing, naming the
 *        vocabulary's entry: the parity vocabulary read as ChatML, or one that has <|im_start|> and no <|im_end|>. One
 *        that lacks <|endoftext|> alone is read.
 */
LPL_TEST(a_vocabulary_without_the_tokens_of_its_format_is_refused_by_name)
{
    lpl::infer::TensorArena arena{kChatArenaBytes};
    lpl::infer::Vocab parityVocab;
    lpl::infer::Vocab turnStartOnly;
    lpl::infer::Vocab withoutEndOfText;
    lpl::infer::Model parity;
    lpl::infer::Model halfTurn;
    lpl::infer::Model noEndOfText;
    lpl::core::u32 paritySize = 0u;
    lpl::core::u32 halfSize = 0u;
    lpl::core::u32 noEndSize = 0u;

    test.check(
        synthesiseParityModel(arena, parityVocab, parity) &&
            lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens + 1u, 1u, turnStartOnly) &&
            lpl::infer::tests::synthesiseParityModelOver(arena, turnStartOnly, lpl::infer::ChatFormat::Plain,
                                                         lpl::infer::parityModelConfig().contextLength, halfTurn) &&
            lpl::infer::tests::buildParityVocabularyWith(arena, kTurnTokens + 1u, 2u, withoutEndOfText) &&
            lpl::infer::tests::synthesiseParityModelOver(arena, withoutEndOfText, lpl::infer::ChatFormat::ChatMl,
                                                         lpl::infer::parityModelConfig().contextLength, noEndOfText),
        "the parity model, a Plain model with <|im_start|> alone and a ChatML one without <|endoftext|> build");
    lpl::core::u8 *const parityImage =
        lpl::infer::tests::writeModelImage(arena, parity, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, paritySize);
    lpl::core::u8 *const halfImage =
        lpl::infer::tests::writeModelImage(arena, halfTurn, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, halfSize);
    lpl::core::u8 *const noEndImage = lpl::infer::tests::writeModelImage(
        arena, noEndOfText, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, noEndSize);
    if (!test.check(parityImage != nullptr && halfImage != nullptr && noEndImage != nullptr, "and are written"))
        return;

    lpl::infer::tests::putChatFormatWord(parityImage, kChatMlWord);
    lpl::infer::tests::putChatFormatWord(halfImage, kChatMlWord);
    lpl::infer::Model viewed;
    lpl::infer::Model copied;
    lpl::infer::Model half;
    lpl::infer::Model whole;
    const lpl::infer::ModelReadReport inPlace = viewed.view(arena, parityImage, paritySize);
    const lpl::infer::ModelReadReport copy = copied.copy(arena, parityImage, paritySize);
    const lpl::infer::ModelReadReport halfReport = half.view(arena, halfImage, halfSize);
    const lpl::infer::ModelReadReport wholeReport = whole.view(arena, noEndImage, noEndSize);

    test.check(inPlace.verdict == lpl::infer::ModelReadVerdict::ChatTokensMissing && inPlace.segment == 1u &&
                   !viewed.ready() &&
                   sameText(lpl::infer::modelReadVerdictName(inPlace.verdict), "chat_tokens_missing"),
               "the parity vocabulary read as ChatML in place is refused as chat_tokens_missing, naming entry 1");
    test.check(copy.verdict == lpl::infer::ModelReadVerdict::ChatTokensMissing && copy.segment == 1u && !copied.ready(),
               "and so is its copy");
    test.check(halfReport.verdict == lpl::infer::ModelReadVerdict::ChatTokensMissing && halfReport.segment == 1u &&
                   !half.ready(),
               "a vocabulary with <|im_start|> and no <|im_end|> is refused too");
    test.check(wholeReport.verdict == lpl::infer::ModelReadVerdict::Read && whole.chatTokens().turnStart == 121u &&
                   whole.chatTokens().turnEnd == 122u && whole.chatTokens().endOfText == lpl::infer::kNoToken,
               "and one without <|endoftext|> is read, its turn without an end of text");

    lpl::infer::Model derived;
    test.check(!lpl::infer::tests::synthesiseParityModelOver(arena, turnStartOnly, lpl::infer::ChatFormat::ChatMl,
                                                             lpl::infer::parityModelConfig().contextLength, derived) &&
                   !derived.ready(),
               "nor can a model a seed derives over a vocabulary without <|im_end|> be ChatML");
}
