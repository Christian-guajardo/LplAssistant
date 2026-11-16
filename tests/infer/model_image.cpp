#include <lpl/infer/ModelImage.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(model_image);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;
constexpr lpl::core::u8 kDirtyByte = 0xA5u;
constexpr lpl::core::u32 kParityVocabulary = 121u;
constexpr lpl::core::u32 kMadeUpVocabularyText = 300u;
constexpr lpl::core::u32 kParitySegments = 7u;
constexpr lpl::core::u32 kTableRoom = 16u;
constexpr lpl::core::u32 kEntryRole = 0u;
constexpr lpl::core::u32 kEntryOffset = 16u;
constexpr lpl::core::u32 kEntryLength = 24u;
constexpr lpl::core::u32 kEntryReserved = 36u;
constexpr lpl::core::u32 kHeaderBytesWord = 8u;
constexpr lpl::core::u32 kHeaderAlignmentWord = 12u;
constexpr lpl::core::u32 kHeaderFlagsWord = 16u;
constexpr lpl::core::u32 kHeaderCountWord = 20u;
constexpr lpl::core::u32 kHeaderImageBytesWord = 24u;
constexpr lpl::core::u32 kHeaderHeadsWord = 40u;
constexpr lpl::core::u32 kHeaderReservedWord = 60u;

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
 * @brief Writes @p model's version 2 image at @p alignmentLog2 into @p arena.
 * @return The image, or nullptr when the writer refused or the arena is short; @p size receives its bytes.
 */
[[nodiscard]] lpl::core::u8 *writeImage(lpl::infer::TensorArena &arena, const lpl::infer::Model &model,
                                        lpl::core::u32 alignmentLog2, lpl::core::u32 &size)
{
    size = model.blobBytes(alignmentLog2);
    lpl::core::u8 *const image = size != 0u ? arena.claim<lpl::core::u8>(size) : nullptr;
    if (image == nullptr)
        return nullptr;

    for (lpl::core::u32 i = 0u; i < size; ++i)
        image[i] = kDirtyByte;
    return model.writeBlob(image, size, alignmentLog2) == size ? image : nullptr;
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
 * @brief Whether @p image, its word at @p offset set to @p value, is refused with @p verdict naming @p segment; the
 *        word is put back after.
 */
[[nodiscard]] bool refusedWith(lpl::core::u8 *image, lpl::core::u32 size, lpl::core::u32 offset, lpl::core::u32 value,
                               lpl::infer::ModelImageVerdict verdict, lpl::core::u32 segment)
{
    lpl::infer::ModelSegment segments[kTableRoom];
    const lpl::core::u32 kept = wordAt(image, offset);

    putWordAt(image, offset, value);
    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    putWordAt(image, offset, kept);
    return inspection.verdict == verdict && inspection.segment == segment;
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

/**
 * @brief Appends the little-endian word @p value at @p cursor of a hand-built image.
 */
void appendWord(lpl::core::u8 *bytes, lpl::core::u32 &cursor, lpl::core::u32 value)
{
    putWordAt(bytes, cursor, value);
    cursor += 4u;
}

/**
 * @brief Appends @p rows quantised blocks of scale @p scale, lanes counting up from @p first, to a hand-built image.
 */
void appendBlocks(lpl::core::u8 *bytes, lpl::core::u32 &cursor, lpl::core::u32 rows, lpl::core::u32 scale,
                  lpl::core::u32 first)
{
    for (lpl::core::u32 row = 0u; row < rows; ++row)
    {
        appendWord(bytes, cursor, scale + row);
        for (lpl::core::u32 lane = 0u; lane < lpl::infer::kQuantBlockLanes; ++lane)
            bytes[cursor++] = static_cast<lpl::core::u8>(first + lane);
    }
}

} // namespace

/**
 * @brief A dense model's image holds its residents after the table, 64-byte aligned, then one expert per layer on a
 *        4 KiB boundary.
 *
 * @details Worked by hand for the parity shape (dim 48, 2 layers, feed-forward 96) with 121 tokens and 300 bytes of
 *          text, a quantised block 36 bytes: the table ends at 64 + 7 * 40 = 344; the vocabulary (8 * 121 + 300 =
 *          1268) at 384; the embedding (121 rows of 2 blocks, 8712) at 1664; each attention (2 norms of 48 words and
 *          4 matrices of 48 rows of 2 blocks, 14208) at 10432 and 24640; the final norm (192) at 38848, ending at
 *          39040; each expert (gate and up of 96 rows of 2 blocks, down of 48 rows of 3, 19008) at 40960 and 61440,
 *          the image ending at 80448.
 */
LPL_TEST(a_dense_model_lays_out_its_residents_then_one_expert_per_layer)
{
    lpl::infer::ModelSegment segments[kTableRoom];
    const lpl::core::u64 bytes = lpl::infer::layOutModelImage(
        parityShape(), kMadeUpVocabularyText, lpl::infer::kModelImageDefaultAlignmentLog2, segments, kTableRoom);
    const lpl::infer::ModelSegmentResidency resident = lpl::infer::ModelSegmentResidency::Resident;
    const lpl::infer::ModelSegmentResidency shared = lpl::infer::ModelSegmentResidency::SharedExpert;

    test.check(bytes == 80448u, "the image is 80448 bytes");
    test.check(isSegment(segments[0], lpl::infer::ModelSegmentRole::Vocabulary, resident, 0u, 0u) &&
                   isSegment(segments[1], lpl::infer::ModelSegmentRole::Embedding, resident, 0u, 0u) &&
                   isSegment(segments[2], lpl::infer::ModelSegmentRole::Attention, resident, 1u, 0u) &&
                   isSegment(segments[3], lpl::infer::ModelSegmentRole::Attention, resident, 2u, 0u) &&
                   isSegment(segments[4], lpl::infer::ModelSegmentRole::FinalNorm, resident, 0u, 0u) &&
                   isSegment(segments[5], lpl::infer::ModelSegmentRole::FeedForward, shared, 1u, 1u) &&
                   isSegment(segments[6], lpl::infer::ModelSegmentRole::FeedForward, shared, 2u, 1u),
               "the residents in order, then each layer's one shared expert");
    test.check(segments[0].offset == 384u && segments[1].offset == 1664u && segments[2].offset == 10432u &&
                   segments[3].offset == 24640u && segments[4].offset == 38848u,
               "each resident on the next 64 bytes after the one before, the first after the table");
    test.check(segments[0].length == 1268u && segments[1].length == 8712u && segments[2].length == 14208u &&
                   segments[3].length == 14208u && segments[4].length == 192u && segments[5].length == 19008u &&
                   segments[6].length == 19008u,
               "each length the shape gives its role");
    test.check(segments[5].offset == 40960u && segments[6].offset == 61440u, "each expert on the next 4 KiB boundary");
    test.measure("image_bytes", static_cast<lpl::core::u32>(bytes));
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
    const lpl::core::u64 at2Mib =
        lpl::infer::layOutModelImage(parityShape(), kMadeUpVocabularyText, 21u, segments, kTableRoom);

    test.check(at2Mib == 4194304u + 19008u && segments[5].offset == 2097152u && segments[6].offset == 4194304u,
               "at 2^21, the experts at 2 MiB and 4 MiB");
    test.check(lpl::infer::layOutModelImage(parityShape(), kMadeUpVocabularyText, 11u, segments, kTableRoom) == 0u &&
                   lpl::infer::layOutModelImage(parityShape(), kMadeUpVocabularyText, 23u, segments, kTableRoom) == 0u,
               "2^11 and 2^23 are refused");
    test.check(lpl::infer::layOutModelImage(badHeads, kMadeUpVocabularyText, 12u, segments, kTableRoom) == 0u &&
                   lpl::infer::layOutModelImage(parityShape(), 0u, 12u, segments, kTableRoom) == 0u,
               "so are 5 heads over a width of 48, and no vocabulary text");
    test.check(lpl::infer::layOutModelImage(parityShape(), kMadeUpVocabularyText, 12u, segments, 6u) == 0u,
               "and a table of 7 entries into room for 6");
}

/**
 * @brief The parity model written as a version 2 image is accepted, each segment's signature holds, and it reads back
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
    lpl::core::u8 *const image = writeImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, size);
    test.check(image != nullptr && lpl::infer::modelImageVersion(image, size) == lpl::infer::kModelImageVersion2,
               "it is written as a version 2 image");
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
               "at 4 KiB, with 7 segments, as long as written");
    test.check(everySegmentSigned, "and each segment's bytes give its signature");
    test.check(segments[4].signature == foldBytes(image, segments[4].offset, segments[4].length),
               "the final norm's signature is the FNV-1a of its bytes, folded apart");
    test.check(zeroBetween(image, segments[4].offset + segments[4].length, segments[5].offset),
               "the padding before the first expert is zero");
    test.check(reopened.open(arena, image, size) && reopened.fold(kFnv1aOffsetBasis) == model.fold(kFnv1aOffsetBasis),
               "it reads back to the same weights, bit for bit");
    test.check(sameVocabulary(reopened.vocab(), model.vocab()), "and the same tokens, which the weights do not fold");
    test.measure("image_bytes", size);
    test.measure("first_expert_offset", static_cast<lpl::core::u32>(segments[5].offset));
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
    lpl::core::u8 *const image = writeImage(arena, model, 16u, size);
    test.check(image != nullptr, "it is written at 2^16");
    if (image == nullptr)
        return;

    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image, size, segments, kTableRoom);
    test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted &&
                   inspection.header.alignmentLog2 == 16u && segments[5].offset % 65536u == 0u &&
                   segments[6].offset % 65536u == 0u,
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
    lpl::core::u8 *const image = writeImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, size);
    test.check(image != nullptr, "and is written");
    if (image == nullptr)
        return;

    const lpl::core::u32 firstExpertOffset = wordAt(image, entryAt(6u) + kEntryOffset);
    const lpl::core::u32 embeddingOffset = wordAt(image, entryAt(2u) + kEntryOffset);
    const lpl::core::u32 lastExpertOffset = wordAt(image, entryAt(7u) + kEntryOffset);
    const lpl::core::u32 lastExpertLength = wordAt(image, entryAt(7u) + kEntryLength);

    test.check(lpl::infer::inspectModelImage(image, 63u, segments, kTableRoom).verdict ==
                   lpl::infer::ModelImageVerdict::TooShort,
               "an image shorter than its header");
    test.check(refusedWith(image, size, 0u, 0x4D4C504Du, lpl::infer::ModelImageVerdict::BadMagic, 0u),
               "a magic not LPLM");
    test.check(refusedWith(image, size, 4u, 3u, lpl::infer::ModelImageVerdict::NotVersion2, 0u), "a version 3");
    test.check(refusedWith(image, size, kHeaderBytesWord, 60u, lpl::infer::ModelImageVerdict::BadHeader, 0u) &&
                   refusedWith(image, size, kHeaderFlagsWord, 1u, lpl::infer::ModelImageVerdict::BadHeader, 0u) &&
                   refusedWith(image, size, kHeaderReservedWord, 1u, lpl::infer::ModelImageVerdict::BadHeader, 0u),
               "header bytes of 60, a flag set, a reserved word set");
    test.check(refusedWith(image, size, kHeaderAlignmentWord, 11u, lpl::infer::ModelImageVerdict::BadAlignment, 0u) &&
                   refusedWith(image, size, kHeaderAlignmentWord, 23u, lpl::infer::ModelImageVerdict::BadAlignment, 0u),
               "an alignment of 2^11 or 2^23");
    test.check(refusedWith(image, size, kHeaderHeadsWord, 5u, lpl::infer::ModelImageVerdict::BadShape, 0u),
               "5 heads over a width of 48");
    test.check(
        refusedWith(image, size, kHeaderImageBytesWord, size + 1u, lpl::infer::ModelImageVerdict::BadImageBytes, 0u),
        "a header claiming one byte more than the image");
    test.check(refusedWith(image, size, kHeaderCountWord, 6u, lpl::infer::ModelImageVerdict::BadSegmentCount, 0u) &&
                   refusedWith(image, size, kHeaderCountWord, kTableRoom + 1u,
                               lpl::infer::ModelImageVerdict::BadSegmentCount, 0u),
               "6 segments for 2 layers, or more than the room given");
    test.check(
        refusedWith(image, size, entryAt(1u) + kEntryRole, 2u, lpl::infer::ModelImageVerdict::SegmentUnexpected, 1u) &&
            refusedWith(image, size, entryAt(3u) + kEntryReserved, 1u, lpl::infer::ModelImageVerdict::SegmentUnexpected,
                        3u),
        "an embedding where the vocabulary goes, an entry's reserved word set");
    test.check(refusedWith(image, size, entryAt(7u) + kEntryLength, lastExpertLength + 1u,
                           lpl::infer::ModelImageVerdict::SegmentWrongLength, 7u),
               "an expert one byte longer than its shape");
    test.check(refusedWith(image, size, entryAt(6u) + kEntryOffset, firstExpertOffset + 64u,
                           lpl::infer::ModelImageVerdict::SegmentMisaligned, 6u),
               "an expert 64 bytes off its boundary");
    test.check(refusedWith(image, size, entryAt(2u) + kEntryOffset, embeddingOffset - 64u,
                           lpl::infer::ModelImageVerdict::SegmentOverlaps, 2u),
               "an embedding starting inside the vocabulary");
    test.check(refusedWith(image, size, entryAt(7u) + kEntryOffset, lastExpertOffset + 4096u,
                           lpl::infer::ModelImageVerdict::SegmentOutside, 7u),
               "the last expert one page further, past the image");
    test.check(refusedWith(image, size, entryAt(7u) + kEntryOffset, ((size / 4096u) + 2u) * 4096u,
                           lpl::infer::ModelImageVerdict::SegmentOutside, 7u),
               "the last expert starting past the image's end");
    test.check(lpl::infer::inspectModelImage(image, size, segments, kParitySegments - 1u).verdict ==
                   lpl::infer::ModelImageVerdict::BadSegmentCount,
               "a table of 7 entries into room for 6");
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
    lpl::core::u8 *const image = writeImage(arena, model, lpl::infer::kModelImageDefaultAlignmentLog2, size);
    test.check(image != nullptr, "and is written");
    if (image == nullptr)
        return;

    const lpl::core::u32 inside = wordAt(image, entryAt(7u) + kEntryOffset) + 1000u;
    image[inside] = static_cast<lpl::core::u8>(image[inside] ^ 0x01u);
    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(image, size, segments, kTableRoom);

    test.check(inspection.verdict == lpl::infer::ModelImageVerdict::Accepted, "the table is still accepted");
    test.check(lpl::infer::modelSegmentSignature(image, segments[6]) != segments[6].signature &&
                   lpl::infer::modelSegmentSignature(image, segments[5]) == segments[5].signature,
               "the second expert's signature fails, the first's holds");
    test.check(!reopened.open(arena, image, size), "the image does not open");
    image[inside] = static_cast<lpl::core::u8>(image[inside] ^ 0x01u);
    test.check(reopened.open(arena, image, size), "and opens once the byte is put back");
}

/**
 * @brief A version 1 image, built by hand from its layout rather than by the writer, still opens to the weights its
 *        bytes give, and one byte short it is refused.
 *
 * @details dim 2, 1 layer, 1 head, feed-forward 2, context 1, one token "a": 9 header words, one entry, the text padded
 *          to 4 bytes, the embedding's block, two norms, four attention matrices and gate and up of 2 rows, down of 2
 *          rows, the final norm: 612 bytes.
 */
LPL_TEST(a_hand_built_version_1_image_still_opens)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::core::u8 *const bytes = arena.claim<lpl::core::u8>(612u);
    lpl::infer::Model model;
    lpl::infer::Model shortened;
    lpl::core::u32 cursor = 0u;

    test.check(bytes != nullptr, "room for the image");
    if (bytes == nullptr)
        return;
    static const lpl::core::u32 header[] = {
        lpl::infer::kModelMagic, lpl::infer::kModelImageVersion1, 2u, 1u, 1u, 2u, 1u, 1u, 1u, 0u, 1u};
    static const lpl::core::u32 norms[] = {0x20000u, 0x20001u, 0x30000u, 0x30001u};
    static const lpl::core::u32 finalNorm[] = {0x50000u, 0x50001u};

    for (const lpl::core::u32 word : header)
        appendWord(bytes, cursor, word);
    appendWord(bytes, cursor, static_cast<lpl::core::u32>('a'));
    appendBlocks(bytes, cursor, 1u, 0x10000u, 1u);
    for (const lpl::core::u32 word : norms)
        appendWord(bytes, cursor, word);
    for (lpl::core::u32 matrix = 0u; matrix < 7u; ++matrix)
        appendBlocks(bytes, cursor, 2u, 0x40000u + 0x100u * matrix, 10u * matrix);
    for (const lpl::core::u32 word : finalNorm)
        appendWord(bytes, cursor, word);

    test.check(cursor == 612u, "the hand-built image is 612 bytes");
    test.check(lpl::infer::modelImageVersion(bytes, 612u) == lpl::infer::kModelImageVersion1, "of version 1");
    test.check(model.open(arena, bytes, 612u), "it opens");
    if (!model.ready())
        return;
    test.check(model.config().dim == 2u && model.config().layers == 1u && model.config().ffnHidden == 2u &&
                   model.config().vocabSize == 1u,
               "to the shape its header gives");
    test.check(model.embedding().blocks[0].scaleRaw == 0x10000 && model.embedding().blocks[0].lanes[31] == 32,
               "its embedding's block, scale and last lane");
    test.check(model.layer(0u).attentionNorm.values[1].raw() == 0x20001 &&
                   model.layer(0u).feedForwardNorm.values[0].raw() == 0x30000,
               "its two norms");
    test.check(model.layer(0u).query.blocks[0].scaleRaw == 0x40000 &&
                   model.layer(0u).down.blocks[1].scaleRaw == 0x40601 && model.layer(0u).down.blocks[1].lanes[0] == 60,
               "its first and last matrices, in the order written");
    test.check(model.finalNorm().values[1].raw() == 0x50001, "its final norm");
    test.check(!shortened.open(arena, bytes, 611u), "one byte short, it is refused");
}
