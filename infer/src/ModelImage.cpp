#include <lpl/infer/ModelImage.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::infer {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/** Bytes one quantised block occupies in an image: the scale word, then the lanes. */
constexpr core::u32 kBlockImageBytes = 4u + kQuantBlockLanes;

/** Bytes a vocabulary entry occupies: its text's offset and length. */
constexpr core::u32 kVocabEntryBytes = 8u;

/** Segments a dense model's image holds besides two per layer: the vocabulary, the embedding, the final norm. */
constexpr core::u32 kSegmentsBesidesLayers = 3u;

/**
 * @name Byte positions inside the header
 * @{
 */
constexpr core::u32 kHeaderMagic = 0u;
constexpr core::u32 kHeaderVersion = 4u;
constexpr core::u32 kHeaderBytes = 8u;
constexpr core::u32 kHeaderAlignment = 12u;
constexpr core::u32 kHeaderFlags = 16u;
constexpr core::u32 kHeaderSegmentCount = 20u;
constexpr core::u32 kHeaderImageBytes = 24u;
constexpr core::u32 kHeaderDim = 32u;
constexpr core::u32 kHeaderLayers = 36u;
constexpr core::u32 kHeaderHeads = 40u;
constexpr core::u32 kHeaderFeedForward = 44u;
constexpr core::u32 kHeaderContext = 48u;
constexpr core::u32 kHeaderVocabSize = 52u;
constexpr core::u32 kHeaderVocabText = 56u;
constexpr core::u32 kHeaderReserved = 60u;
/** @} */

/**
 * @name Byte positions inside a table entry
 * @{
 */
constexpr core::u32 kEntryRole = 0u;
constexpr core::u32 kEntryResidency = 4u;
constexpr core::u32 kEntryLayer = 8u;
constexpr core::u32 kEntryExpert = 12u;
constexpr core::u32 kEntryOffset = 16u;
constexpr core::u32 kEntryLength = 24u;
constexpr core::u32 kEntrySignature = 32u;
constexpr core::u32 kEntryReserved = 36u;
/** @} */

/**
 * @brief The little-endian word at @p at of @p bytes.
 */
core::u32 wordAt(const core::u8 *bytes, core::u64 at) noexcept
{
    return static_cast<core::u32>(bytes[at]) | (static_cast<core::u32>(bytes[at + 1u]) << 8) |
           (static_cast<core::u32>(bytes[at + 2u]) << 16) | (static_cast<core::u32>(bytes[at + 3u]) << 24);
}

/**
 * @brief The little-endian 64-bit word at @p at of @p bytes.
 */
core::u64 doubleWordAt(const core::u8 *bytes, core::u64 at) noexcept
{
    return static_cast<core::u64>(wordAt(bytes, at)) | (static_cast<core::u64>(wordAt(bytes, at + 4u)) << 32);
}

/**
 * @brief Writes @p value little-endian at @p at of @p bytes.
 */
void putWordAt(core::u8 *bytes, core::u64 at, core::u32 value) noexcept
{
    for (core::u32 i = 0u; i < 4u; ++i)
        bytes[at + i] = static_cast<core::u8>((value >> (8u * i)) & 0xFFu);
}

/**
 * @brief Writes the 64-bit @p value little-endian at @p at of @p bytes.
 */
void putDoubleWordAt(core::u8 *bytes, core::u64 at, core::u64 value) noexcept
{
    putWordAt(bytes, at, static_cast<core::u32>(value & 0xFFFFFFFFu));
    putWordAt(bytes, at + 4u, static_cast<core::u32>(value >> 32));
}

/**
 * @brief Whether @p alignmentLog2 is one an image may ask for.
 */
bool alignmentAccepted(core::u32 alignmentLog2) noexcept
{
    return alignmentLog2 >= kModelImageLeastAlignmentLog2 && alignmentLog2 <= kModelImageMostAlignmentLog2;
}

/**
 * @brief The zero-based @p index -th entry a dense model of @p shape holds, its offset and signature at 0: the
 *        vocabulary, the embedding, each layer's attention, the final norm, then each layer's expert.
 */
ModelSegment expectedSegment(const ModelConfig &shape, core::u32 vocabTextBytes, core::u32 index) noexcept
{
    const core::u64 dim = shape.dim;
    ModelSegment segment{};

    if (index == 0u)
    {
        segment.role = ModelSegmentRole::Vocabulary;
        segment.length = static_cast<core::u64>(shape.vocabSize) * kVocabEntryBytes + vocabTextBytes;
    }
    else if (index == 1u)
    {
        segment.role = ModelSegmentRole::Embedding;
        segment.length = modelMatrixImageBytes(shape.vocabSize, shape.dim);
    }
    else if (index < shape.layers + 2u)
    {
        segment.role = ModelSegmentRole::Attention;
        segment.layer = index - 1u;
        segment.length = 2u * dim * 4u + 4u * modelMatrixImageBytes(shape.dim, shape.dim);
    }
    else if (index == shape.layers + 2u)
    {
        segment.role = ModelSegmentRole::FinalNorm;
        segment.length = dim * 4u;
    }
    else
    {
        segment.role = ModelSegmentRole::FeedForward;
        segment.residency = ModelSegmentResidency::SharedExpert;
        segment.layer = index - (shape.layers + 2u);
        segment.expert = 1u;
        segment.length =
            2u * modelMatrixImageBytes(shape.ffnHidden, shape.dim) + modelMatrixImageBytes(shape.dim, shape.ffnHidden);
    }
    return segment;
}

/**
 * @brief The alignment a segment of @p residency starts on, in an image of 2^@p alignmentLog2.
 */
core::u64 segmentAlignment(ModelSegmentResidency residency, core::u32 alignmentLog2) noexcept
{
    return residency == ModelSegmentResidency::Resident ? kModelImageResidentAlignment : (1ull << alignmentLog2);
}

/**
 * @brief @p value rounded up to a multiple of the power of two @p alignment.
 */
core::u64 alignedUp(core::u64 value, core::u64 alignment) noexcept
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

/**
 * @brief Reads the header's fields after its magic and version.
 */
ModelImageHeader readHeader(const core::u8 *bytes) noexcept
{
    ModelImageHeader header{};

    header.alignmentLog2 = wordAt(bytes, kHeaderAlignment);
    header.segmentCount = wordAt(bytes, kHeaderSegmentCount);
    header.imageBytes = doubleWordAt(bytes, kHeaderImageBytes);
    header.shape.dim = wordAt(bytes, kHeaderDim);
    header.shape.layers = wordAt(bytes, kHeaderLayers);
    header.shape.heads = wordAt(bytes, kHeaderHeads);
    header.shape.ffnHidden = wordAt(bytes, kHeaderFeedForward);
    header.shape.contextLength = wordAt(bytes, kHeaderContext);
    header.shape.vocabSize = wordAt(bytes, kHeaderVocabSize);
    header.vocabTextBytes = wordAt(bytes, kHeaderVocabText);
    return header;
}

/**
 * @brief The verdict on the header of @p bytes, of @p size bytes, whose magic and version hold.
 */
ModelImageVerdict judgeHeader(const core::u8 *bytes, core::u64 size, const ModelImageHeader &header) noexcept
{
    if (wordAt(bytes, kHeaderBytes) != kModelImageHeaderBytes || wordAt(bytes, kHeaderFlags) != 0u ||
        wordAt(bytes, kHeaderReserved) != 0u)
        return ModelImageVerdict::BadHeader;
    if (!alignmentAccepted(header.alignmentLog2))
        return ModelImageVerdict::BadAlignment;
    if (!header.shape.valid() || header.vocabTextBytes == 0u)
        return ModelImageVerdict::BadShape;
    if (header.imageBytes != size)
        return ModelImageVerdict::BadImageBytes;
    return ModelImageVerdict::Accepted;
}

/**
 * @brief The verdict on @p segment, the zero-based @p index -th entry, its reserved word @p reserved, which may not
 *        start before @p cursor in an image of @p size bytes.
 */
ModelImageVerdict judgeEntry(const ModelImageHeader &header, core::u32 index, const ModelSegment &segment,
                             core::u32 reserved, core::u64 cursor, core::u64 size) noexcept
{
    const ModelSegment expected = expectedSegment(header.shape, header.vocabTextBytes, index);

    if (segment.role != expected.role || segment.residency != expected.residency || segment.layer != expected.layer ||
        segment.expert != expected.expert || reserved != 0u)
        return ModelImageVerdict::SegmentUnexpected;
    if (segment.length != expected.length)
        return ModelImageVerdict::SegmentWrongLength;
    if ((segment.offset & (segmentAlignment(segment.residency, header.alignmentLog2) - 1u)) != 0u)
        return ModelImageVerdict::SegmentMisaligned;
    if (segment.offset < cursor)
        return ModelImageVerdict::SegmentOverlaps;
    if (segment.offset > size || segment.length > size - segment.offset)
        return ModelImageVerdict::SegmentOutside;
    return ModelImageVerdict::Accepted;
}

/**
 * @brief An inspection that refuses with @p verdict, naming the one-based @p segment, @p header read so far.
 */
ModelImageInspection refused(ModelImageVerdict verdict, core::u32 segment, const ModelImageHeader &header) noexcept
{
    return ModelImageInspection{verdict, segment, header};
}

} // namespace

core::u64 modelMatrixImageBytes(core::u32 rows, core::u32 columns) noexcept
{
    return static_cast<core::u64>(rows) * quantBlockCount(columns) * kBlockImageBytes;
}

core::u32 modelImageSegmentCount(const ModelConfig &shape) noexcept
{
    const core::u64 count = 2ull * shape.layers + kSegmentsBesidesLayers;

    return count <= 0xFFFFFFFFull ? static_cast<core::u32>(count) : 0u;
}

ModelSegment placeModelSegment(const ModelConfig &shape, core::u32 vocabTextBytes, core::u32 alignmentLog2,
                               core::u32 index, core::u64 &cursor) noexcept
{
    ModelSegment segment = expectedSegment(shape, vocabTextBytes, index);

    segment.offset = alignedUp(cursor, segmentAlignment(segment.residency, alignmentLog2));
    cursor = segment.offset + segment.length;
    return segment;
}

core::u64 layOutModelImage(const ModelConfig &shape, core::u32 vocabTextBytes, core::u32 alignmentLog2,
                           ModelSegment *segments, core::u32 capacity) noexcept
{
    const core::u32 count = modelImageSegmentCount(shape);

    if (!shape.valid() || vocabTextBytes == 0u || !alignmentAccepted(alignmentLog2) || count == 0u ||
        (segments != nullptr && count > capacity))
        return 0u;

    core::u64 cursor = kModelImageHeaderBytes + static_cast<core::u64>(count) * kModelSegmentEntryBytes;
    for (core::u32 index = 0u; index < count; ++index)
    {
        const ModelSegment segment = placeModelSegment(shape, vocabTextBytes, alignmentLog2, index, cursor);
        if (segments != nullptr)
            segments[index] = segment;
    }
    return cursor;
}

core::u32 modelImageVersion(const core::u8 *bytes, core::u64 size) noexcept
{
    if (bytes == nullptr || size < 8u || wordAt(bytes, kHeaderMagic) != kModelMagic)
        return 0u;
    return wordAt(bytes, kHeaderVersion);
}

ModelImageInspection inspectModelImage(const core::u8 *bytes, core::u64 size, ModelSegment *segments,
                                       core::u32 capacity) noexcept
{
    if (bytes == nullptr || size < kModelImageHeaderBytes)
        return refused(ModelImageVerdict::TooShort, 0u, ModelImageHeader{});
    if (wordAt(bytes, kHeaderMagic) != kModelMagic)
        return refused(ModelImageVerdict::BadMagic, 0u, ModelImageHeader{});
    if (wordAt(bytes, kHeaderVersion) != kModelImageVersion2)
        return refused(ModelImageVerdict::NotVersion2, 0u, ModelImageHeader{});

    const ModelImageHeader header = readHeader(bytes);
    const ModelImageVerdict verdict = judgeHeader(bytes, size, header);
    if (verdict != ModelImageVerdict::Accepted)
        return refused(verdict, 0u, header);
    if (header.segmentCount != modelImageSegmentCount(header.shape) ||
        (segments != nullptr && header.segmentCount > capacity))
        return refused(ModelImageVerdict::BadSegmentCount, 0u, header);

    core::u64 cursor = kModelImageHeaderBytes + static_cast<core::u64>(header.segmentCount) * kModelSegmentEntryBytes;
    if (cursor > size)
        return refused(ModelImageVerdict::TooShort, 0u, header);
    for (core::u32 index = 0u; index < header.segmentCount; ++index)
    {
        const ModelSegment segment = readModelSegmentEntry(bytes, index);
        const core::u32 reserved = wordAt(
            bytes, kModelImageHeaderBytes + static_cast<core::u64>(index) * kModelSegmentEntryBytes + kEntryReserved);
        const ModelImageVerdict entry = judgeEntry(header, index, segment, reserved, cursor, size);
        if (entry != ModelImageVerdict::Accepted)
            return refused(entry, index + 1u, header);
        if (segments != nullptr)
            segments[index] = segment;
        cursor = segment.offset + segment.length;
    }
    return ModelImageInspection{ModelImageVerdict::Accepted, 0u, header};
}

ModelSegment readModelSegmentEntry(const core::u8 *bytes, core::u32 index) noexcept
{
    const core::u64 at = kModelImageHeaderBytes + static_cast<core::u64>(index) * kModelSegmentEntryBytes;
    ModelSegment segment{};

    segment.role = static_cast<ModelSegmentRole>(wordAt(bytes, at + kEntryRole));
    segment.residency = static_cast<ModelSegmentResidency>(wordAt(bytes, at + kEntryResidency));
    segment.layer = wordAt(bytes, at + kEntryLayer);
    segment.expert = wordAt(bytes, at + kEntryExpert);
    segment.offset = doubleWordAt(bytes, at + kEntryOffset);
    segment.length = doubleWordAt(bytes, at + kEntryLength);
    segment.signature = wordAt(bytes, at + kEntrySignature);
    return segment;
}

void writeModelImageHeader(core::u8 *image, const ModelImageHeader &header) noexcept
{
    putWordAt(image, kHeaderMagic, kModelMagic);
    putWordAt(image, kHeaderVersion, kModelImageVersion2);
    putWordAt(image, kHeaderBytes, kModelImageHeaderBytes);
    putWordAt(image, kHeaderAlignment, header.alignmentLog2);
    putWordAt(image, kHeaderFlags, 0u);
    putWordAt(image, kHeaderSegmentCount, header.segmentCount);
    putDoubleWordAt(image, kHeaderImageBytes, header.imageBytes);
    putWordAt(image, kHeaderDim, header.shape.dim);
    putWordAt(image, kHeaderLayers, header.shape.layers);
    putWordAt(image, kHeaderHeads, header.shape.heads);
    putWordAt(image, kHeaderFeedForward, header.shape.ffnHidden);
    putWordAt(image, kHeaderContext, header.shape.contextLength);
    putWordAt(image, kHeaderVocabSize, header.shape.vocabSize);
    putWordAt(image, kHeaderVocabText, header.vocabTextBytes);
    putWordAt(image, kHeaderReserved, 0u);
}

void writeModelSegmentEntry(core::u8 *image, core::u32 index, const ModelSegment &segment) noexcept
{
    const core::u64 at = kModelImageHeaderBytes + static_cast<core::u64>(index) * kModelSegmentEntryBytes;

    putWordAt(image, at + kEntryRole, static_cast<core::u32>(segment.role));
    putWordAt(image, at + kEntryResidency, static_cast<core::u32>(segment.residency));
    putWordAt(image, at + kEntryLayer, segment.layer);
    putWordAt(image, at + kEntryExpert, segment.expert);
    putDoubleWordAt(image, at + kEntryOffset, segment.offset);
    putDoubleWordAt(image, at + kEntryLength, segment.length);
    putWordAt(image, at + kEntrySignature, segment.signature);
    putWordAt(image, at + kEntryReserved, 0u);
}

core::u32 modelSegmentSignature(const core::u8 *bytes, const ModelSegment &segment) noexcept
{
    core::u32 hash = kFnv1aOffsetBasis;

    for (core::u64 at = segment.offset; at < segment.offset + segment.length; ++at)
        hash = (hash ^ bytes[at]) * kFnv1aPrime;
    return hash;
}

const char *modelImageVerdictName(ModelImageVerdict verdict) noexcept
{
    switch (verdict)
    {
    case ModelImageVerdict::Accepted: return "accepted";
    case ModelImageVerdict::TooShort: return "too_short";
    case ModelImageVerdict::BadMagic: return "bad_magic";
    case ModelImageVerdict::NotVersion2: return "not_version_2";
    case ModelImageVerdict::BadHeader: return "bad_header";
    case ModelImageVerdict::BadAlignment: return "bad_alignment";
    case ModelImageVerdict::BadShape: return "bad_shape";
    case ModelImageVerdict::BadImageBytes: return "bad_image_bytes";
    case ModelImageVerdict::BadSegmentCount: return "bad_segment_count";
    case ModelImageVerdict::SegmentUnexpected: return "segment_unexpected";
    case ModelImageVerdict::SegmentWrongLength: return "segment_wrong_length";
    case ModelImageVerdict::SegmentMisaligned: return "segment_misaligned";
    case ModelImageVerdict::SegmentOverlaps: return "segment_overlaps";
    case ModelImageVerdict::SegmentOutside: return "segment_outside";
    }
    return "unknown";
}

} // namespace lpl::infer

#endif /* LPL_HAS_FOUNDATION */
