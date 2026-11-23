/**
 * @file ModelImage.hpp
 * @brief The layout of a `.lplmind` image, laid out for paging.
 *
 * An image opens on a header of 64 bytes and the table of its segments, so that whoever maps it reads
 * where every segment lies without touching a page that is not mapped yet. The resident segments (the
 * vocabulary, the embedding, each layer's attention, the final norm and the rotary table) follow the table, 64-byte
 * aligned; each expert, a layer's feed-forward matrices one after the other, starts on a boundary of the image's
 * alignment, 4 KiB unless the writer asked for more, so that one page fault can bring exactly one expert in
 * (LplKernel#198). A dense model's feed-forward network of a layer is that layer's one shared expert. The rotary table
 * holds a word per pair of a head, how far that pair turns per position in units of 2^-32 turn (RotaryFrequencies).
 *
 * The table is checked whole on its own, which is cheap enough for a boot; the signature of a segment is checked
 * when the segment is read, so that a pager checks only what a fault brought in. Each refusal names what it refuses.
 *
 * All words are little-endian. The header: magic, version, header bytes, alignment as a power of two, flags,
 * segment count, the image's bytes on 64 bits, the shape (dim, layers, heads, feed-forward width, context,
 * vocabulary size), the vocabulary's text bytes, the norm's epsilon in units of 2^-32. A segment: role, residency,
 * layer and expert one-based (0 for none), offset and length on 64 bits, FNV-1a signature, a reserved word.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_MODELIMAGE_HPP
#    define LPL_LPL_INFER_MODELIMAGE_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/Model.hpp>

namespace lpl::infer {

/**
 * The version word of the one layout read and written: a header, the segment table, then the segments. LplAssistant
 * 0.4.0 and 0.5.0 wrote this layout as 2, a number left from the one it replaced. That retired layout said 1 too, and
 * fails as a bad header: its width stands where this header gives its own size, and its heads where the flags are 0.
 * An image of LplAssistant 0.6.0 to 0.6.2 holds no rotary table, 2L + 3 segments, and fails as a bad segment count.
 */
inline constexpr core::u32 kModelImageVersion = 1u;

/** Bytes of a header. */
inline constexpr core::u32 kModelImageHeaderBytes = 64u;

/** Bytes of one entry of the segment table. */
inline constexpr core::u32 kModelSegmentEntryBytes = 40u;

/** The alignment of an expert unless the writer asks for another, as a power of two: 4 KiB, a page. */
inline constexpr core::u32 kModelImageDefaultAlignmentLog2 = 12u;

/** The smallest alignment an image may ask for: a page of 4 KiB. */
inline constexpr core::u32 kModelImageLeastAlignmentLog2 = 12u;

/** The largest alignment an image may ask for: 4 MiB, the large page of i686. */
inline constexpr core::u32 kModelImageMostAlignmentLog2 = 22u;

/** The alignment of a resident segment: a cache line, which a quantised block's scale also needs. */
inline constexpr core::u32 kModelImageResidentAlignment = 64u;

/**
 * @enum ModelSegmentRole
 * @brief What a segment holds.
 */
enum class ModelSegmentRole : core::u32 {
    Vocabulary = 1u,  ///< The token table: each token's offset and length, then the text.
    Embedding = 2u,   ///< The tied embedding and output matrix.
    Attention = 3u,   ///< A layer's two norms, then its query, key, value and output matrices.
    FeedForward = 4u, ///< A layer's gate, up and down matrices: one expert.
    FinalNorm = 5u,   ///< The norm before the output head.
    Rotary = 6u,      ///< How far each pair of a head turns per position: RotaryFrequencies, one word per pair.
};

/**
 * @enum ModelSegmentResidency
 * @brief Whether a pager keeps a segment in memory.
 */
enum class ModelSegmentResidency : core::u32 {
    Resident = 1u,     ///< Read for every token, never paged.
    SharedExpert = 2u, ///< An expert every token reads: pinned.
    RoutedExpert = 3u, ///< An expert a router picks: evictable.
};

/**
 * @struct ModelSegment
 * @brief One entry of the segment table.
 */
struct ModelSegment {
    ModelSegmentRole role{ModelSegmentRole::Vocabulary};              ///< What it holds.
    ModelSegmentResidency residency{ModelSegmentResidency::Resident}; ///< Whether a pager keeps it.
    core::u32 layer{0u};                                              ///< One-based layer; 0 outside the layers.
    core::u32 expert{0u};                                             ///< One-based expert; 0 for no expert.
    core::u64 offset{0u};                                             ///< From the image's first byte.
    core::u64 length{0u};                                             ///< Bytes, padding excluded.
    core::u32 signature{0u};                                          ///< FNV-1a of its bytes.
};

/**
 * @struct ModelImageHeader
 * @brief What the header of an image says.
 */
struct ModelImageHeader {
    core::u32 alignmentLog2{0u};  ///< Each expert starts on a multiple of 2^alignmentLog2.
    core::u32 segmentCount{0u};   ///< Entries of the table.
    core::u64 imageBytes{0u};     ///< The whole image.
    ModelConfig shape{};          ///< The transformer's extents.
    core::u32 vocabTextBytes{0u}; ///< Bytes of the vocabulary's text.
};

/**
 * @enum ModelImageVerdict
 * @brief What checking an image's header and table found.
 */
enum class ModelImageVerdict : core::u32 {
    Accepted = 0u,            ///< The header and every entry of the table hold up.
    TooShort = 1u,            ///< Shorter than a header, or than the header and its table.
    BadMagic = 2u,            ///< The first word is not "LPLM".
    UnknownVersion = 3u,      ///< A version word other than @ref kModelImageVersion.
    BadHeader = 4u,           ///< The header's bytes or flags are not those written.
    BadAlignment = 5u,        ///< An alignment below 2^12 or above 2^22.
    BadShape = 6u,            ///< An extent at 0, heads that do not divide the width, or no vocabulary text.
    BadImageBytes = 7u,       ///< The bytes the header gives are not the image's.
    BadSegmentCount = 8u,     ///< Not the 2L + 4 segments of a dense model of L layers.
    SegmentUnexpected = 9u,   ///< A role, residency, layer or expert out of the table's order, or a reserved word set.
    SegmentWrongLength = 10u, ///< A length the shape does not give that role.
    SegmentMisaligned = 11u,  ///< A resident off 64 bytes, or an expert off the image's alignment.
    SegmentOverlaps = 12u,    ///< Starts before the table or the segment before it ends.
    SegmentOutside = 13u,     ///< Ends past the image.
};

/**
 * @struct ModelImageInspection
 * @brief The verdict, the segment it names, and the header read.
 */
struct ModelImageInspection {
    ModelImageVerdict verdict{ModelImageVerdict::TooShort}; ///< What the check found.
    core::u32 segment{0u};                                  ///< One-based entry refused; 0 when the header is.
    ModelImageHeader header{};                              ///< Valid as far as the verdict got.
};

/**
 * @brief Bytes a quantised matrix occupies in an image: each row's blocks, a block its scale word then its lanes.
 * @param rows    Output rows.
 * @param columns Input width.
 * @return The byte count.
 */
[[nodiscard]] core::u64 modelMatrixImageBytes(core::u32 rows, core::u32 columns) noexcept;

/**
 * @brief The segments of a dense model's image: the vocabulary, the embedding, each layer's attention, the
 *        final norm, the rotary table, then each layer's expert.
 * @param shape The transformer's extents.
 * @return 2L + 4; 0 for a shape whose count passes 32 bits.
 */
[[nodiscard]] core::u32 modelImageSegmentCount(const ModelConfig &shape) noexcept;

/**
 * @brief Places the zero-based @p index -th segment of a dense model's image at the first offset at or past
 *        @p cursor its residency allows: 64 bytes for a resident, 2^@p alignmentLog2 for an expert.
 *
 * @param shape          The transformer's extents.
 * @param vocabTextBytes Bytes of the vocabulary's text.
 * @param alignmentLog2  Each expert's alignment.
 * @param index          Below modelImageSegmentCount(@p shape).
 * @param cursor         Where the segment may start; moved past it.
 * @return The entry, its signature at 0.
 */
[[nodiscard]] ModelSegment placeModelSegment(const ModelConfig &shape, core::u32 vocabTextBytes,
                                             core::u32 alignmentLog2, core::u32 index, core::u64 &cursor) noexcept;

/**
 * @brief Places the segments of a dense model's image: the residents after the table, 64-byte aligned,
 *        then one expert per layer on boundaries of 2^@p alignmentLog2.
 *
 * @param shape          The transformer's extents.
 * @param vocabTextBytes Bytes of the vocabulary's text.
 * @param alignmentLog2  Each expert's alignment, kModelImageLeastAlignmentLog2 to kModelImageMostAlignmentLog2.
 * @param segments       Receives 2L + 4 entries, signatures at 0; nullptr for the image's bytes alone.
 * @param capacity       Room in @p segments.
 * @return The image's bytes; 0 when the shape or the alignment is refused, or the room is short.
 */
[[nodiscard]] core::u64 layOutModelImage(const ModelConfig &shape, core::u32 vocabTextBytes, core::u32 alignmentLog2,
                                         ModelSegment *segments, core::u32 capacity) noexcept;

/**
 * @brief The version word of an image.
 * @param bytes The image.
 * @param size  Its length.
 * @return The second word; 0 when the image is shorter than two words or its magic is not "LPLM".
 */
[[nodiscard]] core::u32 modelImageVersion(const core::u8 *bytes, core::u64 size) noexcept;

/**
 * @brief Checks an image's header and table against the layout its shape gives, reading nothing past the
 *        table: segment signatures are checked when the segments are read.
 *
 * @param bytes    The image.
 * @param size     Its length.
 * @param segments Receives the table's entries, as far as they were read; nullptr to check them alone.
 * @param capacity Room in @p segments; a table larger than it is refused as BadSegmentCount.
 * @return The verdict, the one-based entry it names, and the header.
 */
[[nodiscard]] ModelImageInspection inspectModelImage(const core::u8 *bytes, core::u64 size, ModelSegment *segments,
                                                     core::u32 capacity) noexcept;

/**
 * @brief Reads the zero-based @p index -th entry of an image's table.
 * @param bytes An image inspectModelImage accepted.
 * @param index Below its segment count.
 * @return The entry.
 */
[[nodiscard]] ModelSegment readModelSegmentEntry(const core::u8 *bytes, core::u32 index) noexcept;

/**
 * @brief Writes the header of an image, its reserved words and flags at 0.
 * @param image  Room for the header at least.
 * @param header What it says.
 */
void writeModelImageHeader(core::u8 *image, const ModelImageHeader &header) noexcept;

/**
 * @brief Writes the zero-based @p index -th entry of an image's table, its reserved word at 0.
 * @param image   Room for the header and the table at least.
 * @param index   The entry's place.
 * @param segment What it says.
 */
void writeModelSegmentEntry(core::u8 *image, core::u32 index, const ModelSegment &segment) noexcept;

/**
 * @brief The FNV-1a signature of a segment's bytes.
 * @param bytes   The image, whose table @p segment comes from.
 * @param segment An entry inspectModelImage accepted.
 * @return The signature, to compare with the entry's.
 */
[[nodiscard]] core::u32 modelSegmentSignature(const core::u8 *bytes, const ModelSegment &segment) noexcept;

/**
 * @brief The name a boot line gives a verdict.
 * @param verdict A verdict.
 * @return "accepted", "too_short", "bad_magic", "unknown_version", "bad_header", "bad_alignment", "bad_shape",
 *         "bad_image_bytes", "bad_segment_count", "segment_unexpected", "segment_wrong_length",
 *         "segment_misaligned", "segment_overlaps" or "segment_outside"; "unknown" otherwise.
 */
[[nodiscard]] const char *modelImageVerdictName(ModelImageVerdict verdict) noexcept;

} // namespace lpl::infer

#    endif /* LPL_HAS_FOUNDATION */

#endif /* LPL_LPL_INFER_MODELIMAGE_HPP */
