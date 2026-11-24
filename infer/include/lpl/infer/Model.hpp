/**
 * @file Model.hpp
 * @brief A read-only view over a weights blob.
 *
 * The blob arrives as a multiboot module, exactly like the cartridge. Same slot
 * machinery, same validation, same refusal to silently substitute a fallback.
 *
 * Three ways to obtain one, and they answer different questions. @ref Model::open
 * reads an image somebody baked, which is what a demon with a trained mind does.
 * @ref Model::synthesise derives every weight from a seed, which is what the parity
 * gate does — because a gate about ARITHMETIC must not depend on a file: shipping a
 * blob to both sides would prove the two targets can read the same bytes, not that
 * they compute the same forward pass from them. @ref Model::assemble borrows weights
 * its caller holds, which is what a converter does before @ref Model::writeBlob writes
 * them: the one writer of the layout stays one.
 *
 * The embedding table is quantised like every other matrix and is used twice: read
 * by row to embed a token, multiplied by to produce logits. Tied weights are the
 * usual arrangement for a small model, and here they also mean there is one table to
 * fold rather than two that could disagree.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_MODEL_HPP
#    define LPL_LPL_INFER_MODEL_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/Tensor.hpp>
#        include <lpl/infer/TensorArena.hpp>
#        include <lpl/infer/Tokenizer.hpp>
#        include <lpl/infer/Vocab.hpp>

#        include <bit>

namespace lpl::infer {

/// First four bytes of a `.lplmind` image, little-endian "LPLM".
inline constexpr core::u32 kModelMagic = 0x4D4C504Cu;

/** What checking an image's header and table found, defined in lpl/infer/ModelImage.hpp. */
enum class ModelImageVerdict : core::u32;

/**
 * @enum ModelReadVerdict
 * @brief What reading an image found.
 */
enum class ModelReadVerdict : core::u32 {
    Read = 0u,              ///< Every segment read: in place by Model::view, into the arena by Model::open.
    Misaligned = 1u,        ///< The image does not start on a multiple of kModelViewAlignment: Model::open copies it.
    BigEndianHost = 2u,     ///< This processor orders a word's bytes the other way round: Model::open copies it.
    ImageRefused = 3u,      ///< The header or the table does not hold up; the report's image verdict says how.
    SignatureMismatch = 4u, ///< A segment's bytes are not the ones its entry signs.
    VocabularyRefused = 5u, ///< A token is empty, longer than a token may be, or past the vocabulary's text.
    ArenaExhausted = 6u,    ///< The arena cannot hold what the reading claims.
    MergesRefused = 7u,     ///< A merge names a token past the vocabulary, is not the concatenation, or repeats a pair.
    ChatTokensMissing = 8u, ///< The vocabulary lacks a token its chat format needs: `<|im_start|>` or `<|im_end|>`.
    PartsMisshapen = 9u,    ///< A part Model::assemble is given is not of the shape's extents, or is missing.
};

/**
 * @struct ModelReadReport
 * @brief The verdict of a reading, and what it names.
 */
struct ModelReadReport {
    ModelReadVerdict verdict{ModelReadVerdict::ImageRefused}; ///< What the reading found.
    ModelImageVerdict image{}; ///< Why the table was refused; meaningful only when the verdict is ImageRefused.
    core::u32 segment{0u};     ///< One-based entry refused; 0 when none is.
};

/** Whether this processor orders a word's bytes as an image does, which reading an image in place needs. */
inline constexpr bool kHostIsLittleEndian = std::endian::native == std::endian::little;

/** The alignment an image's base needs to be read in place: a block's and a Q16.16 value's, 4 bytes. */
inline constexpr core::u32 kModelViewAlignment = alignof(QuantBlock);

/**
 * @brief Whether an image whose first byte is at @p bytes can be read where it lies.
 * @param bytes            The image's first byte; never read.
 * @param littleEndianHost Whether this processor orders a word's bytes as the image does.
 * @return Read; Misaligned when @p bytes is not a multiple of kModelViewAlignment; BigEndianHost on such a host.
 */
[[nodiscard]] ModelReadVerdict judgeModelView(const core::u8 *bytes, bool littleEndianHost) noexcept;

/**
 * @brief The name a boot line gives a reading's verdict.
 * @param verdict A verdict.
 * @return "read", "misaligned", "big_endian_host", "image_refused", "signature_mismatch", "vocabulary_refused",
 *         "arena_exhausted", "merges_refused", "chat_tokens_missing" or "parts_misshapen"; "unknown" otherwise.
 */
[[nodiscard]] const char *modelReadVerdictName(ModelReadVerdict verdict) noexcept;

/**
 * @struct ModelConfig
 * @brief The shape of a transformer.
 */
struct ModelConfig {
    core::u32 dim{0u};           ///< Width of the residual stream.
    core::u32 layers{0u};        ///< Blocks stacked.
    core::u32 heads{0u};         ///< Attention heads; must divide @c dim.
    core::u32 ffnHidden{0u};     ///< Width inside the position-wise network.
    core::u32 contextLength{0u}; ///< Positions the cache can hold.
    core::u32 vocabSize{0u};     ///< Tokens; must equal the bound vocabulary's size.
    core::u32 normEpsilon{0u};   ///< Added to the mean of the squares before its root, in units of 2^-32; 0 adds none.
    core::u32 keyValueHeads{0u}; ///< Key and value heads; must divide @c heads, each read by that many query heads.

    /**
     * @brief Width of one attention head.
     * @return dim / heads, or 0 when unset.
     */
    [[nodiscard]] constexpr core::u32 headDim() const noexcept { return heads == 0u ? 0u : dim / heads; }

    /**
     * @brief Channel pairs of one attention head, each turned by its own rotary frequency.
     * @return headDim() / 2, rounded down: an odd head's last channel is not turned.
     */
    [[nodiscard]] constexpr core::u32 rotaryPairs() const noexcept { return headDim() / 2u; }

    /**
     * @brief Width of one position's keys in one layer, and of its values.
     * @return keyValueHeads * headDim().
     */
    [[nodiscard]] constexpr core::u32 keyValueWidth() const noexcept { return keyValueHeads * headDim(); }

    /**
     * @brief Query heads that read one key and value head: query head h reads key and value head
     *        h / headsPerKeyValueHead(), so consecutive query heads share one, as arXiv:2305.13245 §2.2 groups them.
     * @return heads / keyValueHeads; 0 when keyValueHeads is.
     */
    [[nodiscard]] constexpr core::u32 headsPerKeyValueHead() const noexcept;

    /**
     * @brief Is this shape self-consistent?
     * @return true when every extent is non-zero, the heads divide the width and the key/value heads divide the heads.
     */
    [[nodiscard]] constexpr bool valid() const noexcept;
};

/**
 * @struct RotaryFrequencies
 * @brief How far each channel pair of a head turns per position, one table shared by every layer and head.
 *
 * Pair i turns by turnsPerPosition[i] / 2^32 of a turn per position, an unsigned Q0.32 fraction of a turn: the angle
 * at a position is their product modulo 2^32, which 32-bit arithmetic wraps exactly at every position. Whoever builds
 * the model computes the table from its own base and scaling, so ring 0 computes no power.
 */
struct RotaryFrequencies {
    const core::u32 *turnsPerPosition{nullptr}; ///< One entry per pair, in units of 2^-32 turn per position.
    core::u32 pairs{0u};                        ///< Entries: a head's channels halved, rounded down.
};

/**
 * @struct LayerWeights
 * @brief One transformer block's tensors.
 */
struct LayerWeights {
    QuantMatrixView query{};           ///< dim x dim
    QuantMatrixView key{};             ///< keyValueWidth x dim
    QuantMatrixView value{};           ///< keyValueWidth x dim
    QuantMatrixView output{};          ///< dim x dim
    QuantMatrixView gate{};            ///< ffnHidden x dim
    QuantMatrixView up{};              ///< ffnHidden x dim
    QuantMatrixView down{};            ///< dim x ffnHidden
    ConstVectorView attentionNorm{};   ///< dim
    ConstVectorView feedForwardNorm{}; ///< dim
};

/**
 * @struct ModelParts
 * @brief What a model is made of, held by whoever assembles it: Model::assemble borrows every part where it lies.
 *
 * A matrix's blocks are rows times quantBlockCount(columns) of them, which no check can see.
 */
struct ModelParts {
    ModelConfig shape{};                                ///< The transformer's extents.
    const char *vocabularyText{nullptr};                ///< The tokens' bytes, one after the other.
    core::u32 vocabularyTextBytes{0u};                  ///< Bytes of the vocabulary's text.
    const VocabEntry *vocabularyEntries{nullptr};       ///< shape.vocabSize entries, in identifier order.
    const BytePairMerge *merges{nullptr};               ///< The merges in the order of their ranks; nullptr for none.
    core::u32 mergeCount{0u};                           ///< Merges.
    PreTokenizer preTokenizer{PreTokenizer::WholeText}; ///< How text splits before it is segmented.
    ChatFormat chatFormat{ChatFormat::Plain};           ///< How messages are put in the model's turns.
    QuantMatrixView embedding{};                        ///< vocabSize rows of dim columns.
    const LayerWeights *layers{nullptr};                ///< shape.layers of them.
    ConstVectorView finalNorm{};                        ///< dim gains.
    RotaryFrequencies rotary{};                         ///< shape.rotaryPairs() entries.
};

/**
 * @class Model
 * @brief Weights, shape and vocabulary, all living in one arena.
 */
class Model {
public:
    Model() = default;

    /**
     * @brief Derives every weight from a seed.
     *
     * Each tensor draws from its OWN stream, keyed by a salt. Sharing one generator
     * would mean that inserting a tensor — or changing how many values an earlier
     * one consumes — shifted every later tensor, so an unrelated edit would silently
     * produce a different model. The same rule the world generator lives by.
     *
     * @param arena Storage for the weights.
     * @param shape The transformer's extents.
     * @param vocab An already-built table; its size must match @c shape.vocabSize.
     * @param seed  Master seed.
     * @return false when the shape is invalid or the arena is exhausted.
     */
    bool synthesise(TensorArena &arena, const ModelConfig &shape, const Vocab &vocab, core::u32 seed);

    /**
     * @brief Derives every weight from a seed, as the overload without merges does, and carries @p merges and
     *        @p preTokenizer into the image it writes.
     *
     * @param arena        Storage for the weights.
     * @param shape        The transformer's extents.
     * @param vocab        An already-built table; its size must match @c shape.vocabSize.
     * @param merges       Its ranked merges, indexed against @p vocab; borrowed, so they must outlive the model.
     * @param preTokenizer How text splits before it is segmented.
     * @param chatFormat   How messages are put in its turns; their tokens are found in @p vocab by their text.
     * @param seed         Master seed.
     * @return false when the shape is invalid, the arena is exhausted or @p vocab lacks a token @p chatFormat needs.
     */
    bool synthesise(TensorArena &arena, const ModelConfig &shape, const Vocab &vocab, const BytePairMerges &merges,
                    PreTokenizer preTokenizer, ChatFormat chatFormat, core::u32 seed);

    /**
     * @brief Makes a model over parts its caller holds, as @ref view makes one over an image's bytes, with the checks
     *        reading an image runs.
     *
     * Every part is borrowed: it must outlive the model and must not change. @p arena holds the vocabulary's sorted
     * index and the merges' index, nothing else. The refusals, in the order they are checked:
     * - the shape, the vocabulary's text bytes, the pre-tokenizer and the chat format as an image's header would hold
     *   them: ImageRefused, with BadShape, UnknownPreTokenizer or UnknownChatFormat;
     * - a missing part, or one whose rows, columns, count or pairs are not the shape's: PartsMisshapen;
     * - a vocabulary entry that is empty, longer than a token may be or past the text: VocabularyRefused;
     * - a vocabulary that lacks a token its chat format needs: ChatTokensMissing;
     * - merges that do not hold against the vocabulary: MergesRefused;
     * - an arena that cannot hold an index: ArenaExhausted.
     *
     * @param arena Storage for the indexes.
     * @param parts What the model is made of.
     * @return Read; otherwise what was refused, the one-based entry the part would take in the model's image, and the
     *         model is not ready.
     */
    [[nodiscard]] ModelReadReport assemble(TensorArena &arena, const ModelParts &parts);

    /**
     * @brief Reads a baked image into @p arena.
     *
     * Everything is copied, so the caller may free the bytes afterwards and they may lie at any address: a
     * @ref QuantBlock read through a misaligned pointer is undefined on the target this has to run on. @ref view
     * reads an image on a word boundary where it lies instead, and @ref copy says what a refused image failed.
     *
     * The image is one of version 1 (lpl/infer/ModelImage.hpp), read through its table, each segment's signature
     * checked before its bytes are taken; any other version is refused. The tokens of its chat format's turns are then
     * found in its vocabulary by their text, and a vocabulary that lacks one the format needs is refused.
     *
     * @param arena Storage.
     * @param bytes The image.
     * @param size  Its length.
     * @return false when the magic, version, shape, table, a signature or an extent does not hold up.
     */
    bool open(TensorArena &arena, const core::u8 *bytes, core::u32 size);

    /**
     * @brief Reads an image into @p arena, as @ref open does, and says what it refused.
     * @param arena Storage.
     * @param bytes The image, at any address.
     * @param size  Its length.
     * @return Read; otherwise what was refused, and the model is not ready.
     */
    [[nodiscard]] ModelReadReport copy(TensorArena &arena, const core::u8 *bytes, core::u32 size);

    /**
     * @brief Reads an image where it lies: every matrix, norm and the vocabulary's text and entries are
     *        viewed in @p bytes, and @p arena holds only the layers' views and the vocabulary's sorted index.
     *
     * The model borrows @p bytes: they must outlive it and must not change once read. Each segment's signature is
     * checked before its bytes are taken, as by @ref open.
     *
     * @param arena Storage for the views and the index.
     * @param bytes The image, its first byte on a multiple of kModelViewAlignment.
     * @param size  Its length.
     * @return Read; otherwise what was refused, and the model is not ready. Misaligned and BigEndianHost are left
     *         to @ref open, which copies.
     */
    [[nodiscard]] ModelReadReport view(TensorArena &arena, const core::u8 *bytes, core::u32 size);

    /**
     * @brief Bytes @ref writeBlob would produce for this model, its experts on 4 KiB boundaries.
     * @return The image size, or 0 when the model is unbuilt.
     */
    [[nodiscard]] core::u32 blobBytes() const noexcept;

    /**
     * @brief Bytes @ref writeBlob would produce for this model, its experts on boundaries of 2^@p alignmentLog2.
     * @param alignmentLog2 From 12, 4 KiB, to 22, 4 MiB (lpl/infer/ModelImage.hpp).
     * @return The image size, or 0 when the model is unbuilt, the alignment is refused or the image would pass 32
     *         bits.
     */
    [[nodiscard]] core::u32 blobBytes(core::u32 alignmentLog2) const noexcept;

    /**
     * @brief Serialises this model as an image of version 1 (lpl/infer/ModelImage.hpp), its experts on 4 KiB
     * boundaries.
     * @param out      Receives the image.
     * @param capacity Room in @p out.
     * @return Bytes written, or 0 when the model is unbuilt or the room is short.
     */
    [[nodiscard]] core::u32 writeBlob(core::u8 *out, core::u32 capacity) const noexcept;

    /**
     * @brief Serialises this model as an image of version 1, its experts on boundaries of 2^@p alignmentLog2, the
     *        padding zero.
     * @param out           Receives the image.
     * @param capacity      Room in @p out.
     * @param alignmentLog2 From 12, 4 KiB, to 22, 4 MiB.
     * @return Bytes written, or 0 when the model is unbuilt, the alignment is refused or the room is short.
     */
    [[nodiscard]] core::u32 writeBlob(core::u8 *out, core::u32 capacity, core::u32 alignmentLog2) const noexcept;

    /**
     * @brief Writes a token's embedding into @p out.
     * @param token Identifier.
     * @param out   Receives @c config().dim values.
     */
    void embed(core::u32 token, VectorView out) const noexcept;

    /**
     * @brief The shape.
     * @return The configuration.
     */
    [[nodiscard]] const ModelConfig &config() const noexcept { return _config; }

    /**
     * @brief The token table.
     * @return The vocabulary.
     */
    [[nodiscard]] const Vocab &vocab() const noexcept { return _vocab; }

    /**
     * @brief The ranked merges the vocabulary is encoded by.
     * @return The merges; a table that holds none for a vocabulary segmented by longest match.
     */
    [[nodiscard]] const BytePairMerges &merges() const noexcept { return _merges; }

    /**
     * @brief How text splits before it is segmented.
     * @return The pre-tokenizer; WholeText for a model a seed derives without merges.
     */
    [[nodiscard]] PreTokenizer preTokenizer() const noexcept { return _preTokenizer; }

    /**
     * @brief How messages are put in the model's turns, and the tokens of its turns.
     * @return The format and its tokens, found in the vocabulary; Plain, without tokens, for a model a seed derives
     *         without one.
     */
    [[nodiscard]] const ChatTokens &chatTokens() const noexcept { return _chatTokens; }

    /**
     * @brief One block's tensors.
     * @param index Block index.
     * @return Its weights.
     */
    [[nodiscard]] const LayerWeights &layer(core::u32 index) const noexcept
    {
        LPL_VERIFY(_layers != nullptr && index < _config.layers);
        return _layers[index];
    }

    /**
     * @brief The tied embedding / output matrix.
     * @return vocabSize rows of dim columns.
     */
    [[nodiscard]] const QuantMatrixView &embedding() const noexcept { return _embedding; }

    /**
     * @brief The normalisation applied before the output head.
     * @return dim gains.
     */
    [[nodiscard]] ConstVectorView finalNorm() const noexcept { return _finalNorm; }

    /**
     * @brief How far each pair of a head turns per position: the image's own table, or, for a model a seed derives,
     *        2^-i radians for pair i, rounded to a unit.
     * @return config().rotaryPairs() entries, in units of 2^-32 turn per position.
     */
    [[nodiscard]] RotaryFrequencies rotaryFrequencies() const noexcept { return _rotary; }

    /**
     * @brief Is there a model here?
     * @return true once @ref synthesise or @ref open has succeeded.
     */
    [[nodiscard]] bool ready() const noexcept { return _layers != nullptr; }

    /**
     * @brief Folds every weight into a running signature.
     *
     * Blocks are folded as scale plus lanes, in declaration order, then the final
     * norm and the rotary table. Folding the
     * QUANTISED form and not the values it came from is what makes the signature
     * mean something: two targets that rounded a weight differently would agree on
     * the source values and disagree here, which is precisely the failure the gate
     * is looking for.
     *
     * @param hash Running FNV-1a value.
     * @return The updated value.
     */
    [[nodiscard]] core::u32 fold(core::u32 hash) const noexcept;

private:
    /**
     * @enum Placement
     * @brief Where a reading puts the weights.
     */
    enum class Placement : core::u8 {
        CopyIntoArena, ///< Each value decoded into the arena, as Model::open does.
        InPlace,       ///< Each view over the image's bytes, as Model::view does.
    };

    /**
     * @brief Reads an image through its table, each segment's signature checked first.
     * @return Read, or what the table, a signature, the vocabulary or the arena refused.
     */
    [[nodiscard]] ModelReadReport readImage(TensorArena &arena, const core::u8 *bytes, core::u32 size,
                                            Placement placement);

    ModelConfig _config{};
    Vocab _vocab{};
    BytePairMerges _merges{};
    PreTokenizer _preTokenizer{PreTokenizer::WholeText};
    ChatTokens _chatTokens{};
    QuantMatrixView _embedding{};
    ConstVectorView _finalNorm{};
    RotaryFrequencies _rotary{};
    const LayerWeights *_layers{nullptr};
};

} // namespace lpl::infer

#        include "Model.inl"

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_MODEL_HPP
