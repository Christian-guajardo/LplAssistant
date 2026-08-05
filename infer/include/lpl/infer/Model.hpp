/**
 * @file Model.hpp
 * @brief A read-only view over a weights blob.
 *
 * The blob arrives as a multiboot module, exactly like the cartridge. Same slot
 * machinery, same validation, same refusal to silently substitute a fallback.
 *
 * Two ways to obtain one, and they answer different questions. @ref Model::open
 * reads an image somebody baked, which is what a demon with a trained mind does.
 * @ref Model::synthesise derives every weight from a seed, which is what the parity
 * gate does — because a gate about ARITHMETIC must not depend on a file: shipping a
 * blob to both sides would prove the two targets can read the same bytes, not that
 * they compute the same forward pass from them.
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
#        include <lpl/infer/Vocab.hpp>

namespace lpl::infer {

/// First four bytes of a `.lplmind` image, little-endian "LPLM".
inline constexpr core::u32 kModelMagic = 0x4D4C504Cu;

/// Image layout revision. Bumped when the byte order of the sections changes.
inline constexpr core::u32 kModelVersion = 1u;

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

    /**
     * @brief Width of one attention head.
     * @return dim / heads, or 0 when unset.
     */
    [[nodiscard]] constexpr core::u32 headDim() const noexcept { return heads == 0u ? 0u : dim / heads; }

    /**
     * @brief Is this shape self-consistent?
     * @return true when every extent is non-zero and the heads divide the width.
     */
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return dim != 0u && layers != 0u && heads != 0u && ffnHidden != 0u && contextLength != 0u &&
               vocabSize != 0u && (dim % heads) == 0u;
    }
};

/**
 * @struct LayerWeights
 * @brief One transformer block's tensors.
 */
struct LayerWeights {
    QuantMatrixView query{};   ///< dim x dim
    QuantMatrixView key{};     ///< dim x dim
    QuantMatrixView value{};   ///< dim x dim
    QuantMatrixView output{};  ///< dim x dim
    QuantMatrixView gate{};    ///< ffnHidden x dim
    QuantMatrixView up{};      ///< ffnHidden x dim
    QuantMatrixView down{};    ///< dim x ffnHidden
    ConstVectorView attentionNorm{};   ///< dim
    ConstVectorView feedForwardNorm{}; ///< dim
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
     * @brief Reads a baked image.
     *
     * Everything is copied into @p arena rather than pointed at in place. A boot
     * module lands wherever the loader put it, with no guarantee about alignment,
     * and a @ref QuantBlock read through a misaligned pointer is undefined on the
     * target this has to run on.
     *
     * @param arena Storage.
     * @param bytes The image.
     * @param size  Its length.
     * @return false when the magic, version, shape or extent does not hold up.
     */
    bool open(TensorArena &arena, const core::u8 *bytes, core::u32 size);

    /**
     * @brief Bytes @ref writeBlob would produce for this model.
     * @return The image size, or 0 when the model is unbuilt.
     */
    [[nodiscard]] core::u32 blobBytes() const noexcept;

    /**
     * @brief Serialises this model.
     * @param out      Receives the image.
     * @param capacity Room in @p out.
     * @return Bytes written, or 0 when the model is unbuilt or the room is short.
     */
    [[nodiscard]] core::u32 writeBlob(core::u8 *out, core::u32 capacity) const noexcept;

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
     * @brief Is there a model here?
     * @return true once @ref synthesise or @ref open has succeeded.
     */
    [[nodiscard]] bool ready() const noexcept { return _layers != nullptr; }

    /**
     * @brief Folds every weight into a running signature.
     *
     * Blocks are folded as scale plus lanes, in declaration order. Folding the
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
    ModelConfig _config{};
    Vocab _vocab{};
    QuantMatrixView _embedding{};
    ConstVectorView _finalNorm{};
    LayerWeights *_layers{nullptr};
};

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_MODEL_HPP
