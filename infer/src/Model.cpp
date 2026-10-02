/**
 * @file Model.cpp
 * @brief A read-only view over a weights blob.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Model.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/math/Random.hpp>

namespace lpl::infer {

namespace {

static_assert(sizeof(QuantBlock) == 4u + kQuantBlockLanes, "QuantBlock must have no padding to serialise field-wise");

/// Bytes one block occupies in an image: the scale, then the lanes.
constexpr core::u32 kBlockImageBytes = 4u + kQuantBlockLanes;

/// Header words before the vocabulary: magic, version, six extents, blob length.
constexpr core::u32 kHeaderWords = 9u;

/**
 * @brief Per-tensor stream salts.
 *
 * Distinct constants rather than an incrementing counter: a counter would renumber
 * every later tensor the day one is inserted, and the model would change without any
 * edit that named it.
 */
constexpr core::u32 kSaltEmbedding = 0x00E1u;
constexpr core::u32 kSaltQuery = 0x1010u;
constexpr core::u32 kSaltKey = 0x2020u;
constexpr core::u32 kSaltValue = 0x3030u;
constexpr core::u32 kSaltOutput = 0x4040u;
constexpr core::u32 kSaltGate = 0x5050u;
constexpr core::u32 kSaltUp = 0x6060u;
constexpr core::u32 kSaltDown = 0x7070u;
constexpr core::u32 kSaltAttentionNorm = 0x8080u;
constexpr core::u32 kSaltFeedForwardNorm = 0x9090u;
constexpr core::u32 kSaltFinalNorm = 0xA0A0u;

constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Folds one word into a running FNV-1a hash.
 * @param hash Running value.
 * @param word Word to absorb.
 */
void foldWord(core::u32 &hash, core::u32 word) noexcept { hash = (hash ^ word) * kFnv1aPrime; }

/**
 * @brief Writes a 32-bit word little-endian and advances.
 * @param cursor Write position, advanced by four.
 * @param value  The word.
 */
void putWord(core::u8 *&cursor, core::u32 value) noexcept
{
    cursor[0] = static_cast<core::u8>(value & 0xFFu);
    cursor[1] = static_cast<core::u8>((value >> 8) & 0xFFu);
    cursor[2] = static_cast<core::u8>((value >> 16) & 0xFFu);
    cursor[3] = static_cast<core::u8>((value >> 24) & 0xFFu);
    cursor += 4;
}

/**
 * @brief Reads a 32-bit word little-endian and advances.
 * @param cursor Read position, advanced by four.
 * @return The word.
 */
core::u32 takeWord(const core::u8 *&cursor) noexcept
{
    const core::u32 value = static_cast<core::u32>(cursor[0]) | (static_cast<core::u32>(cursor[1]) << 8) |
                            (static_cast<core::u32>(cursor[2]) << 16) | (static_cast<core::u32>(cursor[3]) << 24);
    cursor += 4;
    return value;
}

/**
 * @brief Fills a run with values drawn uniformly from [-@p amplitude, @p amplitude).
 * @param stream    Source of randomness.
 * @param values    Run to fill.
 * @param count     Entries.
 * @param amplitude Half-width of the range.
 */
void fillUniform(math::Random &stream, math::Fixed32 *values, core::u32 count, math::Fixed32 amplitude) noexcept
{
    for (core::u32 i = 0u; i < count; ++i)
        values[i] = (stream.unit() - math::Fixed32::half()) * amplitude * math::Fixed32::fromInt(2);
}

/**
 * @brief Claims and fills one quantised matrix.
 *
 * @param arena   Storage.
 * @param scratch Room for @p columns unquantised values.
 * @param stream  Source of randomness.
 * @param rows    Output rows.
 * @param columns Input width.
 * @param out     Receives the view.
 * @return false when the arena is exhausted.
 */
bool synthesiseMatrix(TensorArena &arena, math::Fixed32 *scratch, math::Random &stream, core::u32 rows,
                      core::u32 columns, QuantMatrixView &out)
{
    const core::u32 perRow = quantBlockCount(columns);
    QuantBlock *const blocks = arena.claim<QuantBlock>(static_cast<core::usize>(rows) * perRow);
    if (blocks == nullptr)
        return false;

    // A quarter is small enough that a stack of blocks does not saturate the Q16.16
    // residual stream, and large enough that eight-bit quantisation keeps more than
    // a handful of distinct levels.
    const math::Fixed32 amplitude = math::Fixed32::fromRaw(math::Fixed32::kOne / 4);
    for (core::u32 r = 0u; r < rows; ++r)
    {
        fillUniform(stream, scratch, columns, amplitude);
        quantiseRow(scratch, columns, blocks + static_cast<core::usize>(r) * perRow);
    }

    out = QuantMatrixView{blocks, rows, columns};
    return true;
}

/**
 * @brief Claims and fills one normalisation gain vector.
 * @param arena  Storage.
 * @param stream Source of randomness.
 * @param count  Entries.
 * @param out    Receives the view.
 * @return false when the arena is exhausted.
 */
bool synthesiseNorm(TensorArena &arena, math::Random &stream, core::u32 count, ConstVectorView &out)
{
    math::Fixed32 *const values = arena.claim<math::Fixed32>(count);
    if (values == nullptr)
        return false;

    // Centred on one: a gain is a correction to a normalised vector, and starting it
    // anywhere else would make the very first block scale the residual stream by a
    // constant that has nothing to do with the weights.
    const math::Fixed32 spread = math::Fixed32::fromRaw(math::Fixed32::kOne / 8);
    for (core::u32 i = 0u; i < count; ++i)
        values[i] = math::Fixed32::one() + (stream.unit() - math::Fixed32::half()) * spread;

    out = ConstVectorView{values, count};
    return true;
}

/**
 * @brief Folds a quantised matrix.
 * @param hash   Running value.
 * @param matrix The matrix.
 */
void foldMatrix(core::u32 &hash, const QuantMatrixView &matrix) noexcept
{
    const core::u32 total = matrix.rows * matrix.blocksPerRow();
    for (core::u32 b = 0u; b < total; ++b)
    {
        foldWord(hash, static_cast<core::u32>(matrix.blocks[b].scaleRaw));
        for (core::u32 i = 0u; i < kQuantBlockLanes; ++i)
            foldWord(hash, static_cast<core::u32>(static_cast<core::u8>(matrix.blocks[b].lanes[i])));
    }
}

/**
 * @brief Folds a run of activations.
 * @param hash   Running value.
 * @param vector The run.
 */
void foldVector(core::u32 &hash, ConstVectorView vector) noexcept
{
    for (core::u32 i = 0u; i < vector.count; ++i)
        foldWord(hash, static_cast<core::u32>(vector.values[i].raw()));
}

/**
 * @brief Writes a quantised matrix into an image.
 * @param cursor Write position.
 * @param matrix The matrix.
 */
void writeMatrix(core::u8 *&cursor, const QuantMatrixView &matrix) noexcept
{
    const core::u32 total = matrix.rows * matrix.blocksPerRow();
    for (core::u32 b = 0u; b < total; ++b)
    {
        putWord(cursor, static_cast<core::u32>(matrix.blocks[b].scaleRaw));
        for (core::u32 i = 0u; i < kQuantBlockLanes; ++i)
            *cursor++ = static_cast<core::u8>(matrix.blocks[b].lanes[i]);
    }
}

/**
 * @brief Reads a quantised matrix out of an image and into the arena.
 * @param arena   Storage.
 * @param cursor  Read position.
 * @param rows    Output rows.
 * @param columns Input width.
 * @param out     Receives the view.
 * @return false when the arena is exhausted.
 */
bool readMatrix(TensorArena &arena, const core::u8 *&cursor, core::u32 rows, core::u32 columns, QuantMatrixView &out)
{
    const core::u32 perRow = quantBlockCount(columns);
    const core::u32 total = rows * perRow;
    QuantBlock *const blocks = arena.claim<QuantBlock>(total);
    if (blocks == nullptr)
        return false;

    for (core::u32 b = 0u; b < total; ++b)
    {
        blocks[b].scaleRaw = static_cast<core::i32>(takeWord(cursor));
        for (core::u32 i = 0u; i < kQuantBlockLanes; ++i)
            blocks[b].lanes[i] = static_cast<core::i8>(*cursor++);
    }

    out = QuantMatrixView{blocks, rows, columns};
    return true;
}

/**
 * @brief Bytes a quantised matrix of this shape occupies in an image.
 * @param rows    Output rows.
 * @param columns Input width.
 * @return The byte count.
 */
core::u32 matrixImageBytes(core::u32 rows, core::u32 columns) noexcept
{
    return rows * quantBlockCount(columns) * kBlockImageBytes;
}

} // namespace

bool Model::synthesise(TensorArena &arena, const ModelConfig &shape, const Vocab &vocab, core::u32 seed)
{
    if (!shape.valid() || vocab.size() != shape.vocabSize)
        return false;

    const core::u32 widest = shape.dim > shape.ffnHidden ? shape.dim : shape.ffnHidden;
    math::Fixed32 *const scratch = arena.claim<math::Fixed32>(widest);
    LayerWeights *const layers = arena.claim<LayerWeights>(shape.layers);
    if (scratch == nullptr || layers == nullptr)
        return false;

    math::Random embeddingStream = math::deriveStream(seed, kSaltEmbedding);
    if (!synthesiseMatrix(arena, scratch, embeddingStream, shape.vocabSize, shape.dim, _embedding))
        return false;

    for (core::u32 l = 0u; l < shape.layers; ++l)
    {
        // The layer index enters the salt, so two blocks of the same model never
        // share a stream and inserting a block does not renumber the others.
        const core::u32 tag = l * 64u;
        math::Random queryStream = math::deriveStream(seed, kSaltQuery + tag);
        math::Random keyStream = math::deriveStream(seed, kSaltKey + tag);
        math::Random valueStream = math::deriveStream(seed, kSaltValue + tag);
        math::Random outputStream = math::deriveStream(seed, kSaltOutput + tag);
        math::Random gateStream = math::deriveStream(seed, kSaltGate + tag);
        math::Random upStream = math::deriveStream(seed, kSaltUp + tag);
        math::Random downStream = math::deriveStream(seed, kSaltDown + tag);
        math::Random attentionNormStream = math::deriveStream(seed, kSaltAttentionNorm + tag);
        math::Random feedForwardNormStream = math::deriveStream(seed, kSaltFeedForwardNorm + tag);

        if (!synthesiseMatrix(arena, scratch, queryStream, shape.dim, shape.dim, layers[l].query) ||
            !synthesiseMatrix(arena, scratch, keyStream, shape.dim, shape.dim, layers[l].key) ||
            !synthesiseMatrix(arena, scratch, valueStream, shape.dim, shape.dim, layers[l].value) ||
            !synthesiseMatrix(arena, scratch, outputStream, shape.dim, shape.dim, layers[l].output) ||
            !synthesiseMatrix(arena, scratch, gateStream, shape.ffnHidden, shape.dim, layers[l].gate) ||
            !synthesiseMatrix(arena, scratch, upStream, shape.ffnHidden, shape.dim, layers[l].up) ||
            !synthesiseMatrix(arena, scratch, downStream, shape.dim, shape.ffnHidden, layers[l].down) ||
            !synthesiseNorm(arena, attentionNormStream, shape.dim, layers[l].attentionNorm) ||
            !synthesiseNorm(arena, feedForwardNormStream, shape.dim, layers[l].feedForwardNorm))
            return false;
    }

    math::Random finalStream = math::deriveStream(seed, kSaltFinalNorm);
    if (!synthesiseNorm(arena, finalStream, shape.dim, _finalNorm))
        return false;

    _config = shape;
    _vocab = vocab;
    _layers = layers;
    return true;
}

bool Model::open(TensorArena &arena, const core::u8 *bytes, core::u32 size)
{
    if (bytes == nullptr || size < kHeaderWords * 4u)
        return false;

    const core::u8 *cursor = bytes;
    if (takeWord(cursor) != kModelMagic)
        return false;
    if (takeWord(cursor) != kModelVersion)
        return false;

    ModelConfig shape{};
    shape.dim = takeWord(cursor);
    shape.layers = takeWord(cursor);
    shape.heads = takeWord(cursor);
    shape.ffnHidden = takeWord(cursor);
    shape.contextLength = takeWord(cursor);
    shape.vocabSize = takeWord(cursor);
    const core::u32 vocabBlobBytes = takeWord(cursor);
    if (!shape.valid() || vocabBlobBytes == 0u)
        return false;

    // The declared extent is checked before a single byte past the header is read.
    // An image that says it holds more than it does is the ordinary shape of
    // corruption, and reading it would fault rather than report.
    const core::u32 vocabPadded = (vocabBlobBytes + 3u) & ~3u;
    core::u64 expected = static_cast<core::u64>(kHeaderWords) * 4u;
    expected += static_cast<core::u64>(shape.vocabSize) * 8u;
    expected += vocabPadded;
    expected += matrixImageBytes(shape.vocabSize, shape.dim);
    for (core::u32 l = 0u; l < shape.layers; ++l)
    {
        expected += static_cast<core::u64>(shape.dim) * 4u * 2u;
        expected += 4u * static_cast<core::u64>(matrixImageBytes(shape.dim, shape.dim));
        expected += 2u * static_cast<core::u64>(matrixImageBytes(shape.ffnHidden, shape.dim));
        expected += matrixImageBytes(shape.dim, shape.ffnHidden);
    }
    expected += static_cast<core::u64>(shape.dim) * 4u;
    if (expected != static_cast<core::u64>(size))
        return false;

    VocabEntry *const entries = arena.claim<VocabEntry>(shape.vocabSize);
    char *const blob = arena.claim<char>(vocabBlobBytes);
    LayerWeights *const layers = arena.claim<LayerWeights>(shape.layers);
    if (entries == nullptr || blob == nullptr || layers == nullptr)
        return false;

    for (core::u32 i = 0u; i < shape.vocabSize; ++i)
    {
        entries[i].offset = takeWord(cursor);
        entries[i].length = takeWord(cursor);
        if (entries[i].length == 0u || entries[i].length > kMaxTokenBytes ||
            entries[i].offset + entries[i].length > vocabBlobBytes)
            return false;
    }
    for (core::u32 i = 0u; i < vocabBlobBytes; ++i)
        blob[i] = static_cast<char>(cursor[i]);
    cursor += vocabPadded;

    Vocab table;
    if (!table.build(arena, blob, vocabBlobBytes, entries, shape.vocabSize))
        return false;

    if (!readMatrix(arena, cursor, shape.vocabSize, shape.dim, _embedding))
        return false;

    for (core::u32 l = 0u; l < shape.layers; ++l)
    {
        math::Fixed32 *const attentionNorm = arena.claim<math::Fixed32>(shape.dim);
        math::Fixed32 *const feedForwardNorm = arena.claim<math::Fixed32>(shape.dim);
        if (attentionNorm == nullptr || feedForwardNorm == nullptr)
            return false;
        for (core::u32 i = 0u; i < shape.dim; ++i)
            attentionNorm[i] = math::Fixed32::fromRaw(static_cast<core::i32>(takeWord(cursor)));
        for (core::u32 i = 0u; i < shape.dim; ++i)
            feedForwardNorm[i] = math::Fixed32::fromRaw(static_cast<core::i32>(takeWord(cursor)));
        layers[l].attentionNorm = ConstVectorView{attentionNorm, shape.dim};
        layers[l].feedForwardNorm = ConstVectorView{feedForwardNorm, shape.dim};

        if (!readMatrix(arena, cursor, shape.dim, shape.dim, layers[l].query) ||
            !readMatrix(arena, cursor, shape.dim, shape.dim, layers[l].key) ||
            !readMatrix(arena, cursor, shape.dim, shape.dim, layers[l].value) ||
            !readMatrix(arena, cursor, shape.dim, shape.dim, layers[l].output) ||
            !readMatrix(arena, cursor, shape.ffnHidden, shape.dim, layers[l].gate) ||
            !readMatrix(arena, cursor, shape.ffnHidden, shape.dim, layers[l].up) ||
            !readMatrix(arena, cursor, shape.dim, shape.ffnHidden, layers[l].down))
            return false;
    }

    math::Fixed32 *const finalNorm = arena.claim<math::Fixed32>(shape.dim);
    if (finalNorm == nullptr)
        return false;
    for (core::u32 i = 0u; i < shape.dim; ++i)
        finalNorm[i] = math::Fixed32::fromRaw(static_cast<core::i32>(takeWord(cursor)));

    _config = shape;
    _vocab = table;
    _finalNorm = ConstVectorView{finalNorm, shape.dim};
    _layers = layers;
    return true;
}

core::u32 Model::blobBytes() const noexcept
{
    if (!ready())
        return 0u;

    const core::u32 vocabPadded = (_vocab.blobBytes() + 3u) & ~3u;
    core::u32 total = kHeaderWords * 4u;
    total += _config.vocabSize * 8u;
    total += vocabPadded;
    total += matrixImageBytes(_config.vocabSize, _config.dim);
    for (core::u32 l = 0u; l < _config.layers; ++l)
    {
        total += _config.dim * 4u * 2u;
        total += 4u * matrixImageBytes(_config.dim, _config.dim);
        total += 2u * matrixImageBytes(_config.ffnHidden, _config.dim);
        total += matrixImageBytes(_config.dim, _config.ffnHidden);
    }
    total += _config.dim * 4u;
    return total;
}

core::u32 Model::writeBlob(core::u8 *out, core::u32 capacity) const noexcept
{
    const core::u32 needed = blobBytes();
    if (out == nullptr || needed == 0u || capacity < needed)
        return 0u;

    core::u8 *cursor = out;
    putWord(cursor, kModelMagic);
    putWord(cursor, kModelVersion);
    putWord(cursor, _config.dim);
    putWord(cursor, _config.layers);
    putWord(cursor, _config.heads);
    putWord(cursor, _config.ffnHidden);
    putWord(cursor, _config.contextLength);
    putWord(cursor, _config.vocabSize);
    putWord(cursor, _vocab.blobBytes());

    for (core::u32 i = 0u; i < _config.vocabSize; ++i)
    {
        putWord(cursor, _vocab.entries()[i].offset);
        putWord(cursor, _vocab.entries()[i].length);
    }
    for (core::u32 i = 0u; i < _vocab.blobBytes(); ++i)
        *cursor++ = static_cast<core::u8>(_vocab.blob()[i]);
    for (core::u32 i = _vocab.blobBytes(); i < ((_vocab.blobBytes() + 3u) & ~3u); ++i)
        *cursor++ = 0u;

    writeMatrix(cursor, _embedding);
    for (core::u32 l = 0u; l < _config.layers; ++l)
    {
        for (core::u32 i = 0u; i < _config.dim; ++i)
            putWord(cursor, static_cast<core::u32>(_layers[l].attentionNorm.values[i].raw()));
        for (core::u32 i = 0u; i < _config.dim; ++i)
            putWord(cursor, static_cast<core::u32>(_layers[l].feedForwardNorm.values[i].raw()));
        writeMatrix(cursor, _layers[l].query);
        writeMatrix(cursor, _layers[l].key);
        writeMatrix(cursor, _layers[l].value);
        writeMatrix(cursor, _layers[l].output);
        writeMatrix(cursor, _layers[l].gate);
        writeMatrix(cursor, _layers[l].up);
        writeMatrix(cursor, _layers[l].down);
    }
    for (core::u32 i = 0u; i < _config.dim; ++i)
        putWord(cursor, static_cast<core::u32>(_finalNorm.values[i].raw()));

    return needed;
}

void Model::embed(core::u32 token, VectorView out) const noexcept
{
    LPL_VERIFY(ready() && out.count == _config.dim);
    if (token >= _config.vocabSize)
    {
        for (core::u32 i = 0u; i < out.count; ++i)
            out.at(i) = math::Fixed32::zero();
        return;
    }

    const QuantBlock *const row = _embedding.row(token);
    for (core::u32 i = 0u; i < _config.dim; ++i)
        out.at(i) = dequantiseLane(row[i / kQuantBlockLanes], i % kQuantBlockLanes);
}

core::u32 Model::fold(core::u32 hash) const noexcept
{
    if (!ready())
        return hash;

    foldWord(hash, _config.dim);
    foldWord(hash, _config.layers);
    foldWord(hash, _config.heads);
    foldWord(hash, _config.ffnHidden);
    foldWord(hash, _config.contextLength);
    foldWord(hash, _config.vocabSize);

    foldMatrix(hash, _embedding);
    for (core::u32 l = 0u; l < _config.layers; ++l)
    {
        foldVector(hash, _layers[l].attentionNorm);
        foldVector(hash, _layers[l].feedForwardNorm);
        foldMatrix(hash, _layers[l].query);
        foldMatrix(hash, _layers[l].key);
        foldMatrix(hash, _layers[l].value);
        foldMatrix(hash, _layers[l].output);
        foldMatrix(hash, _layers[l].gate);
        foldMatrix(hash, _layers[l].up);
        foldMatrix(hash, _layers[l].down);
    }
    foldVector(hash, _finalNorm);
    return hash;
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
