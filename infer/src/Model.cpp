/**
 * @file Model.cpp
 * @brief A read-only view over a weights blob.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Model.hpp>
#include <lpl/infer/ModelImage.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/math/Random.hpp>

namespace lpl::infer {

namespace {

static_assert(sizeof(QuantBlock) == 4u + kQuantBlockLanes, "QuantBlock must have no padding to serialise field-wise");

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
 * @brief Writes a vector of Q16.16 values into an image.
 * @param cursor Write position.
 * @param vector The values.
 */
void writeVector(core::u8 *&cursor, ConstVectorView vector) noexcept
{
    for (core::u32 i = 0u; i < vector.count; ++i)
        putWord(cursor, static_cast<core::u32>(vector.values[i].raw()));
}

/**
 * @brief Reads @p count Q16.16 values out of an image and into the arena.
 * @return false when the arena is exhausted.
 */
bool readVector(TensorArena &arena, const core::u8 *&cursor, core::u32 count, ConstVectorView &out)
{
    math::Fixed32 *const values = arena.claim<math::Fixed32>(count);
    if (values == nullptr)
        return false;
    for (core::u32 i = 0u; i < count; ++i)
        values[i] = math::Fixed32::fromRaw(static_cast<core::i32>(takeWord(cursor)));
    out = ConstVectorView{values, count};
    return true;
}

/**
 * @brief Reads a vocabulary of @p count entries and @p textBytes of text out of an image, every entry inside the text.
 * @return false when an entry is empty, longer than a token, past the text, or the arena is exhausted.
 */
bool readVocabulary(TensorArena &arena, const core::u8 *&cursor, core::u32 count, core::u32 textBytes, Vocab &out)
{
    VocabEntry *const entries = arena.claim<VocabEntry>(count);
    char *const text = arena.claim<char>(textBytes);
    if (entries == nullptr || text == nullptr)
        return false;

    for (core::u32 i = 0u; i < count; ++i)
    {
        entries[i].offset = takeWord(cursor);
        entries[i].length = takeWord(cursor);
        if (entries[i].length == 0u || entries[i].length > kMaxTokenBytes || entries[i].offset > textBytes ||
            entries[i].length > textBytes - entries[i].offset)
            return false;
    }
    for (core::u32 i = 0u; i < textBytes; ++i)
        text[i] = static_cast<char>(cursor[i]);
    cursor += textBytes;
    return out.build(arena, text, textBytes, entries, count);
}

/**
 * @brief Reads a layer's attention segment: its two norms, then its query, key, value and output matrices.
 * @return false when the arena is exhausted.
 */
bool readAttention(TensorArena &arena, const core::u8 *&cursor, const ModelConfig &shape, LayerWeights &out)
{
    return readVector(arena, cursor, shape.dim, out.attentionNorm) &&
           readVector(arena, cursor, shape.dim, out.feedForwardNorm) &&
           readMatrix(arena, cursor, shape.dim, shape.dim, out.query) &&
           readMatrix(arena, cursor, shape.dim, shape.dim, out.key) &&
           readMatrix(arena, cursor, shape.dim, shape.dim, out.value) &&
           readMatrix(arena, cursor, shape.dim, shape.dim, out.output);
}

/**
 * @brief Reads a layer's expert segment: its gate, up and down matrices.
 * @return false when the arena is exhausted.
 */
bool readExpert(TensorArena &arena, const core::u8 *&cursor, const ModelConfig &shape, LayerWeights &out)
{
    return readMatrix(arena, cursor, shape.ffnHidden, shape.dim, out.gate) &&
           readMatrix(arena, cursor, shape.ffnHidden, shape.dim, out.up) &&
           readMatrix(arena, cursor, shape.dim, shape.ffnHidden, out.down);
}

/**
 * @brief Writes the bytes of @p segment of @p model at @p at: what its role holds, in the order the readers take it.
 */
void writeSegment(core::u8 *at, const Model &model, const ModelSegment &segment) noexcept
{
    core::u8 *cursor = at;

    switch (segment.role)
    {
    case ModelSegmentRole::Vocabulary:
        for (core::u32 i = 0u; i < model.vocab().size(); ++i)
        {
            putWord(cursor, model.vocab().entries()[i].offset);
            putWord(cursor, model.vocab().entries()[i].length);
        }
        for (core::u32 i = 0u; i < model.vocab().blobBytes(); ++i)
            *cursor++ = static_cast<core::u8>(model.vocab().blob()[i]);
        return;
    case ModelSegmentRole::Embedding: writeMatrix(cursor, model.embedding()); return;
    case ModelSegmentRole::Attention:
        writeVector(cursor, model.layer(segment.layer - 1u).attentionNorm);
        writeVector(cursor, model.layer(segment.layer - 1u).feedForwardNorm);
        writeMatrix(cursor, model.layer(segment.layer - 1u).query);
        writeMatrix(cursor, model.layer(segment.layer - 1u).key);
        writeMatrix(cursor, model.layer(segment.layer - 1u).value);
        writeMatrix(cursor, model.layer(segment.layer - 1u).output);
        return;
    case ModelSegmentRole::FinalNorm: writeVector(cursor, model.finalNorm()); return;
    case ModelSegmentRole::FeedForward:
        writeMatrix(cursor, model.layer(segment.layer - 1u).gate);
        writeMatrix(cursor, model.layer(segment.layer - 1u).up);
        writeMatrix(cursor, model.layer(segment.layer - 1u).down);
        return;
    }
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

bool Model::openImage(TensorArena &arena, const core::u8 *bytes, core::u32 size)
{
    const ModelImageInspection inspection = inspectModelImage(bytes, size, nullptr, 0u);
    if (inspection.verdict != ModelImageVerdict::Accepted)
        return false;

    const ModelConfig &shape = inspection.header.shape;
    LayerWeights *const layers = arena.claim<LayerWeights>(shape.layers);
    Vocab table;
    if (layers == nullptr)
        return false;

    for (core::u32 index = 0u; index < inspection.header.segmentCount; ++index)
    {
        const ModelSegment segment = readModelSegmentEntry(bytes, index);
        const core::u8 *cursor = bytes + segment.offset;
        if (modelSegmentSignature(bytes, segment) != segment.signature)
            return false;

        bool read = false;
        switch (segment.role)
        {
        case ModelSegmentRole::Vocabulary:
            read = readVocabulary(arena, cursor, shape.vocabSize, inspection.header.vocabTextBytes, table);
            break;
        case ModelSegmentRole::Embedding:
            read = readMatrix(arena, cursor, shape.vocabSize, shape.dim, _embedding);
            break;
        case ModelSegmentRole::Attention: read = readAttention(arena, cursor, shape, layers[segment.layer - 1u]); break;
        case ModelSegmentRole::FinalNorm: read = readVector(arena, cursor, shape.dim, _finalNorm); break;
        case ModelSegmentRole::FeedForward: read = readExpert(arena, cursor, shape, layers[segment.layer - 1u]); break;
        }
        if (!read)
            return false;
    }

    _config = shape;
    _vocab = table;
    _layers = layers;
    return true;
}

bool Model::open(TensorArena &arena, const core::u8 *bytes, core::u32 size) { return openImage(arena, bytes, size); }

core::u32 Model::blobBytes() const noexcept { return blobBytes(kModelImageDefaultAlignmentLog2); }

core::u32 Model::blobBytes(core::u32 alignmentLog2) const noexcept
{
    if (!ready())
        return 0u;

    const core::u64 bytes = layOutModelImage(_config, _vocab.blobBytes(), alignmentLog2, nullptr, 0u);
    return bytes <= 0xFFFFFFFFull ? static_cast<core::u32>(bytes) : 0u;
}

core::u32 Model::writeBlob(core::u8 *out, core::u32 capacity) const noexcept
{
    return writeBlob(out, capacity, kModelImageDefaultAlignmentLog2);
}

core::u32 Model::writeBlob(core::u8 *out, core::u32 capacity, core::u32 alignmentLog2) const noexcept
{
    const core::u32 needed = blobBytes(alignmentLog2);
    if (out == nullptr || needed == 0u || capacity < needed)
        return 0u;

    const core::u32 count = modelImageSegmentCount(_config);
    core::u64 cursor = kModelImageHeaderBytes + static_cast<core::u64>(count) * kModelSegmentEntryBytes;
    for (core::u32 i = 0u; i < needed; ++i)
        out[i] = 0u;
    for (core::u32 index = 0u; index < count; ++index)
    {
        ModelSegment segment = placeModelSegment(_config, _vocab.blobBytes(), alignmentLog2, index, cursor);
        writeSegment(out + segment.offset, *this, segment);
        segment.signature = modelSegmentSignature(out, segment);
        writeModelSegmentEntry(out, index, segment);
    }

    ModelImageHeader header{};
    header.alignmentLog2 = alignmentLog2;
    header.segmentCount = count;
    header.imageBytes = needed;
    header.shape = _config;
    header.vocabTextBytes = _vocab.blobBytes();
    writeModelImageHeader(out, header);
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
