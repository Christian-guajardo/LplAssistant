#include "model_image_writer.hpp"

#include <lpl/infer/Attention.hpp>
#include <lpl/infer/KvCache.hpp>
#include <lpl/infer/Model.hpp>
#include <lpl/infer/ModelImage.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/infer/Quant.hpp>
#include <lpl/infer/TensorArena.hpp>
#include <lpl/infer/Transformer.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(grouped_attention);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;
constexpr lpl::core::u32 kGroupedHeads = 6u;         ///< Query heads of the grouped model: three per key/value head.
constexpr lpl::core::u32 kGroupedKeyValueHeads = 2u; ///< Its key and value heads.
constexpr lpl::core::u32 kThinkingSteps = 12u;       ///< Tokens each model is fed, one position each.
constexpr lpl::core::u32 kTokenStride = 37u;         ///< Step between the tokens fed, a number prime to 121.
constexpr lpl::core::u32 kTableRoom = 16u;
constexpr lpl::core::u32 kWordBytes = 4u;
constexpr lpl::core::usize kCacheArenaBytes = 192u * 1024u;
constexpr lpl::core::u32 kParityWidth = 48u;     ///< The parity model's residual stream.
constexpr lpl::core::u32 kGroupedKeyWidth = 16u; ///< Its keys with 2 key/value heads of 8 channels.
constexpr lpl::core::u32 kTurnedPosition = 5u;   ///< Where the token whose key is checked stands.
constexpr lpl::core::u32 kTurnedToken = 7u;      ///< That token.

/**
 * @name SmolLM2-135M-Instruct's shape, from its config.json
 *       (https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/config.json)
 * @{
 */
constexpr lpl::core::u32 kSmolWidth = 576u;
constexpr lpl::core::u32 kSmolLayers = 30u;
constexpr lpl::core::u32 kSmolHeads = 9u;
constexpr lpl::core::u32 kSmolKeyValueHeads = 3u;
constexpr lpl::core::u32 kSmolFeedForward = 1536u;
constexpr lpl::core::u32 kSmolVocabulary = 49152u;
/** @} */

/**
 * @name Bytes the cache holds per position for SmolLM2-135M: keys and values, 30 layers, 4-byte values, worked by hand
 * @{
 */
constexpr lpl::core::usize kSmolGroupedCacheBytes = 2u * 30u * (3u * 64u) * 4u; ///< 46,080: 3 heads of 64 channels.
constexpr lpl::core::usize kSmolFullCacheBytes = 2u * 30u * (9u * 64u) * 4u;    ///< 138,240: 9 heads of 64 channels.
/** @} */

/**
 * @enum KeyValueSpread
 * @brief Which key/value head of a grouped model a full-head model repeats under each query head.
 */
enum class KeyValueSpread : lpl::core::u32 {
    ByGroup = 1u,  ///< Query head h repeats head h / (heads / keyValueHeads): consecutive query heads share one.
    ByModulo = 2u, ///< Query head h repeats head h mod keyValueHeads: the control, which interleaves them.
};

/**
 * @struct Thought
 * @brief What feeding the same tokens to two models gave.
 */
struct Thought {
    bool ran{false};              ///< Every model was allocated and fed every token.
    lpl::core::u32 differing{0u}; ///< Logits that differ between the two, over every step.
    lpl::core::u32 signature{0u}; ///< FNV-1a of the first model's logits, over every step.
};

/**
 * @brief The SmolLM2-135M shape with @p keyValueHeads key and value heads and one position.
 */
[[nodiscard]] lpl::infer::ModelConfig smolShape(lpl::core::u32 keyValueHeads)
{
    lpl::infer::ModelConfig shape{};

    shape.dim = kSmolWidth;
    shape.layers = kSmolLayers;
    shape.heads = kSmolHeads;
    shape.keyValueHeads = keyValueHeads;
    shape.ffnHidden = kSmolFeedForward;
    shape.contextLength = 1u;
    shape.vocabSize = kSmolVocabulary;
    return shape;
}

/**
 * @brief Bytes a cache of @p shape claims from a fresh arena, or 0 when it could not be allocated.
 */
[[nodiscard]] lpl::core::usize cacheBytes(const lpl::infer::ModelConfig &shape)
{
    lpl::infer::TensorArena arena{kCacheArenaBytes};
    lpl::infer::KvCache cache;
    const lpl::core::usize before = arena.used();

    return cache.allocate(arena, shape, lpl::infer::CachePolicy::Refuse) ? arena.used() - before : 0u;
}

/**
 * @brief Copies @p count bytes from @p from to @p to.
 */
void copyBytes(const lpl::core::u8 *from, lpl::core::u8 *to, lpl::core::u64 count)
{
    for (lpl::core::u64 i = 0u; i < count; ++i)
        to[i] = from[i];
}

/**
 * @brief Writes at @p to the image of a full-head key or value matrix whose query head h holds the rows of the
 * key/value head @p spread gives it in the grouped matrix of @p grouped at @p from.
 */
void spreadMatrix(const lpl::core::u8 *from, lpl::core::u8 *to, const lpl::infer::ModelConfig &grouped,
                  KeyValueSpread spread)
{
    const lpl::core::u64 rowBytes = lpl::infer::modelMatrixImageBytes(1u, grouped.dim);
    const lpl::core::u32 headChannels = grouped.dim / grouped.heads;
    const lpl::core::u32 headsPerGroup = grouped.heads / grouped.keyValueHeads;

    for (lpl::core::u32 head = 0u; head < grouped.heads; ++head)
    {
        const lpl::core::u32 source =
            spread == KeyValueSpread::ByGroup ? head / headsPerGroup : head % grouped.keyValueHeads;
        copyBytes(from + source * headChannels * rowBytes, to + head * headChannels * rowBytes,
                  headChannels * rowBytes);
    }
}

/**
 * @brief Writes at @p to the attention segment of the full-head model @p spread derives from the grouped segment at
 *        @p from: its two norms, then its query, key, value and output matrices, the key and value spread out.
 */
void spreadAttention(const lpl::core::u8 *from, lpl::core::u8 *to, const lpl::infer::ModelConfig &grouped,
                     KeyValueSpread spread)
{
    const lpl::core::u64 normsAndQuery =
        2u * grouped.dim * kWordBytes + lpl::infer::modelMatrixImageBytes(grouped.dim, grouped.dim);
    const lpl::core::u64 square = lpl::infer::modelMatrixImageBytes(grouped.dim, grouped.dim);
    const lpl::core::u64 narrow =
        lpl::infer::modelMatrixImageBytes(grouped.keyValueHeads * (grouped.dim / grouped.heads), grouped.dim);

    copyBytes(from, to, normsAndQuery);
    spreadMatrix(from + normsAndQuery, to + normsAndQuery, grouped, spread);
    spreadMatrix(from + normsAndQuery + narrow, to + normsAndQuery + square, grouped, spread);
    copyBytes(from + normsAndQuery + 2u * narrow, to + normsAndQuery + 2u * square, square);
}

/**
 * @brief Writes the image of the full-head model @p spread derives from the grouped image @p grouped: every segment
 *        copied, each attention's key and value rows repeated under the query heads that read them.
 * @return The image, on a word boundary, or nullptr when @p grouped is refused, holds no key/value heads, or the arena
 *         is short; @p size receives its bytes.
 */
[[nodiscard]] lpl::core::u8 *spreadKeyValueHeads(lpl::infer::TensorArena &arena, const lpl::core::u8 *grouped,
                                                 lpl::core::u32 groupedSize, KeyValueSpread spread,
                                                 lpl::core::u32 &size)
{
    lpl::infer::ModelSegment groupedSegments[kTableRoom];
    lpl::infer::ModelSegment segments[kTableRoom];
    const lpl::infer::ModelImageInspection inspection =
        lpl::infer::inspectModelImage(grouped, groupedSize, groupedSegments, kTableRoom);
    const lpl::infer::ModelConfig groupedShape = inspection.header.shape;
    if (inspection.verdict != lpl::infer::ModelImageVerdict::Accepted || groupedShape.keyValueHeads == 0u)
        return nullptr;

    lpl::infer::ModelImageHeader header = inspection.header;
    header.shape.keyValueHeads = header.shape.heads;
    header.imageBytes = lpl::infer::layOutModelImage(
        header.shape, lpl::infer::ModelTokenizerExtents{header.vocabTextBytes, header.mergeCount}, header.alignmentLog2,
        segments, kTableRoom);
    lpl::core::u32 *const words =
        header.imageBytes != 0u ? arena.claim<lpl::core::u32>(header.imageBytes / kWordBytes + 1u) : nullptr;
    if (words == nullptr)
        return nullptr;

    lpl::core::u8 *const image = reinterpret_cast<lpl::core::u8 *>(words);
    for (lpl::core::u32 index = 0u; index < header.segmentCount; ++index)
    {
        if (segments[index].role == lpl::infer::ModelSegmentRole::Attention)
            spreadAttention(grouped + groupedSegments[index].offset, image + segments[index].offset, groupedShape,
                            spread);
        else
            copyBytes(grouped + groupedSegments[index].offset, image + segments[index].offset, segments[index].length);
        segments[index].signature = lpl::infer::modelSegmentSignature(image, segments[index]);
        lpl::infer::writeModelSegmentEntry(image, index, segments[index]);
    }
    lpl::infer::writeModelImageHeader(image, header);
    size = static_cast<lpl::core::u32>(header.imageBytes);
    return image;
}

/**
 * @brief Feeds the same kThinkingSteps tokens to @p first and @p second, and compares their logits at every step.
 */
[[nodiscard]] Thought thinkSideBySide(lpl::infer::TensorArena &arena, const lpl::infer::Model &first,
                                      const lpl::infer::Model &second)
{
    const lpl::core::u32 vocabulary = first.config().vocabSize;
    lpl::infer::Transformer firstPass;
    lpl::infer::Transformer secondPass;
    lpl::infer::KvCache firstCache;
    lpl::infer::KvCache secondCache;
    lpl::math::Fixed32 *const firstLogits = arena.claim<lpl::math::Fixed32>(vocabulary);
    lpl::math::Fixed32 *const secondLogits = arena.claim<lpl::math::Fixed32>(vocabulary);
    Thought thought{};
    if (firstLogits == nullptr || secondLogits == nullptr || second.config().vocabSize != vocabulary ||
        !firstPass.allocate(arena, first) || !secondPass.allocate(arena, second) ||
        !firstCache.allocate(arena, first.config(), lpl::infer::CachePolicy::Refuse) ||
        !secondCache.allocate(arena, second.config(), lpl::infer::CachePolicy::Refuse))
        return thought;

    thought.signature = kFnv1aOffsetBasis;
    for (lpl::core::u32 position = 0u; position < kThinkingSteps; ++position)
    {
        const lpl::core::u32 token = (position * kTokenStride + 1u) % vocabulary;
        if (!firstPass.forward(token, position, firstCache, lpl::infer::VectorView{firstLogits, vocabulary}) ||
            !secondPass.forward(token, position, secondCache, lpl::infer::VectorView{secondLogits, vocabulary}))
            return Thought{};
        for (lpl::core::u32 i = 0u; i < vocabulary; ++i)
        {
            thought.differing += firstLogits[i].raw() != secondLogits[i].raw() ? 1u : 0u;
            thought.signature = (thought.signature ^ static_cast<lpl::core::u32>(firstLogits[i].raw())) * kFnv1aPrime;
        }
    }
    thought.ran = true;
    return thought;
}

} // namespace

/**
 * @brief A shape needs key/value heads that divide its heads: none, or 4 for 6 heads, is refused; 2 or 6 for 6 is not.
 */
LPL_TEST(a_shape_needs_key_value_heads_that_divide_its_heads)
{
    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();
    shape.vocabSize = 1u;
    shape.heads = kGroupedHeads;

    shape.keyValueHeads = 0u;
    const bool noneRefused = !shape.valid() && shape.headsPerKeyValueHead() == 0u;
    shape.keyValueHeads = 4u;
    const bool fourRefused = !shape.valid();
    shape.keyValueHeads = kGroupedKeyValueHeads;
    const bool twoAccepted = shape.valid() && shape.keyValueWidth() == 16u && shape.headsPerKeyValueHead() == 3u;
    shape.keyValueHeads = kGroupedHeads;
    const bool sixAccepted = shape.valid() && shape.keyValueWidth() == 48u && shape.headsPerKeyValueHead() == 1u;

    test.check(noneRefused, "a shape without key/value heads is refused, no query head reading any");
    test.check(fourRefused, "so is one of 4 key/value heads for 6 heads");
    test.check(twoAccepted, "2 for 6 holds: keys 16 wide, 3 query heads to each");
    test.check(sixAccepted, "and 6 for 6, one key/value head to each query head");
}

/**
 * @brief A token fed at position 5 leaves in the first layer's cache its key turned by that position: the key projected
 *        from its normed embedding, then each of its 2 key/value heads rotated as a query head is.
 *
 * @details Recomputed apart from attentionBlock from the steps it is made of, each tested on its own: rmsNorm, the
 *          quantised projection, applyRotaryEmbedding. The key left unturned is the control, which must differ.
 */
LPL_TEST(the_cache_holds_each_key_turned_by_its_position)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;
    lpl::infer::Transformer pass;
    lpl::infer::KvCache cache;
    lpl::math::Fixed32 residual[kParityWidth] = {};
    lpl::math::Fixed32 normed[kParityWidth] = {};
    lpl::math::Fixed32 unturned[kGroupedKeyWidth] = {};
    lpl::math::Fixed32 turned[kGroupedKeyWidth] = {};
    lpl::infer::QuantBlock blocks[lpl::infer::quantBlockCount(kParityWidth)] = {};

    test.check(lpl::infer::buildParityVocab(arena, vocab), "the canonical vocabulary builds");
    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();
    shape.vocabSize = vocab.size();
    shape.heads = kGroupedHeads;
    shape.keyValueHeads = kGroupedKeyValueHeads;
    lpl::math::Fixed32 *const logits = arena.claim<lpl::math::Fixed32>(shape.vocabSize);
    test.check(model.synthesise(arena, shape, vocab, lpl::infer::parityModelSeed()) && logits != nullptr &&
                   pass.allocate(arena, model) &&
                   cache.allocate(arena, model.config(), lpl::infer::CachePolicy::Refuse),
               "a model of 6 heads for 2 key/value heads builds, with its pass and its cache");
    if (!model.ready() || logits == nullptr || cache.capacity() == 0u)
        return;
    test.check(pass.forward(kTurnedToken, kTurnedPosition, cache, lpl::infer::VectorView{logits, shape.vocabSize}),
               "token 7 is fed at position 5");

    model.embed(kTurnedToken, lpl::infer::VectorView{residual, kParityWidth});
    lpl::infer::rmsNorm(lpl::infer::ConstVectorView{residual, kParityWidth}, model.layer(0u).attentionNorm,
                        shape.normEpsilon, lpl::infer::VectorView{normed, kParityWidth});
    lpl::infer::quantiseRow(normed, kParityWidth, blocks);
    lpl::infer::quantisedMatVecPrepared(model.layer(0u).key, blocks,
                                        lpl::infer::VectorView{unturned, kGroupedKeyWidth});
    for (lpl::core::u32 i = 0u; i < kGroupedKeyWidth; ++i)
        turned[i] = unturned[i];
    lpl::infer::applyRotaryEmbedding(lpl::infer::VectorView{turned, kGroupedKeyWidth}, kGroupedKeyValueHeads,
                                     kGroupedKeyWidth / kGroupedKeyValueHeads, model.rotaryFrequencies(),
                                     kTurnedPosition);

    const lpl::infer::VectorView cached = cache.key(0u, 0u);
    lpl::core::u32 differingFromTurned = cached.count == kGroupedKeyWidth ? 0u : kGroupedKeyWidth;
    lpl::core::u32 differingFromUnturned = 0u;
    for (lpl::core::u32 i = 0u; i < kGroupedKeyWidth && i < cached.count; ++i)
    {
        differingFromTurned += cached.at(i).raw() != turned[i].raw() ? 1u : 0u;
        differingFromUnturned += cached.at(i).raw() != unturned[i].raw() ? 1u : 0u;
    }
    test.check(differingFromTurned == 0u, "the cached key is the projected key turned by position 5, bit for bit");
    test.check(differingFromUnturned != 0u, "and not the key left unturned");
    test.measure("unturned_differing_channels", differingFromUnturned);
}

/**
 * @brief The cache of SmolLM2-135M's shape holds 46,080 bytes per position, its 3 key/value heads, and 138,240 with 9:
 *        it shrinks by heads / keyValueHeads.
 */
LPL_TEST(the_cache_of_smollm2_135m_s_shape_holds_46080_bytes_a_position)
{
    const lpl::core::usize grouped = cacheBytes(smolShape(kSmolKeyValueHeads));
    const lpl::core::usize full = cacheBytes(smolShape(kSmolHeads));

    test.check(grouped == kSmolGroupedCacheBytes, "3 key/value heads of 64 channels: 46,080 bytes a position");
    test.check(full == kSmolFullCacheBytes, "9 of them: 138,240 bytes");
    test.measure("grouped_cache_bytes", static_cast<lpl::core::u32>(grouped));
    test.measure("full_cache_bytes", static_cast<lpl::core::u32>(full));
}

/**
 * @brief A model of 6 heads reading 2 key/value heads gives, bit for bit, the logits of the full-head model whose image
 *        repeats each key/value head's rows under the 3 consecutive query heads that read it; repeated under every
 *        other query head instead, the logits differ.
 *
 * @details The full-head model is built apart from the grouped attention: its image is the grouped one rewritten by
 *          this test, and it runs one key/value head per query head, the path every model ran before. Quantised rows
 *          copied are the same bytes, so the identity is exact: each query head dots the same keys and blends the same
 *          values in both.
 */
LPL_TEST(a_grouped_model_thinks_what_its_heads_spread_out_think)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model grouped;
    lpl::infer::Model spreadByGroup;
    lpl::infer::Model spreadByModulo;
    lpl::core::u32 groupedSize = 0u;
    lpl::core::u32 byGroupSize = 0u;
    lpl::core::u32 byModuloSize = 0u;

    test.check(lpl::infer::buildParityVocab(arena, vocab), "the canonical vocabulary builds");
    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();
    shape.vocabSize = vocab.size();
    shape.heads = kGroupedHeads;
    shape.keyValueHeads = kGroupedKeyValueHeads;
    test.check(grouped.synthesise(arena, shape, vocab, lpl::infer::parityModelSeed()),
               "a model of 6 heads for 2 builds");
    test.check(grouped.ready() && grouped.layer(0u).key.rows == 16u && grouped.layer(0u).value.rows == 16u &&
                   grouped.layer(0u).query.rows == 48u,
               "its key and value matrices are 2 heads of 8 channels wide, its query 6");

    const lpl::core::u8 *const image =
        grouped.ready() ? lpl::infer::tests::writeModelImage(
                              arena, grouped, lpl::infer::kModelImageDefaultAlignmentLog2, 0u, groupedSize) :
                          nullptr;
    const lpl::core::u8 *const byGroup =
        image != nullptr ? spreadKeyValueHeads(arena, image, groupedSize, KeyValueSpread::ByGroup, byGroupSize) :
                           nullptr;
    const lpl::core::u8 *const byModulo =
        image != nullptr ? spreadKeyValueHeads(arena, image, groupedSize, KeyValueSpread::ByModulo, byModuloSize) :
                           nullptr;
    test.check(byGroup != nullptr && byModulo != nullptr, "its image is written, and spread out two ways");
    if (byGroup == nullptr || byModulo == nullptr)
        return;

    test.check(spreadByGroup.view(arena, byGroup, byGroupSize).verdict == lpl::infer::ModelReadVerdict::Read &&
                   spreadByModulo.view(arena, byModulo, byModuloSize).verdict == lpl::infer::ModelReadVerdict::Read &&
                   spreadByGroup.config().keyValueHeads == kGroupedHeads,
               "both spread-out images read as models of 6 key/value heads");
    if (!spreadByGroup.ready() || !spreadByModulo.ready())
        return;

    const Thought same = thinkSideBySide(arena, grouped, spreadByGroup);
    const Thought control = thinkSideBySide(arena, grouped, spreadByModulo);
    test.check(same.ran && control.ran, "all three are fed 12 tokens");
    test.check(same.ran && same.differing == 0u,
               "query head h reads key/value head h / 3: every logit is the spread-out model's, bit for bit");
    test.check(control.ran && control.differing != 0u,
               "the control differs: rows repeated under every other query head change the logits");
    test.measureHexadecimal("grouped_logit_signature", same.signature);
    test.measure("control_differing_logits", control.differing);
}
