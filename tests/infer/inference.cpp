#include <lpl/infer/Attention.hpp>
#include <lpl/infer/Inference.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/infer/Tokenizer.hpp>
#include <lpl/math/FixedMath.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(inference);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;

[[nodiscard]] lpl::core::u32 textLength(const char *text)
{
    lpl::core::u32 length = 0u;

    while (text[length] != '\0')
        ++length;
    return length;
}

[[nodiscard]] lpl::math::Fixed32 dotProduct(const lpl::math::Fixed32 *lhs, const lpl::math::Fixed32 *rhs,
                                            lpl::core::u32 count)
{
    lpl::core::i64 total = 0;

    for (lpl::core::u32 index = 0u; index < count; ++index)
        total += (static_cast<lpl::core::i64>(lhs[index].raw()) * rhs[index].raw()) >> 16;
    return lpl::math::Fixed32::fromRaw(static_cast<lpl::core::i32>(total));
}

/**
 * @brief Builds the canonical vocabulary and the model derived from @p seed, in @p arena.
 *
 * @return Whether the model was built.
 */
[[nodiscard]] bool synthesiseCanonicalModel(lpl::infer::TensorArena &arena, lpl::infer::Vocab &vocab,
                                            lpl::infer::Model &model, lpl::core::u32 seed)
{
    if (!lpl::infer::buildParityVocab(arena, vocab))
        return false;

    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();

    shape.vocabSize = vocab.size();
    return model.synthesise(arena, shape, vocab, seed);
}

} // namespace

/**
 * @brief The exponential the softmax stands on, without libm: exact on the powers of two, the
 *        inverse of log2 on every power of two Q16.16 holds (2^15 is past its top, so exp2
 *        saturates there by design), and an underflow or an overflow clamps rather than wraps.
 */
LPL_TEST(exponential_needs_no_libm)
{
    test.check(lpl::math::fixedExp2(lpl::math::Fixed32::zero()) == lpl::math::Fixed32::one(), "2^0 is exactly one");
    test.check(lpl::math::fixedExp2(lpl::math::Fixed32::one()) == lpl::math::Fixed32::fromInt(2), "2^1 is exactly two");
    test.check(lpl::math::fixedExp2(lpl::math::Fixed32::fromInt(-1)) == lpl::math::Fixed32::half(),
               "2^-1 is exactly one half");
    test.check(lpl::math::fixedExp2(lpl::math::Fixed32::fromInt(10)) == lpl::math::Fixed32::fromInt(1024),
               "2^10 is exactly 1024: the integer part is a shift, not an approximation");

    bool inverts = true;

    for (lpl::core::u32 power = 0u; power <= 14u; ++power)
    {
        const lpl::core::u32 value = lpl::core::u32{1} << power;

        inverts = inverts && lpl::math::fixedExp2(lpl::math::fixedLog2(value)) ==
                                 lpl::math::Fixed32::fromInt(static_cast<lpl::core::i32>(value));
    }
    test.check(inverts, "exp2 undoes log2 on every power of two Fixed32 can hold");
    test.check(lpl::math::fixedExp2(lpl::math::Fixed32::fromInt(-40)) == lpl::math::Fixed32::zero(),
               "a hopeless score underflows to zero rather than wrapping");
    test.check(lpl::math::fixedExp2(lpl::math::Fixed32::fromInt(40)) == lpl::math::Fixed32::max(),
               "an impossible score saturates rather than wrapping");
}

/**
 * @brief Eight bits cost at most half a level of the block's own scale, which is what correct
 *        rounding means; anything more is a bug, not a precision limit.
 */
LPL_TEST(eight_bits_cost_half_a_level_at_most)
{
    lpl::infer::QuantBlock block{};
    lpl::math::Fixed32 row[lpl::infer::kQuantBlockLanes];

    for (lpl::core::u32 lane = 0u; lane < lpl::infer::kQuantBlockLanes; ++lane)
        row[lane] = lpl::math::Fixed32::fromRaw(static_cast<lpl::core::i32>(lane) * 1024 - 16384);
    lpl::infer::quantiseRow(row, lpl::infer::kQuantBlockLanes, &block);

    lpl::core::i32 worstError = 0;

    for (lpl::core::u32 lane = 0u; lane < lpl::infer::kQuantBlockLanes; ++lane)
    {
        const lpl::core::i32 error = (lpl::infer::dequantiseLane(block, lane) - row[lane]).abs().raw();

        if (error > worstError)
            worstError = error;
    }

    const lpl::core::i32 halfLevel = block.scaleRaw / (2 * lpl::infer::kQuantLevels) + 1;

    test.check(worstError <= halfLevel, "no lane is off by more than half a quantisation level");
    test.measure("worst_lane_error", worstError);

    lpl::math::Fixed32 zeros[lpl::infer::kQuantBlockLanes] = {};
    lpl::infer::QuantBlock empty{};

    lpl::infer::quantiseRow(zeros, lpl::infer::kQuantBlockLanes, &empty);
    test.check(empty.scaleRaw == 0 && lpl::infer::dequantiseLane(empty, 0u) == lpl::math::Fixed32::zero(),
               "an all-zero block quantises to zero instead of dividing by its own scale");
}

/**
 * @brief rmsNorm leaves a vector with unit root mean square.
 */
LPL_TEST(normalising_gives_a_unit_root_mean_square)
{
    lpl::math::Fixed32 values[4] = {lpl::math::Fixed32::fromInt(3), lpl::math::Fixed32::fromInt(-4),
                                    lpl::math::Fixed32::fromInt(0), lpl::math::Fixed32::fromInt(5)};
    lpl::math::Fixed32 gains[4] = {lpl::math::Fixed32::one(), lpl::math::Fixed32::one(), lpl::math::Fixed32::one(),
                                   lpl::math::Fixed32::one()};
    lpl::math::Fixed32 normed[4] = {};

    lpl::infer::rmsNorm(lpl::infer::ConstVectorView{values, 4u}, lpl::infer::ConstVectorView{gains, 4u},
                        lpl::infer::VectorView{normed, 4u});

    const lpl::math::Fixed32 meanSquare = lpl::math::Fixed32::fromRaw(dotProduct(normed, normed, 4u).raw() / 4);
    const lpl::math::Fixed32 rootMeanSquare = lpl::math::fixedSqrt(meanSquare);

    test.check((rootMeanSquare - lpl::math::Fixed32::one()).abs() <
                   lpl::math::Fixed32::fromRaw(lpl::math::Fixed32::kOne / 16),
               "rmsNorm leaves a vector with unit root mean square");
    test.measureHexadecimal("root_mean_square", static_cast<lpl::core::u32>(rootMeanSquare.raw()));
}

/**
 * @brief The rotation is what makes attention see relative position: two vectors rotated by the
 *        same position keep their dot product.
 */
LPL_TEST(a_shared_rotation_keeps_the_dot_product)
{
    lpl::math::Fixed32 first[4] = {
        lpl::math::Fixed32::half(), lpl::math::Fixed32::fromRaw(lpl::math::Fixed32::kOne / 4),
        lpl::math::Fixed32::fromRaw(-lpl::math::Fixed32::kOne / 8), lpl::math::Fixed32::half()};
    lpl::math::Fixed32 second[4] = {lpl::math::Fixed32::fromRaw(lpl::math::Fixed32::kOne / 3),
                                    lpl::math::Fixed32::half(), lpl::math::Fixed32::half(),
                                    lpl::math::Fixed32::fromRaw(-lpl::math::Fixed32::kOne / 5)};
    const lpl::math::Fixed32 before = dotProduct(first, second, 4u);

    lpl::infer::applyRotaryEmbedding(lpl::infer::VectorView{first, 4u}, 1u, 4u, 5u);
    lpl::infer::applyRotaryEmbedding(lpl::infer::VectorView{second, 4u}, 1u, 4u, 5u);

    const lpl::math::Fixed32 after = dotProduct(first, second, 4u);

    test.check((before - after).abs() < lpl::math::Fixed32::fromRaw(lpl::math::Fixed32::kOne / 64),
               "rotating both vectors by the same position leaves their dot product alone");
    test.measureHexadecimal("dot_product_after", static_cast<lpl::core::u32>(after.raw()));
}

/**
 * @brief Softmax turns scores into weights that sum to one and keep their order. Subtracting the
 *        maximum first is not cosmetic: without it, forty and twenty clamp to the same value.
 */
LPL_TEST(softmax_turns_scores_into_a_distribution)
{
    lpl::math::Fixed32 scores[5] = {lpl::math::Fixed32::fromInt(1), lpl::math::Fixed32::fromInt(3),
                                    lpl::math::Fixed32::fromInt(2), lpl::math::Fixed32::fromInt(-1),
                                    lpl::math::Fixed32::fromInt(0)};

    lpl::infer::softmaxInPlace(lpl::infer::VectorView{scores, 5u});

    lpl::core::i64 total = 0;

    for (const lpl::math::Fixed32 score : scores)
        total += score.raw();
    test.check(total > lpl::math::Fixed32::kOne - 64 && total <= lpl::math::Fixed32::kOne,
               "the weights sum to one, up to the format's resolution");
    test.check(scores[1] > scores[2] && scores[2] > scores[0] && scores[0] > scores[4] && scores[4] > scores[3],
               "the order of the scores survives");

    lpl::math::Fixed32 extreme[2] = {lpl::math::Fixed32::fromInt(20), lpl::math::Fixed32::fromInt(40)};

    lpl::infer::softmaxInPlace(lpl::infer::VectorView{extreme, 2u});
    test.check(extreme[1] > extreme[0], "a score of forty still beats a score of twenty after the exponential");
}

/**
 * @brief Gate P14 mind: the canonical model, derived from a seed, written out and read back,
 *        thinks the same thought on both targets, freely and inside a grammar that can spell two
 *        calls and not the forbidden one.
 *
 * @details The arena's occupancy is checked against its capacity, not recorded: LayerWeights holds
 *          nine views, a pointer each, so the bytes carved differ between a 64-bit host and i686.
 */
LPL_TEST(the_same_seed_thinks_the_same_thought)
{
    lpl::infer::MindFoldResult folded{};

    lpl::infer::foldMindState(folded);
    test.check(folded.vocabSize == 121u, "the canonical vocabulary is 121 tokens");
    test.check(folded.weightSignature != 0u && folded.tokenSignature != 0u, "the run produced signatures");
    test.check(folded.generated == lpl::infer::parityGeneratedTokens(),
               "the free run produced its whole budget of tokens");
    test.check(folded.draws == folded.generated, "one sampling draw per token: the stream is not consumed elsewhere");
    test.check(folded.blobReopened == 1u, "the model written out and read back has the same weights, bit for bit");
    test.check(folded.grammarComplete == 1u, "the constrained run spelled a whole call");
    test.check(folded.forbiddenReachable == 0u, "the grammar cannot spell the forbidden call");
    test.check(folded.arenaBytes > 0u && folded.arenaBytes < lpl::infer::parityArenaBytes(),
               "the whole run fits inside the arena it declared");

    lpl::infer::MindFoldResult again{};

    lpl::infer::foldMindState(again);
    test.check(again.weightSignature == folded.weightSignature && again.tokenSignature == folded.tokenSignature &&
                   again.residualSignature == folded.residualSignature && again.textSignature == folded.textSignature,
               "running it twice produces the same mind");

    test.measureHexadecimal("weight_signature", folded.weightSignature);
    test.measureHexadecimal("prompt_signature", folded.promptSignature);
    test.measureHexadecimal("logit_signature", folded.logitSignature);
    test.measureHexadecimal("residual_signature", folded.residualSignature);
    test.measureHexadecimal("token_signature", folded.tokenSignature);
    test.measureHexadecimal("constrained_signature", folded.constrainedSignature);
    test.measureHexadecimal("text_signature", folded.textSignature);
    test.measure("vocabulary", folded.vocabSize);
    test.measure("prompt_tokens", folded.promptTokens);
    test.measure("generated", folded.generated);
    test.measure("draws", folded.draws);
    test.measure("constrained_tokens", folded.constrainedTokens);
    test.measure("admitted_first", folded.admittedFirst);
    test.measure("blob_bytes", folded.blobBytes);
}

/**
 * @brief A different seed derives a different model, or the seed is decoration.
 */
LPL_TEST(a_different_seed_derives_a_different_model)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab canonicalVocab;
    lpl::infer::Vocab otherVocab;
    lpl::infer::Model canonical;
    lpl::infer::Model other;

    test.check(synthesiseCanonicalModel(arena, canonicalVocab, canonical, lpl::infer::parityModelSeed()) &&
                   synthesiseCanonicalModel(arena, otherVocab, other, lpl::infer::parityModelSeed() + 1u),
               "the models of two seeds build");
    test.check(other.fold(kFnv1aOffsetBasis) != canonical.fold(kFnv1aOffsetBasis), "and their weights differ");
}

/**
 * @brief The canonical prompt is spelled in the canonical vocabulary and decodes back to itself,
 *        and the free run completes. Weights derived from a seed have learnt nothing, so what they
 *        say is babble: the claim is that both targets babble identically.
 */
LPL_TEST(the_canonical_prompt_is_spelled_and_read_back)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;

    test.check(synthesiseCanonicalModel(arena, vocab, model, lpl::infer::parityModelSeed()),
               "the canonical model builds");

    lpl::infer::Tokenizer tokenizer{vocab};
    const char *const prompt = lpl::infer::parityPrompt();
    lpl::core::u32 tokens[64];
    lpl::core::u32 skipped = 0u;
    const lpl::core::u32 count = tokenizer.encode(prompt, textLength(prompt), tokens, 64u, skipped);

    test.check(skipped == 0u, "the canonical prompt is spellable in the canonical vocabulary");

    char decoded[128];
    const lpl::core::u32 bytes = tokenizer.decode(tokens, count, decoded, 128u);
    bool same = (bytes == textLength(prompt));

    for (lpl::core::u32 index = 0u; same && index < bytes; ++index)
        same = (decoded[index] == prompt[index]);
    test.check(same, "tokenising the prompt and decoding it back gives the prompt");

    lpl::infer::Inference inference;

    test.check(inference.initialise(arena, model, lpl::infer::CachePolicy::Refuse),
               "the inference facade claims its cache and scratch");

    lpl::infer::GenerationParams params{};

    params.sampler.topK = 4u;
    params.sampler.seed = lpl::infer::paritySamplerSeed();
    params.maxTokens = lpl::infer::parityGeneratedTokens();

    lpl::core::u32 generated[32];
    lpl::infer::GenerationReport report{};

    test.check(inference.generate(tokens, count, params, nullptr, generated, 32u, report), "the free run completes");
    test.measure("prompt_tokens", count);
    test.measure("generated", report.generated);
}
