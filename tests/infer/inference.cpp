#include <lpl/infer/Attention.hpp>
#include <lpl/infer/Inference.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/infer/Tokenizer.hpp>
#include <lpl/math/FixedMath.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(inference);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kWideNormValues = 64u;
constexpr lpl::core::u32 kEpsilonOneInHundredThousand = 42950u; ///< 1e-5 in units of 2^-32, rounded.
constexpr lpl::core::u32 kQuarterTurn = 0x40000000u;            ///< A quarter turn, in units of 2^-32 turn.
constexpr lpl::core::u32 kEighthTurn = 0x20000000u;             ///< An eighth of a turn, in units of 2^-32 turn.
constexpr lpl::core::u32 kGoldenTurn = 0x9E3779B9u;    ///< 2^32 / phi, rounded: a turn no power of two divides.
constexpr lpl::core::u32 kOneRadianTurn = 683565276u;  ///< One radian, 2^32 / (2 pi) rounded, the seed's pair 0.
constexpr lpl::core::u32 kHalfRadianTurn = 341782638u; ///< Half a radian, 2^31 / (2 pi) rounded, the seed's pair 1.
constexpr lpl::core::u32 kPositionPastTheOldOverflow = 1u << 20; ///< Past 32768, where the angle in Q16.16 overflowed.
constexpr lpl::core::u32 kRotaryRoom = 8u;                       ///< Values the rotation tests turn at most.
constexpr lpl::core::i64 kCordicRawUnits = 16;    ///< CORDIC's error against an expected value rounded to a unit.
constexpr lpl::core::u32 kParityPairs = 6u;       ///< The parity model's heads of 12 channels hold 6 pairs.
constexpr lpl::core::u32 kWideHeadPairs = 36u;    ///< A head of 72 channels holds 36 pairs.
constexpr lpl::core::u32 kWideHeadChannels = 72u; ///< One head that wide.

/**
 * @brief round(2^(32 - i) / (2 pi)) for the parity model's six pairs, computed apart from the code in 80-digit
 *        arithmetic: one radian per position halved per pair, in units of 2^-32 turn.
 */
constexpr lpl::core::u32 kParityTurns[kParityPairs] = {683565276u, 341782638u, 170891319u,
                                                       85445659u,  42722830u,  21361415u};

/**
 * @brief round(2^(32 - i) / (2 pi)) for pairs 28 to 35, computed apart from the code: 2.546, 1.273, 0.637, 0.318, then
 *        halved each pair.
 */
constexpr lpl::core::u32 kWideHeadTailTurns[8] = {3u, 1u, 1u, 0u, 0u, 0u, 0u, 0u};
constexpr lpl::core::u32 kWideHeadTailFirstPair = 28u; ///< The pair kWideHeadTailTurns starts at.

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
 * @brief Normalises @p count values of raw Q16.16 @p raws, each gain one, with @p epsilon, into @p out.
 */
void normaliseRaw(const lpl::core::i32 *raws, lpl::core::u32 count, lpl::core::u32 epsilon, lpl::math::Fixed32 *out)
{
    lpl::math::Fixed32 values[kWideNormValues] = {};
    lpl::math::Fixed32 gains[kWideNormValues] = {};

    for (lpl::core::u32 index = 0u; index < count; ++index)
    {
        values[index] = lpl::math::Fixed32::fromRaw(raws[index]);
        gains[index] = lpl::math::Fixed32::one();
    }
    lpl::infer::rmsNorm(lpl::infer::ConstVectorView{values, count}, lpl::infer::ConstVectorView{gains, count}, epsilon,
                        lpl::infer::VectorView{out, count});
}

/**
 * @brief Whether @p value lies within @p units raw units of @p expectedRaw.
 */
[[nodiscard]] bool withinRawUnits(lpl::math::Fixed32 value, lpl::core::i64 expectedRaw, lpl::core::i64 units)
{
    const lpl::core::i64 gap = static_cast<lpl::core::i64>(value.raw()) - expectedRaw;
    return gap <= units && gap >= -units;
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

/**
 * @brief Builds the canonical vocabulary and a one-layer model of one head of kWideHeadChannels, in @p arena.
 *
 * @return Whether the model was built.
 */
[[nodiscard]] bool synthesiseWideHeadModel(lpl::infer::TensorArena &arena, lpl::infer::Vocab &vocab,
                                           lpl::infer::Model &model)
{
    if (!lpl::infer::buildParityVocab(arena, vocab))
        return false;

    lpl::infer::ModelConfig shape = lpl::infer::parityModelConfig();

    shape.vocabSize = vocab.size();
    shape.dim = kWideHeadChannels;
    shape.heads = 1u;
    shape.layers = 1u;
    return model.synthesise(arena, shape, vocab, lpl::infer::parityModelSeed());
}

/**
 * @brief Turns @p heads heads of @p pairs pairs, each pair (1, 0), by @p turns at @p position, into @p out: pair i of
 *        each head becomes the cosine and the sine of its angle.
 */
void turnUnitPairs(const lpl::core::u32 *turns, lpl::core::u32 pairs, lpl::core::u32 heads, lpl::core::u32 position,
                   lpl::math::Fixed32 *out)
{
    const lpl::core::u32 count = heads * pairs * 2u;

    for (lpl::core::u32 index = 0u; index < count; ++index)
        out[index] = index % 2u == 0u ? lpl::math::Fixed32::one() : lpl::math::Fixed32::zero();
    lpl::infer::applyRotaryEmbedding(lpl::infer::VectorView{out, count}, heads, pairs * 2u,
                                     lpl::infer::RotaryFrequencies{turns, pairs}, position);
}

/**
 * @brief Whether the @p pair -th pair of @p values lies within CORDIC's error of raw @p cosineRaw and @p sineRaw.
 */
[[nodiscard]] bool pairTurnedTo(const lpl::math::Fixed32 *values, lpl::core::u32 pair, lpl::core::i64 cosineRaw,
                                lpl::core::i64 sineRaw)
{
    return withinRawUnits(values[2u * pair], cosineRaw, kCordicRawUnits) &&
           withinRawUnits(values[2u * pair + 1u], sineRaw, kCordicRawUnits);
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

    lpl::infer::rmsNorm(lpl::infer::ConstVectorView{values, 4u}, lpl::infer::ConstVectorView{gains, 4u}, 0u,
                        lpl::infer::VectorView{normed, 4u});

    const lpl::math::Fixed32 meanSquare = lpl::math::Fixed32::fromRaw(dotProduct(normed, normed, 4u).raw() / 4);
    const lpl::math::Fixed32 rootMeanSquare = lpl::math::fixedSqrt(meanSquare);

    test.check((rootMeanSquare - lpl::math::Fixed32::one()).abs() <
                   lpl::math::Fixed32::fromRaw(lpl::math::Fixed32::kOne / 16),
               "rmsNorm leaves a vector with unit root mean square");
    test.measureHexadecimal("root_mean_square", static_cast<lpl::core::u32>(rootMeanSquare.raw()));
}

/**
 * @brief A vector of root mean square 0.0035 keeps the bits its mean square holds below one raw unit.
 *
 * @details Raw 197 and 262: the mean of their squares is 53,726 in units of 2^-32, under one raw unit of Q16.16.
 *          Expected values are x / sqrt(mean(x^2)) in double precision, apart from the code: 55,699.6 and 74,077.6.
 */
LPL_TEST(normalising_a_small_vector_keeps_the_bits_its_mean_square_holds_below_one_raw_unit)
{
    const lpl::core::i32 raws[2] = {197, 262};
    lpl::math::Fixed32 normed[2] = {};

    normaliseRaw(raws, 2u, 0u, normed);
    test.check(withinRawUnits(normed[0], 55700, 2) && withinRawUnits(normed[1], 74078, 2),
               "0.003 and 0.004 normalise to 0.8499 and 1.1303");
    test.measure("small_first_raw", static_cast<lpl::core::u32>(normed[0].raw()));
}

/**
 * @brief A vector of root mean square 3535.5 normalises without wrapping.
 *
 * @details 3000 and -4000: the mean of their squares is 12,500,000, past the 32,768 a Q16.16 word holds. Expected
 *          values, in double precision apart from the code: 55,609.1 and -74,145.5.
 */
LPL_TEST(normalising_a_vector_whose_rms_passes_181_does_not_wrap)
{
    const lpl::core::i32 raws[2] = {3000 * 65536, -4000 * 65536};
    lpl::math::Fixed32 normed[2] = {};

    normaliseRaw(raws, 2u, 0u, normed);
    test.check(withinRawUnits(normed[0], 55609, 2) && withinRawUnits(normed[1], -74146, 2),
               "3000 and -4000 normalise to 0.8485 and -1.1314");
}

/**
 * @brief Sixty-four values of plus or minus 30,000 normalise to plus or minus one, though the sum of their squares
 *        passes 2^64 in units of 2^-32.
 */
LPL_TEST(normalising_a_wide_vector_near_the_range_does_not_overflow_its_sum)
{
    lpl::core::i32 raws[kWideNormValues] = {};
    lpl::math::Fixed32 normed[kWideNormValues] = {};

    for (lpl::core::u32 index = 0u; index < kWideNormValues; ++index)
        raws[index] = (index % 2u == 0u ? 30000 : -30000) * 65536;
    normaliseRaw(raws, kWideNormValues, 0u, normed);

    bool everyOne = true;
    for (lpl::core::u32 index = 0u; everyOne && index < kWideNormValues; ++index)
        everyOne = withinRawUnits(normed[index], index % 2u == 0u ? 65536 : -65536, 1);
    test.check(everyOne, "each normalises to plus or minus one");
}

/**
 * @brief The model's epsilon is added to the mean of the squares before its root.
 *
 * @details Two values of 2^-8: the mean of their squares is 65,536 in units of 2^-32, and an epsilon of 1e-5 is
 *          42,950 of them. The root of 108,486 makes each value 0.77724, raw 50,937.0 in double precision apart from
 *          the code; without the epsilon, each is exactly one.
 */
LPL_TEST(the_model_s_epsilon_is_added_to_the_mean_square_before_its_root)
{
    const lpl::core::i32 raws[2] = {256, 256};
    lpl::math::Fixed32 withEpsilon[2] = {};
    lpl::math::Fixed32 without[2] = {};

    normaliseRaw(raws, 2u, kEpsilonOneInHundredThousand, withEpsilon);
    normaliseRaw(raws, 2u, 0u, without);
    test.check(withinRawUnits(withEpsilon[0], 50937, 2), "an epsilon of 1e-5 makes 2^-8 normalise to 0.77724");
    test.check(without[0] == lpl::math::Fixed32::one(), "and none makes it exactly one");
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
    const lpl::core::u32 turns[2] = {kOneRadianTurn, kHalfRadianTurn};
    const lpl::infer::RotaryFrequencies table{turns, 2u};

    lpl::infer::applyRotaryEmbedding(lpl::infer::VectorView{first, 4u}, 1u, 4u, table, 5u);
    lpl::infer::applyRotaryEmbedding(lpl::infer::VectorView{second, 4u}, 1u, 4u, table, 5u);

    const lpl::math::Fixed32 after = dotProduct(first, second, 4u);

    test.check((before - after).abs() < lpl::math::Fixed32::fromRaw(lpl::math::Fixed32::kOne / 64),
               "rotating both vectors by the same position leaves their dot product alone");
    test.measureHexadecimal("dot_product_after", static_cast<lpl::core::u32>(after.raw()));
}

/**
 * @brief Each pair turns by its own entry of the model's table, in every head: at position 1, a quarter turn takes
 *        (1, 0) to (0, 1), and an eighth to (0.70711, 0.70711).
 *
 * @details Expected values in double precision, apart from the code: 65536 times the cosine and the sine of 2 pi / 4,
 *          0 and 65536, and of 2 pi / 8, 46340.95 twice. Within kCordicRawUnits: over every raw angle, at both ends of
 *          the turns that floor to it, CORDIC's sixteen steps and the angle floored to a raw unit err by at most 16.37
 *          raw units against the exact value in double precision, and by at most 16 against it rounded to a unit; by 4
 *          here.
 */
LPL_TEST(each_pair_turns_by_its_own_frequency_in_every_head)
{
    const lpl::core::u32 turns[2] = {kQuarterTurn, kEighthTurn};
    lpl::math::Fixed32 values[kRotaryRoom] = {};

    turnUnitPairs(turns, 2u, 2u, 1u, values);
    test.check(pairTurnedTo(values, 0u, 0, 65536) && pairTurnedTo(values, 2u, 0, 65536),
               "a quarter turn takes (1, 0) to (0, 1), in each head");
    test.check(pairTurnedTo(values, 1u, 46341, 46341) && pairTurnedTo(values, 3u, 46341, 46341),
               "an eighth takes it to (0.70711, 0.70711), in each head");
    test.measure("quarter_turn_cosine_raw", values[0].raw());
    test.measure("eighth_turn_sine_raw", values[3].raw());
}

/**
 * @brief At position 2^20, past the 32768 where an angle in Q16.16 radians overflowed, each pair turns by its frequency
 *        times the position, modulo a turn, exactly.
 *
 * @details The angle in turns is (2^20 f mod 2^32) / 2^32, exact in double precision since 2^20 f stays below 2^53,
 *          computed apart from the code. For f = 0x9E3779B9 it is 2489 / 4096 = 0.607666015625 of a turn, 3.818078181
 *          rad, whose cosine and sine are -51103.52 and -41029.23 raw units. For one radian per position rounded, f =
 *          683565276, it is 220 / 4096 = 0.0537109375 of a turn, 0.337475773 rad: 61839.34 and 21699.38 raw units. That
 *          angle is 0.00065 rad from 2^20 rad modulo 2 pi, which is the entry's rounding, 0.42 of 2^-32 turn, times the
 *          position: the arithmetic adds no error to the frequency it is given. Both angles are whole multiples of
 * 2^-12 turn, so one position further, at 2^20 + 1, 0x9E3779B9 turns by 0x39C779B9 units, whose low 20 bits
 * count: 1.418114951 rad, whose cosine and sine are 9967.30 and 64773.61 raw units, where 0x39C00000 alone would give
 * 10013.72.
 */
LPL_TEST(a_position_past_32768_turns_by_its_frequency_times_the_position_modulo_a_turn)
{
    const lpl::core::u32 turns[2] = {kGoldenTurn, kOneRadianTurn};
    lpl::math::Fixed32 values[kRotaryRoom] = {};
    lpl::math::Fixed32 further[kRotaryRoom] = {};

    turnUnitPairs(turns, 2u, 1u, kPositionPastTheOldOverflow, values);
    turnUnitPairs(turns, 1u, 1u, kPositionPastTheOldOverflow + 1u, further);
    test.check(pairTurnedTo(values, 0u, -51104, -41029), "0x9E3779B9 turns (1, 0) to (-0.77978, -0.62606) at 2^20");
    test.check(pairTurnedTo(values, 1u, 61839, 21699), "one radian per position turns it to (0.94359, 0.33111)");
    test.check(pairTurnedTo(further, 0u, 9967, 64774),
               "and 0x9E3779B9 turns it to (0.15209, 0.98837) at 2^20 + 1, to the last bit of the turn");
    test.measure("golden_cosine_raw", values[0].raw());
    test.measure("one_radian_sine_raw", values[3].raw());
}

/**
 * @brief The model a seed derives turns pair i by 2^-i radians per position, its table rounded to 2^-32 turn.
 *
 * @details kParityTurns, round(2^(32 - i) / (2 pi)), computed apart from the code.
 */
LPL_TEST(the_seed_s_model_turns_pair_i_by_two_to_the_minus_i_radians)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;

    test.check(synthesiseCanonicalModel(arena, vocab, model, lpl::infer::parityModelSeed()),
               "the canonical model builds");

    const lpl::infer::RotaryFrequencies table = model.rotaryFrequencies();
    bool every = table.pairs == kParityPairs && table.turnsPerPosition != nullptr;

    for (lpl::core::u32 pair = 0u; every && pair < kParityPairs; ++pair)
        every = table.turnsPerPosition[pair] == kParityTurns[pair];
    test.check(table.pairs == kParityPairs, "one entry per pair of a head of 12 channels");
    test.check(every, "pair i turns by round(2^(32 - i) / (2 pi)) units of 2^-32 turn per position");
}

/**
 * @brief In a head of 72 channels, the seed's pairs 28 to 30 turn by 3, 1 and 1 units of 2^-32 turn per position, and
 *        every pair after them by none: what 2^(32 - i) / (2 pi) rounds to, not what a shift by 64 bits or more leaves.
 *
 * @details kWideHeadTailTurns, computed apart from the code.
 */
LPL_TEST(the_seed_s_pairs_past_the_thirtieth_turn_by_none)
{
    lpl::infer::TensorArena arena{lpl::infer::parityArenaBytes()};
    lpl::infer::Vocab vocab;
    lpl::infer::Model model;

    test.check(synthesiseWideHeadModel(arena, vocab, model), "a model of one head of 72 channels builds");

    const lpl::infer::RotaryFrequencies table = model.rotaryFrequencies();
    bool every = table.pairs == kWideHeadPairs && table.turnsPerPosition != nullptr;

    for (lpl::core::u32 pair = kWideHeadTailFirstPair; every && pair < kWideHeadPairs; ++pair)
        every = table.turnsPerPosition[pair] == kWideHeadTailTurns[pair - kWideHeadTailFirstPair];
    test.check(table.pairs == kWideHeadPairs, "one entry per pair, 36");
    test.check(every, "3, 1 and 1 for pairs 28 to 30, then 0");
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
    test.check(folded.blobViewed == 1u, "and read where it lies, it has them too");
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
