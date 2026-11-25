#include <lpl/convert/Requantise.hpp>
#include <lpl/testing/Test.hpp>

#include <cmath>
#include <cstring>
#include <limits>

LPL_TEST_SUITE(convert_requantise);

namespace {

constexpr double kUnit = 1.0 / 65536.0; ///< One unit of Q16.16.

/** A Q8_0 block's 34 bytes: its step's half bits, then its lanes. */
struct EightBitBlock {
    std::uint8_t bytes[lpl::convert::kEightBitBlockBytes]{};
};

/**
 * @brief A Q8_0 block of step @p halfBits whose lanes are all @p lane but the first, @p first.
 */
[[nodiscard]] EightBitBlock eightBitBlock(std::uint16_t halfBits, std::int8_t first, std::int8_t lane)
{
    EightBitBlock block;

    block.bytes[0] = static_cast<std::uint8_t>(halfBits & 0xFFu);
    block.bytes[1] = static_cast<std::uint8_t>(halfBits >> 8);
    for (std::uint32_t i = 0u; i < lpl::infer::kQuantBlockLanes; ++i)
        block.bytes[2u + i] = static_cast<std::uint8_t>(i == 0u ? first : lane);
    return block;
}

/**
 * @brief The raw word @p value rounds to, or a value no rounding gives when it is refused.
 */
[[nodiscard]] std::int64_t rounded(double value, lpl::convert::FixedRounding expected)
{
    std::int32_t raw = 0;

    return lpl::convert::roundToFixed(value, raw) == expected ? raw : std::numeric_limits<std::int64_t>::min();
}

/**
 * @brief The block @p block is taken as, and whether it was taken as @p expected.
 */
[[nodiscard]] bool takenAs(const EightBitBlock &block, lpl::convert::BlockTaking expected, lpl::infer::QuantBlock &out)
{
    return lpl::convert::takeEightBitBlock(block.bytes, out) == expected;
}

/**
 * @brief A tensor of @p type over the @p byteCount bytes at @p bytes.
 */
[[nodiscard]] lpl::convert::SourceTensor tensorOver(lpl::convert::TensorType type, const std::uint8_t *bytes,
                                                    std::uint64_t byteCount)
{
    lpl::convert::SourceTensor tensor;

    tensor.type = type;
    tensor.bytes = bytes;
    tensor.byteCount = byteCount;
    return tensor;
}

} // namespace

/**
 * @brief A value rounds to the nearest unit of 2^-16, half away from zero, and is refused when it is not finite or
 *        rounds past 2^31 - 1 units either way.
 */
LPL_TEST(a_weight_rounds_to_q16_16_half_away_from_zero_inside_the_symmetric_range)
{
    test.check(rounded(0.5 * kUnit, lpl::convert::FixedRounding::Rounded) == 1 &&
                   rounded(-0.5 * kUnit, lpl::convert::FixedRounding::Rounded) == -1,
               "half a unit rounds away from zero, either sign");
    test.check(rounded(1.5 * kUnit, lpl::convert::FixedRounding::Rounded) == 2 &&
                   rounded(2.5 * kUnit, lpl::convert::FixedRounding::Rounded) == 3 &&
                   rounded(-2.5 * kUnit, lpl::convert::FixedRounding::Rounded) == -3,
               "and not to the even unit");
    test.check(rounded(0.49 * kUnit, lpl::convert::FixedRounding::Rounded) == 0 &&
                   rounded(1.25, lpl::convert::FixedRounding::Rounded) == 81920,
               "below half a unit it rounds down, and 1.25 is 81920 units");
    test.check(rounded(32767.99999, lpl::convert::FixedRounding::Rounded) == 2147483647,
               "32767.99999 rounds to the largest word, 2^31 - 1");
    test.check(rounded(32767.999992370605, lpl::convert::FixedRounding::PastRange) == 0 &&
                   rounded(-32767.999992370605, lpl::convert::FixedRounding::PastRange) == 0,
               "2^31 - 1/2 units rounds past it, either sign, and is refused");
    test.check(rounded(-32768.0, lpl::convert::FixedRounding::PastRange) == 0,
               "-32768 is a word, but not one whose magnitude a block can take, and is refused");
    test.check(rounded(std::numeric_limits<double>::quiet_NaN(), lpl::convert::FixedRounding::NotFinite) == 0 &&
                   rounded(std::numeric_limits<double>::infinity(), lpl::convert::FixedRounding::NotFinite) == 0 &&
                   rounded(-std::numeric_limits<double>::infinity(), lpl::convert::FixedRounding::NotFinite) == 0,
               "NaN and either infinity are refused as not finite");
}

/**
 * @brief A Q8_0 block keeps its lanes and rounds its step's 127 d 2^16 to its magnitude, and is refused for a negative,
 *        NaN or infinite step, a magnitude past 2^31 - 1 or a lane of -128.
 */
LPL_TEST(a_q8_0_block_keeps_its_lanes_and_rounds_its_step_to_a_magnitude)
{
    lpl::infer::QuantBlock block{};

    test.check(takenAs(eightBitBlock(0x2E66u, 127, -3), lpl::convert::BlockTaking::Taken, block) &&
                   block.scaleRaw == 832104 && block.lanes[0] == 127 && block.lanes[1] == -3 && block.lanes[31] == -3,
               "a step of 0.0999755859375 is a magnitude of 832104 units, every lane kept");
    test.check(takenAs(eightBitBlock(0x1C01u, -127, 1), lpl::convert::BlockTaking::Taken, block) &&
                   block.scaleRaw == 32544 && block.lanes[0] == -127,
               "a step whose magnitude is 32543.75 units rounds to 32544");
    test.check(takenAs(eightBitBlock(0x5800u, 1, 0), lpl::convert::BlockTaking::Taken, block) &&
                   block.scaleRaw == 1065353216,
               "a step of 128 is 1065353216 units, inside the range");
    test.check(takenAs(eightBitBlock(0x7BFFu, 1, 0), lpl::convert::BlockTaking::PastRange, block),
               "the largest half, 65504, is past it and refused");
    test.check(takenAs(eightBitBlock(0xAE66u, 1, 0), lpl::convert::BlockTaking::ScaleRefused, block) &&
                   takenAs(eightBitBlock(0x8000u, 1, 0), lpl::convert::BlockTaking::ScaleRefused, block),
               "a negative step is refused, a negative zero too");
    test.check(takenAs(eightBitBlock(0x7C00u, 1, 0), lpl::convert::BlockTaking::ScaleRefused, block) &&
                   takenAs(eightBitBlock(0x7E00u, 1, 0), lpl::convert::BlockTaking::ScaleRefused, block),
               "an infinite or NaN step is refused");
    test.check(takenAs(eightBitBlock(0x2E66u, 1, -128), lpl::convert::BlockTaking::LanePastRange, block) &&
                   takenAs(eightBitBlock(0x2E66u, -128, 1), lpl::convert::BlockTaking::LanePastRange, block),
               "a lane of -128, last or first, is refused");
}

/**
 * @brief A lane stands for lane M / 127 in a product, unrounded, and a tensor's values are read as each type stores
 *        them.
 */
LPL_TEST(a_lane_and_a_source_value_read_as_their_formats_say)
{
    lpl::infer::QuantBlock block{};
    block.scaleRaw = 832104;
    block.lanes[0] = 127;
    block.lanes[1] = -1;

    test.check(lpl::convert::laneValue(block, 0u) == 832104.0 / 65536.0,
               "a lane of 127 stands for the magnitude itself");
    test.check(lpl::convert::laneValue(block, 1u) == -832104.0 / 127.0 / 65536.0,
               "a lane of -1 for minus a 127th of it, unrounded");

    const float real = 1.25f;
    std::uint8_t realBytes[4];
    std::memcpy(realBytes, &real, sizeof(real));
    const std::uint8_t halfBytes[] = {0x00u, 0x3Cu, 0x00u, 0xC0u};
    const std::uint8_t brainBytes[] = {0xA0u, 0x3Fu, 0x40u, 0xC0u};
    const EightBitBlock first = eightBitBlock(0x3C00u, 5, -3);
    const EightBitBlock second = eightBitBlock(0x3800u, 7, 2);
    std::uint8_t blocks[2u * lpl::convert::kEightBitBlockBytes];
    std::memcpy(blocks, first.bytes, lpl::convert::kEightBitBlockBytes);
    std::memcpy(blocks + lpl::convert::kEightBitBlockBytes, second.bytes, lpl::convert::kEightBitBlockBytes);

    test.check(lpl::convert::sourceValue(tensorOver(lpl::convert::TensorType::Float32, realBytes, 4u), 0u) == 1.25,
               "an F32 value is its float");
    test.check(lpl::convert::sourceValue(tensorOver(lpl::convert::TensorType::Float16, halfBytes, 4u), 0u) == 1.0 &&
                   lpl::convert::sourceValue(tensorOver(lpl::convert::TensorType::Float16, halfBytes, 4u), 1u) == -2.0,
               "an F16 value is its half, 0x3C00 one and 0xC000 minus two");
    test.check(
        lpl::convert::sourceValue(tensorOver(lpl::convert::TensorType::BrainFloat16, brainBytes, 4u), 0u) == 1.25 &&
            lpl::convert::sourceValue(tensorOver(lpl::convert::TensorType::BrainFloat16, brainBytes, 4u), 1u) == -3.0,
        "a BF16 value is the high half of a float, 0x3FA0 1.25 and 0xC040 minus three");
    const lpl::convert::SourceTensor eightBit =
        tensorOver(lpl::convert::TensorType::QuantisedEightBit, blocks, sizeof(blocks));
    test.check(lpl::convert::sourceValue(eightBit, 0u) == 5.0 && lpl::convert::sourceValue(eightBit, 1u) == -3.0,
               "a Q8_0 value is its lane times its block's step");
    test.check(lpl::convert::sourceValue(eightBit, 32u) == 3.5 && lpl::convert::sourceValue(eightBit, 33u) == 1.0,
               "the 33rd value is the second block's first, its step one half");
}

/**
 * @brief The error statistics sum the squares apart, keep the first largest error's block, and say a perfect copy, an
 *        error over zeros and a relative root mean square computed by hand.
 */
LPL_TEST(the_error_statistics_are_those_computed_by_hand)
{
    lpl::convert::ErrorStatistics error;

    test.check(error.relativeRootMeanSquare() == 0.0, "no value compared is no error");
    error.add(1.0, 1.5, 0u);
    error.add(2.0, 2.0, 0u);
    error.add(-1.0, -0.5, 1u);
    test.check(error.values == 3u && error.squaredError == 0.5 && error.squaredSource == 6.5,
               "three values: 0.5 of squared error over 6.5 of squared source");
    test.check(std::fabs(error.relativeRootMeanSquare() - 0.27735009811261456) < 1e-15,
               "a relative root mean square of sqrt(0.5 / 6.5)");
    test.check(error.largestError == 0.5 && error.worstBlock == 0u,
               "two errors of 0.5: the first one's block is the worst");
    error.add(0.0, 0.8, 2u);
    test.check(error.largestError == 0.8 && error.worstBlock == 2u, "a larger one moves it");

    lpl::convert::ErrorStatistics zeros;
    zeros.add(0.25, 0.0, 0u);
    test.check(std::isinf(zeros.relativeRootMeanSquare()), "an error over a source of zeros is infinite");
    lpl::convert::ErrorStatistics exact;
    exact.add(0.0, 0.0, 0u);
    exact.add(3.0, 3.0, 0u);
    test.check(exact.relativeRootMeanSquare() == 0.0, "and a perfect copy none");
}
