#include <lpl/convert/Pairing.hpp>
#include <lpl/infer/Attention.hpp>
#include <lpl/testing/Test.hpp>

#include <cmath>
#include <vector>

LPL_TEST_SUITE(convert_pairing);

namespace {

constexpr std::uint32_t kHeadDim = 8u;       ///< Channels of a head: four pairs.
constexpr std::uint32_t kQueryHeads = 9u;    ///< SmolLM2's query heads.
constexpr std::uint32_t kKeyValueHeads = 3u; ///< And its key/value heads.
constexpr std::uint32_t kColumns = 16u;      ///< Input width.
constexpr std::uint32_t kPosition = 5u;      ///< A position past 0, so that every pair turns.
constexpr double kTwoPi = 6.283185307179586; ///< A whole turn, in radians.
constexpr double kExact = 1e-12;             ///< Two double computations of one value agree within this.
constexpr double kCordicTolerance = 5e-3;    ///< Ring 0's Q16.16 and CORDIC turn a value of 10 within this.

/**
 * @brief How far pair @p pair turns per position: a frequency of its own for each pair, in radians.
 */
[[nodiscard]] double frequencyOf(std::uint32_t pair) { return 0.3 + 0.17 * static_cast<double>(pair); }

/**
 * @brief A made-up weight of row @p row, column @p column: distinct enough that a wrong row shows.
 */
[[nodiscard]] double weightOf(std::uint32_t row, std::uint32_t column)
{
    return static_cast<double>(static_cast<std::int32_t>((row * 31u + column * 7u) % 17u) - 8) / 8.0 +
           0.001 * static_cast<double>(row);
}

/**
 * @brief The made-up input.
 */
[[nodiscard]] double inputOf(std::uint32_t column) { return 0.25 + 0.05 * static_cast<double>(column); }

/**
 * @brief Each row of @p rows times the input, the row of the source being sourceRowOf(row) for @p pairing.
 */
[[nodiscard]] std::vector<double> projected(std::uint32_t rows, lpl::convert::RotaryPairing pairing)
{
    std::vector<double> out(rows, 0.0);

    for (std::uint32_t row = 0u; row < rows; ++row)
    {
        const std::uint32_t source = lpl::convert::sourceRowOf(row, kHeadDim, pairing);
        for (std::uint32_t column = 0u; column < kColumns; ++column)
            out[row] += weightOf(source, column) * inputOf(column);
    }
    return out;
}

/**
 * @brief The plain product of the source's rows and the input.
 */
[[nodiscard]] std::vector<double> sourceProduct(std::uint32_t rows)
{
    std::vector<double> out(rows, 0.0);

    for (std::uint32_t row = 0u; row < rows; ++row)
    {
        for (std::uint32_t column = 0u; column < kColumns; ++column)
            out[row] += weightOf(row, column) * inputOf(column);
    }
    return out;
}

/**
 * @brief Turns each head of @p values at kPosition, pair j made of channels @p first(j) and @p second(j).
 */
template <typename First, typename Second> void turn(std::vector<double> &values, First first, Second second)
{
    for (std::uint32_t head = 0u; head * kHeadDim < values.size(); ++head)
    {
        for (std::uint32_t pair = 0u; pair < kHeadDim / 2u; ++pair)
        {
            const double angle = frequencyOf(pair) * kPosition;
            const std::uint32_t a = head * kHeadDim + first(pair);
            const std::uint32_t b = head * kHeadDim + second(pair);
            const double x = values[a];
            const double y = values[b];
            values[a] = x * std::cos(angle) - y * std::sin(angle);
            values[b] = x * std::sin(angle) + y * std::cos(angle);
        }
    }
}

/**
 * @brief Turns @p values as a model of half-split pairs does: channels j and j + d/2.
 */
void turnHalfSplit(std::vector<double> &values)
{
    turn(values, [](std::uint32_t pair) { return pair; }, [](std::uint32_t pair) { return pair + kHeadDim / 2u; });
}

/**
 * @brief Turns @p values as ring 0 does: channels 2j and 2j + 1.
 */
void turnAdjacent(std::vector<double> &values)
{
    turn(values, [](std::uint32_t pair) { return 2u * pair; }, [](std::uint32_t pair) { return 2u * pair + 1u; });
}

/**
 * @brief @p halfSplit's channels laid out as ring 0 pairs them: channel 2j + s takes channel s d/2 + j, worked apart
 *        from sourceRowOf.
 */
[[nodiscard]] std::vector<double> laidOutAdjacent(const std::vector<double> &halfSplit)
{
    std::vector<double> out(halfSplit.size(), 0.0);

    for (std::uint32_t head = 0u; head * kHeadDim < halfSplit.size(); ++head)
    {
        for (std::uint32_t pair = 0u; pair < kHeadDim / 2u; ++pair)
        {
            out[head * kHeadDim + 2u * pair] = halfSplit[head * kHeadDim + pair];
            out[head * kHeadDim + 2u * pair + 1u] = halfSplit[head * kHeadDim + kHeadDim / 2u + pair];
        }
    }
    return out;
}

/**
 * @brief The largest difference between @p lhs and @p rhs, or infinity when their sizes differ.
 */
[[nodiscard]] double largestDifference(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    if (lhs.size() != rhs.size())
        return INFINITY;
    double largest = 0.0;
    for (std::size_t i = 0u; i < lhs.size(); ++i)
        largest = std::fmax(largest, std::fabs(lhs[i] - rhs[i]));
    return largest;
}

/**
 * @brief The score of query head @p queryHead against its key/value head, the dot product of their channels.
 */
[[nodiscard]] double scoreOf(const std::vector<double> &query, const std::vector<double> &key, std::uint32_t queryHead)
{
    const std::uint32_t keyHead = queryHead / (kQueryHeads / kKeyValueHeads);
    double score = 0.0;

    for (std::uint32_t channel = 0u; channel < kHeadDim; ++channel)
        score += query[queryHead * kHeadDim + channel] * key[keyHead * kHeadDim + channel];
    return score;
}

} // namespace

/**
 * @brief A model of half-split pairs, its query and key rows read through sourceRowOf then turned by adjacent pairs,
 *        computes the projections, the turned values laid out apart, and the attention scores of the source turned by
 *        half-split pairs; a model of adjacent pairs keeps its rows.
 */
LPL_TEST(rows_read_through_the_pairing_turn_as_the_source_turns_them)
{
    const std::uint32_t queryRows = kQueryHeads * kHeadDim;
    const std::uint32_t keyRows = kKeyValueHeads * kHeadDim;

    std::vector<double> sourceQuery = sourceProduct(queryRows);
    std::vector<double> sourceKey = sourceProduct(keyRows);
    turnHalfSplit(sourceQuery);
    turnHalfSplit(sourceKey);
    std::vector<double> query = projected(queryRows, lpl::convert::RotaryPairing::HalfSplit);
    std::vector<double> key = projected(keyRows, lpl::convert::RotaryPairing::HalfSplit);
    turnAdjacent(query);
    turnAdjacent(key);

    test.check(largestDifference(query, laidOutAdjacent(sourceQuery)) < kExact,
               "nine query heads turned by adjacent pairs are the source's turned by half-split pairs, laid out apart");
    test.check(largestDifference(key, laidOutAdjacent(sourceKey)) < kExact,
               "and so are three key heads, each permuted within itself");
    double scoreDifference = 0.0;
    for (std::uint32_t head = 0u; head < kQueryHeads; ++head)
        scoreDifference =
            std::fmax(scoreDifference, std::fabs(scoreOf(query, key, head) - scoreOf(sourceQuery, sourceKey, head)));
    test.check(scoreDifference < kExact, "so every query head scores its key/value head as the source does");

    std::vector<double> unpermuted = projected(queryRows, lpl::convert::RotaryPairing::Adjacent);
    test.check(largestDifference(unpermuted, sourceProduct(queryRows)) == 0.0,
               "a model of adjacent pairs keeps every row where it is");
    std::vector<double> notPermuted = sourceProduct(queryRows);
    turnAdjacent(notPermuted);
    test.check(largestDifference(notPermuted, laidOutAdjacent(sourceQuery)) > 0.1,
               "while half-split rows turned by adjacent pairs, unpermuted, are not the source's: the control");
}

/**
 * @brief Ring 0's own rotation, Q16.16 and CORDIC, turns the permuted rows as the half-split source turns, within
 *        what its arithmetic costs: the convention the pairing serves is the engine's.
 */
LPL_TEST(ring_0_s_rotation_turns_the_permuted_rows_as_the_source_turns)
{
    const std::uint32_t queryRows = kQueryHeads * kHeadDim;
    std::vector<double> sourceQuery = sourceProduct(queryRows);
    turnHalfSplit(sourceQuery);
    const std::vector<double> expected = laidOutAdjacent(sourceQuery);
    const std::vector<double> query = projected(queryRows, lpl::convert::RotaryPairing::HalfSplit);

    std::vector<lpl::math::Fixed32> values(queryRows);
    for (std::uint32_t i = 0u; i < queryRows; ++i)
        values[i] = lpl::math::Fixed32::fromRaw(static_cast<std::int32_t>(std::lround(query[i] * 65536.0)));
    std::uint32_t turns[kHeadDim / 2u];
    for (std::uint32_t pair = 0u; pair < kHeadDim / 2u; ++pair)
        turns[pair] = static_cast<std::uint32_t>(std::llround(frequencyOf(pair) / kTwoPi * 4294967296.0));
    lpl::infer::applyRotaryEmbedding(lpl::infer::VectorView{values.data(), queryRows}, kQueryHeads, kHeadDim,
                                     lpl::infer::RotaryFrequencies{turns, kHeadDim / 2u}, kPosition);

    double largest = 0.0;
    for (std::uint32_t i = 0u; i < queryRows; ++i)
        largest = std::fmax(largest, std::fabs(static_cast<double>(values[i].raw()) / 65536.0 - expected[i]));
    test.check(largest < kCordicTolerance, "every channel within 5e-3 of the source's, turned by half-split pairs");
    test.measure("largest_difference_millionths", static_cast<std::uint32_t>(largest * 1e6));
}
