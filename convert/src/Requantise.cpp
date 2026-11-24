#include <lpl/convert/Requantise.hpp>

#include <cmath>
#include <cstring>
#include <limits>

namespace lpl::convert {

namespace {

constexpr double kFixedUnits = 65536.0;          ///< Units of Q16.16 in 1.
constexpr double kLargestRaw = 2147483647.0;     ///< 2^31 - 1, the largest magnitude a block takes.
constexpr double kQuantLevels = 127.0;           ///< A lane's largest magnitude.
constexpr std::uint32_t kEightBitStepBytes = 2u; ///< The step that opens a Q8_0 block.

/**
 * @brief @p value rounded to the nearest integer, half away from zero.
 */
[[nodiscard]] double roundedHalfAway(double value) noexcept
{
    return value < 0.0 ? -std::floor(-value + 0.5) : std::floor(value + 0.5);
}

/**
 * @brief The little-endian IEEE 754 half at @p bytes, exactly: a sign, 5 bits of exponent biased by 15, 10 of fraction.
 */
[[nodiscard]] double halfAt(const std::uint8_t *bytes) noexcept
{
    const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8);
    const double sign = (bits & 0x8000u) != 0u ? -1.0 : 1.0;
    const std::uint32_t exponent = (bits >> 10) & 0x1Fu;
    const double fraction = static_cast<double>(bits & 0x3FFu);

    if (exponent == 0x1Fu)
        return fraction == 0.0 ? sign * std::numeric_limits<double>::infinity() :
                                 std::numeric_limits<double>::quiet_NaN();
    if (exponent == 0u)
        return sign * std::ldexp(fraction, -24);
    return sign * std::ldexp(1024.0 + fraction, static_cast<int>(exponent) - 25);
}

} // namespace

FixedRounding roundToFixed(double value, std::int32_t &raw) noexcept
{
    if (!std::isfinite(value))
        return FixedRounding::NotFinite;
    const double rounded = roundedHalfAway(value * kFixedUnits);
    if (std::fabs(rounded) > kLargestRaw)
        return FixedRounding::PastRange;
    raw = static_cast<std::int32_t>(rounded);
    return FixedRounding::Rounded;
}

BlockTaking takeEightBitBlock(const std::uint8_t *block, infer::QuantBlock &out) noexcept
{
    const double step = halfAt(block);
    if (!std::isfinite(step) || std::signbit(step))
        return BlockTaking::ScaleRefused;
    const double magnitude = roundedHalfAway(kQuantLevels * step * kFixedUnits);
    if (magnitude > kLargestRaw)
        return BlockTaking::PastRange;

    infer::QuantBlock taken{};
    taken.scaleRaw = static_cast<std::int32_t>(magnitude);
    for (std::uint32_t i = 0u; i < infer::kQuantBlockLanes; ++i)
    {
        const std::int8_t lane = static_cast<std::int8_t>(block[kEightBitStepBytes + i]);
        if (lane == std::numeric_limits<std::int8_t>::min())
            return BlockTaking::LanePastRange;
        taken.lanes[i] = lane;
    }
    out = taken;
    return BlockTaking::Taken;
}

double laneValue(const infer::QuantBlock &block, std::uint32_t lane) noexcept
{
    return static_cast<double>(block.lanes[lane]) * static_cast<double>(block.scaleRaw) / kQuantLevels / kFixedUnits;
}

double sourceValue(const SourceTensor &tensor, std::uint64_t index) noexcept
{
    switch (tensor.type)
    {
    case TensorType::Float32: {
        float value = 0.0f;
        std::memcpy(&value, tensor.bytes + index * sizeof(float), sizeof(float));
        return static_cast<double>(value);
    }
    case TensorType::Float16: return halfAt(tensor.bytes + index * 2u);
    case TensorType::BrainFloat16: {
        const std::uint32_t bits = (static_cast<std::uint32_t>(tensor.bytes[index * 2u]) |
                                    (static_cast<std::uint32_t>(tensor.bytes[index * 2u + 1u]) << 8))
                                   << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return static_cast<double>(value);
    }
    case TensorType::QuantisedEightBit: {
        const std::uint8_t *const block = tensor.bytes + (index / infer::kQuantBlockLanes) * kEightBitBlockBytes;
        const std::int8_t lane = static_cast<std::int8_t>(block[kEightBitStepBytes + index % infer::kQuantBlockLanes]);
        return static_cast<double>(lane) * halfAt(block);
    }
    case TensorType::Other: break;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

void ErrorStatistics::add(double image, double source, std::uint64_t block) noexcept
{
    const double difference = image - source;
    const double magnitude = std::fabs(difference);

    squaredError += difference * difference;
    squaredSource += source * source;
    ++values;
    if (magnitude > largestError)
    {
        largestError = magnitude;
        worstBlock = block;
    }
}

double ErrorStatistics::relativeRootMeanSquare() const noexcept
{
    if (squaredError == 0.0)
        return 0.0;
    if (squaredSource == 0.0)
        return std::numeric_limits<double>::infinity();
    return std::sqrt(squaredError / squaredSource);
}

} // namespace lpl::convert
