#include <lpl/convert/Rotary.hpp>

#include <cmath>
#include <cstdio>

namespace lpl::convert {

namespace {

constexpr long double kTwoPi = 6.283185307179586476925286766559005768L; ///< A whole turn, in radians.
constexpr long double kTurnUnits = 4294967296.0L;                       ///< 2^32 units of a turn.

/**
 * @brief The refusal of a base or a head the table cannot be computed for.
 */
[[nodiscard]] Refusal outOfRange(double base, std::uint32_t dimensions)
{
    char detail[160];
    std::snprintf(detail, sizeof(detail),
                  "rope.freq_base %.9g over %u channels: the base must be finite and past 1, the channels even and "
                  "not 0",
                  base, dimensions);
    return Refusal{"metadata_out_of_range", detail};
}

/**
 * @brief The refusal of pair @p pair, whose word @p exact lies @p distance of a unit from a tie.
 */
[[nodiscard]] Refusal nearTie(double base, std::uint32_t pair, long double exact, long double distance)
{
    char detail[160];
    std::snprintf(detail, sizeof(detail), "pair %u of base %.9g turns %.9Lf units per position, %.3Le from a tie", pair,
                  base, exact, distance);
    return Refusal{"rotary_near_tie", detail};
}

} // namespace

std::optional<Refusal> rotaryTurnsPerPosition(double base, std::uint32_t dimensions, std::vector<std::uint32_t> &out)
{
    out.clear();
    if (!std::isfinite(base) || !(base > 1.0) || dimensions == 0u || dimensions % 2u != 0u)
        return outOfRange(base, dimensions);

    const long double logBase = std::log(static_cast<long double>(base));
    for (std::uint32_t pair = 0u; pair < dimensions / 2u; ++pair)
    {
        const long double exponent = -2.0L * static_cast<long double>(pair) / static_cast<long double>(dimensions);
        const long double exact = std::exp(exponent * logBase) / kTwoPi * kTurnUnits;
        const long double distance = std::fabs(exact - std::floor(exact) - 0.5L);
        if (distance < static_cast<long double>(kRotaryLeastDistanceToTie))
            return nearTie(base, pair, exact, distance);
        out.push_back(static_cast<std::uint32_t>(std::floor(exact + 0.5L)));
    }
    return std::nullopt;
}

} // namespace lpl::convert
