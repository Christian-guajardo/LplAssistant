#include <lpl/convert/Pairing.hpp>

namespace lpl::convert {

std::uint32_t sourceRowOf(std::uint32_t row, std::uint32_t headDim, RotaryPairing pairing) noexcept
{
    if (pairing == RotaryPairing::Adjacent || headDim < 2u)
        return row;

    const std::uint32_t head = row / headDim;
    const std::uint32_t channel = row % headDim;
    const std::uint32_t pair = channel / 2u;
    const std::uint32_t half = channel % 2u;
    return head * headDim + half * (headDim / 2u) + pair;
}

} // namespace lpl::convert
