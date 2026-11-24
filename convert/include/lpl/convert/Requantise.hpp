/**
 * @file Requantise.hpp
 * @brief A GGUF tensor's values turned into ring 0's 8-bit blocks, and the error that costs.
 *
 * A float weight is rounded to Q16.16, half away from zero, then quantised by lpl::infer::quantiseRow, the quantiser
 * ring 0 runs on its activations: a block of 32 keeps its largest magnitude M and 32 lanes, a lane standing for
 * lane M / 127. Every F32, F16 or BF16 value times 2^16 is exact in a double, so the rounding depends on no rounding
 * mode. A Q8_0 block keeps its 32 lanes and turns its step d into M = round(127 d 2^16): llama.cpp writes its lanes as
 * round(x / d) with d = max|x| / 127 (ggml-quants.c:848-871), so a lane of ±127 is the block's largest.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_CONVERT_REQUANTISE_HPP
#    define LPL_LPL_CONVERT_REQUANTISE_HPP

#    include <lpl/convert/Source.hpp>
#    include <lpl/infer/Quant.hpp>

#    include <cstdint>

namespace lpl::convert {

/** Bytes of a Q8_0 block: its 16-bit step, then 32 lanes. */
inline constexpr std::uint32_t kEightBitBlockBytes = 34u;

/**
 * @enum FixedRounding
 * @brief What rounding a value to Q16.16 found.
 */
enum class FixedRounding : std::uint8_t {
    Rounded,   ///< The raw word holds the value, rounded half away from zero.
    NotFinite, ///< The value is NaN or infinite.
    PastRange, ///< Rounded, it leaves [-(2^31 - 1), 2^31 - 1]: quantiseRow takes a raw word's absolute value.
};

/**
 * @brief Rounds @p value to a Q16.16 raw word, half away from zero.
 * @param value A weight.
 * @param raw   Receives the raw word when the rounding holds.
 * @return Rounded, or why not.
 */
[[nodiscard]] FixedRounding roundToFixed(double value, std::int32_t &raw) noexcept;

/**
 * @enum BlockTaking
 * @brief What taking a Q8_0 block found.
 */
enum class BlockTaking : std::uint8_t {
    Taken,         ///< The block's lanes are kept and its step is a magnitude.
    ScaleRefused,  ///< The step is negative, NaN or infinite.
    PastRange,     ///< The magnitude 127 d 2^16 rounds past 2^31 - 1.
    LanePastRange, ///< A lane is -128, which no symmetric block of ring 0 holds.
};

/**
 * @brief Takes the Q8_0 block at @p block: its lanes as they are, its step d as the magnitude round(127 d 2^16).
 * @param block 34 bytes: d as a little-endian IEEE half, then 32 signed lanes.
 * @param out   Receives the block when it is taken.
 * @return Taken, or why not.
 */
[[nodiscard]] BlockTaking takeEightBitBlock(const std::uint8_t *block, infer::QuantBlock &out) noexcept;

/**
 * @brief The value lane @p lane of @p block stands for in a matrix product: lane M / 127, unrounded.
 * @param block A block.
 * @param lane  Below lpl::infer::kQuantBlockLanes.
 * @return The value, in units of 1.
 */
[[nodiscard]] double laneValue(const infer::QuantBlock &block, std::uint32_t lane) noexcept;

/**
 * @brief The value at @p index of @p tensor, as the file stores it.
 * @param tensor A tensor of type F32, F16, BF16 or Q8_0, whose bytes hold @p index.
 * @param index  Fastest extent first.
 * @return The value; for Q8_0, lane d.
 */
[[nodiscard]] double sourceValue(const SourceTensor &tensor, std::uint64_t index) noexcept;

/**
 * @struct ErrorStatistics
 * @brief How far a tensor's image lies from its source, value by value.
 */
struct ErrorStatistics {
    double squaredError{0.0};     ///< Sum of the squared differences.
    double squaredSource{0.0};    ///< Sum of the squared source values.
    double largestError{0.0};     ///< The largest difference's magnitude.
    std::uint64_t worstBlock{0u}; ///< The block, counted from the tensor's first, that holds it.
    std::uint64_t values{0u};     ///< Values compared.

    /**
     * @brief Adds one value's comparison.
     * @param image  The value the image holds.
     * @param source The value the source holds.
     * @param block  The block, counted from the tensor's first, that holds it.
     */
    void add(double image, double source, std::uint64_t block) noexcept;

    /**
     * @brief The root of the squared error over the squared source.
     * @return The relative root mean square; 0 for no error; infinite for an error over a source of zeros.
     */
    [[nodiscard]] double relativeRootMeanSquare() const noexcept;
};

} // namespace lpl::convert

#endif /* LPL_LPL_CONVERT_REQUANTISE_HPP */
