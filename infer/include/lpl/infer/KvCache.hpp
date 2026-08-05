/**
 * @file KvCache.hpp
 * @brief Bounded attention cache with an explicit eviction policy.
 *
 * The cache is the real memory cost of inference and the usual cause of an
 * unbounded one. Here it has a fixed extent and a stated policy, because the tick
 * deadline does not care why memory ran out.
 *
 * Keys and values are kept UNQUANTISED. Everything else in this module is eight
 * bits, and the difference is not an oversight: weights are quantised once and read
 * a million times, whereas a cache entry is written once and read a few dozen times,
 * so the same trade buys almost nothing and costs the attention scores their
 * precision at exactly the point where a softmax is about to exaggerate the loss.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_KVCACHE_HPP
#    define LPL_LPL_INFER_KVCACHE_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/Model.hpp>

namespace lpl::infer {

/// Returned by @ref KvCache::reserve when the window is full and the policy refuses.
inline constexpr core::u32 kNoSlot = 0xFFFFFFFFu;

/**
 * @enum CachePolicy
 * @brief What happens when the window is full.
 */
enum class CachePolicy : core::u32 {
    /**
     * Refuse the position and let the caller decide.
     *
     * The bounded contract in its strictest form, and the one a parity gate runs
     * under: a generation that would exceed the window stops, visibly, rather than
     * quietly changing what the model can see.
     */
    Refuse = 0u,

    /**
     * Drop the oldest position and shift the rest down.
     *
     * Costs a copy of the whole window per token past it, which is accepted rather
     * than hidden behind a ring buffer. A ring would make a slot's index stop
     * meaning its position, and the rotary embedding is a function of position — so
     * the cheap structure would silently rotate every cached key by the wrong angle.
     */
    SlideOldest = 1u,
};

/**
 * @class KvCache
 * @brief Per-layer keys and values for the positions currently in the window.
 */
class KvCache {
public:
    KvCache() = default;

    /**
     * @brief Claims the whole window up front.
     * @param arena  Storage.
     * @param shape  The model's extents.
     * @param policy What to do when full.
     * @return false when the arena is exhausted.
     */
    bool allocate(TensorArena &arena, const ModelConfig &shape, CachePolicy policy);

    /**
     * @brief Forgets every position without releasing storage.
     */
    void clear() noexcept { _length = 0u; }

    /**
     * @brief Positions currently held.
     * @return The window occupancy.
     */
    [[nodiscard]] core::u32 length() const noexcept { return _length; }

    /**
     * @brief Positions the window can hold.
     * @return The capacity.
     */
    [[nodiscard]] core::u32 capacity() const noexcept { return _capacity; }

    /**
     * @brief Times the window slid, discarding its oldest position.
     * @return The eviction count since the last @ref allocate.
     */
    [[nodiscard]] core::u32 evictions() const noexcept { return _evictions; }

    /**
     * @brief Makes room for one more position.
     * @return The slot to write, or @ref kNoSlot under @ref CachePolicy::Refuse.
     */
    [[nodiscard]] core::u32 reserve() noexcept;

    /**
     * @brief The key run of one slot.
     * @param layer Block index.
     * @param slot  Slot index below @ref length.
     * @return The run, of @c dim values.
     */
    [[nodiscard]] VectorView key(core::u32 layer, core::u32 slot) const noexcept;

    /**
     * @brief The value run of one slot.
     * @param layer Block index.
     * @param slot  Slot index below @ref length.
     * @return The run, of @c dim values.
     */
    [[nodiscard]] VectorView value(core::u32 layer, core::u32 slot) const noexcept;

private:
    math::Fixed32 *_keys{nullptr};
    math::Fixed32 *_values{nullptr};
    core::u32 _dim{0u};
    core::u32 _layers{0u};
    core::u32 _capacity{0u};
    core::u32 _length{0u};
    core::u32 _evictions{0u};
    CachePolicy _policy{CachePolicy::Refuse};
};

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_KVCACHE_HPP
