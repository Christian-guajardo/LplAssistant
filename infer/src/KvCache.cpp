/**
 * @file KvCache.cpp
 * @brief Bounded attention cache with an explicit eviction policy.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/KvCache.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::infer {

bool KvCache::allocate(TensorArena &arena, const ModelConfig &shape, CachePolicy policy)
{
    if (!shape.valid())
        return false;

    const core::usize entries = static_cast<core::usize>(shape.layers) * shape.contextLength * shape.dim;
    math::Fixed32 *const keys = arena.claim<math::Fixed32>(entries);
    math::Fixed32 *const values = arena.claim<math::Fixed32>(entries);
    if (keys == nullptr || values == nullptr)
        return false;

    _keys = keys;
    _values = values;
    _dim = shape.dim;
    _layers = shape.layers;
    _capacity = shape.contextLength;
    _length = 0u;
    _evictions = 0u;
    _policy = policy;
    return true;
}

core::u32 KvCache::reserve() noexcept
{
    if (_keys == nullptr)
        return kNoSlot;

    if (_length < _capacity)
        return _length++;

    if (_policy == CachePolicy::Refuse)
        return kNoSlot;

    // Shift every layer down by one position. Done layer by layer over the whole
    // window rather than by moving a head pointer, for the reason the policy's
    // documentation gives: a slot's index has to keep meaning its position.
    for (core::u32 l = 0u; l < _layers; ++l)
    {
        math::Fixed32 *const keyBase = _keys + static_cast<core::usize>(l) * _capacity * _dim;
        math::Fixed32 *const valueBase = _values + static_cast<core::usize>(l) * _capacity * _dim;
        for (core::u32 slot = 1u; slot < _capacity; ++slot)
            for (core::u32 i = 0u; i < _dim; ++i)
            {
                keyBase[(slot - 1u) * _dim + i] = keyBase[slot * _dim + i];
                valueBase[(slot - 1u) * _dim + i] = valueBase[slot * _dim + i];
            }
    }

    ++_evictions;
    return _capacity - 1u;
}

VectorView KvCache::key(core::u32 layer, core::u32 slot) const noexcept
{
    LPL_VERIFY(_keys != nullptr && layer < _layers && slot < _capacity);
    return VectorView{_keys + (static_cast<core::usize>(layer) * _capacity + slot) * _dim, _dim};
}

VectorView KvCache::value(core::u32 layer, core::u32 slot) const noexcept
{
    LPL_VERIFY(_values != nullptr && layer < _layers && slot < _capacity);
    return VectorView{_values + (static_cast<core::usize>(layer) * _capacity + slot) * _dim, _dim};
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
