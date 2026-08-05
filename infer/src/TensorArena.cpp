/**
 * @file TensorArena.cpp
 * @brief The one big preallocated region weights and activations live in.
 *
 * Header-only by design — the arena is a typed façade over
 * @c lpl::memory::ArenaAllocator and adds no out-of-line state. This translation
 * unit exists so the module has an object file for the header even on a build that
 * inlines everything, and so a later addition has an obvious home.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/TensorArena.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::infer {

static_assert(sizeof(TensorArena) == sizeof(memory::ArenaAllocator),
              "TensorArena must stay a façade: state of its own would be a second arena");

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
