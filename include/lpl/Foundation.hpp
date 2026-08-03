/**
 * @file Foundation.hpp
 * @brief The one place that knows whether the LplPlugin foundation is present.
 *
 * Every repository in the project must build, run and be TESTED on its own. LplPlugin
 * builds without LplKernel; this repository builds without either. The foundation
 * modules (core, math) are therefore an *enhancement*, detected at configure time,
 * never a hard requirement — the same arrangement LplKernel already uses for the
 * engine itself, where a missing submodule sets LPL_PLUGIN_UNAVAILABLE and the kernel
 * still boots.
 *
 * Two builds, and the difference between them is deliberate:
 *
 *   - **with the foundation** (`LPL_HAS_FOUNDATION`): the real `lpl::core` types and
 *     assertions, and `lpl::math::Fixed32`/CORDIC are reachable. This is the build
 *     that can honour the determinism contract and be compiled -ffreestanding into
 *     the kernel.
 *   - **standalone**: primitive ALIASES only, declared here. They are aliases of the
 *     very same standard types `lpl::core` uses, so the two cannot disagree — but
 *     Fixed32 and CORDIC are NOT emulated, and no substitute is offered.
 *
 * That last point is the honest part. A fake fixed-point type would let a standalone
 * build claim a parity it cannot possibly have, and the first person to trust that
 * claim would be debugging a divergence that no test could reproduce. So the
 * standalone build is host-only by construction: targets that need the determinism
 * contract are SKIPPED when the foundation is absent, not stubbed.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_FOUNDATION_HPP
#    define LPL_FOUNDATION_HPP

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/core/Assert.hpp>
#        include <lpl/core/Types.hpp>

#    else // standalone build — no LplPlugin on the include path

#        include <cstddef>
#        include <cstdint>
#        include <cstdio>
#        include <cstdlib>

/// Aliases, never reimplementations. `lpl::core` defines these as aliases of exactly
/// the same standard types, so a translation unit compiled either way agrees with a
/// translation unit compiled the other way.
namespace lpl::core {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using usize = std::size_t;
using isize = std::ptrdiff_t;
using f32 = float;
using f64 = double;

} // namespace lpl::core

/// Same contract as the foundation's macro: fail loudly at the call site. A stub that
/// returned a plausible default would look like a working feature in every test that
/// touched it.
#        define LPL_NOT_IMPLEMENTED(what)                                                                              \
            do                                                                                                         \
            {                                                                                                          \
                std::fprintf(stderr, "[LPL] NOT IMPLEMENTED: %s (%s:%d)\n", what, __FILE__, __LINE__);                 \
                std::abort();                                                                                          \
            } while (false)

#        define LPL_VERIFY(cond)                                                                                       \
            do                                                                                                         \
            {                                                                                                          \
                if (!(cond))                                                                                           \
                {                                                                                                      \
                    std::fprintf(stderr, "[LPL] VERIFY failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__);              \
                    std::abort();                                                                                      \
                }                                                                                                      \
            } while (false)

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_FOUNDATION_HPP
