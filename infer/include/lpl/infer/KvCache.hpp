/**
 * @file KvCache.hpp
 * @brief Bounded attention cache with an explicit eviction policy.
 *
 * The cache is the real memory cost of inference and the usual cause of an
 * unbounded one. Here it has a fixed extent and a stated policy, because the tick
 * deadline does not care why memory ran out.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_KVCACHE_HPP
#    define LPL_LPL_INFER_KVCACHE_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::infer {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::infer

#endif // LPL_LPL_INFER_KVCACHE_HPP
