/**
 * @file TensorArena.hpp
 * @brief The one big preallocated region weights and activations live in.
 *
 * Sized at boot from the host profile, never grown. A server profile renders
 * nothing, and that is exactly the memory this arena claims.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_TENSORARENA_HPP
#    define LPL_LPL_INFER_TENSORARENA_HPP

#    include <lpl/Foundation.hpp>

namespace lpl::infer {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::infer

#endif // LPL_LPL_INFER_TENSORARENA_HPP
