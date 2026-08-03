/**
 * @file PowerState.hpp
 * @brief What a node does when nobody is talking to it.
 *
 * Most of a satellite's life is silence. The wake-word gate is the only thing that
 * must stay awake, so everything else may be clock-gated, and the processor may sleep
 * between audio buffers rather than spinning on a periodic tick. This is the module's
 * declaration of what may be turned off and what may not.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_POWERSTATE_HPP
#    define LPL_SATELLITE_POWERSTATE_HPP

#    include <lpl/Foundation.hpp>

namespace lpl::satellite {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::satellite

#endif // LPL_SATELLITE_POWERSTATE_HPP
