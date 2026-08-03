/**
 * @file Parity.hpp
 * @brief The constexpr exchange all three implementations must agree on.
 *
 * A fixed sequence of datagrams must produce the same decisions on the host oracle
 * and in ring 0. Three consumers of one protocol is exactly the situation where a
 * gate stops being bureaucracy.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_PARITY_HPP
#    define LPL_SATELLITE_PARITY_HPP

#    include <lpl/Foundation.hpp>

namespace lpl::satellite {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::satellite

#endif // LPL_SATELLITE_PARITY_HPP
