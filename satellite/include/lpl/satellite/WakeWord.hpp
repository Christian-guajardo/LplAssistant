/**
 * @file WakeWord.hpp
 * @brief The gate that keeps the network quiet.
 *
 * A satellite that streamed continuously would saturate the link and the server for
 * nothing. Until the word is heard, nothing leaves the node. On a microcontroller
 * this is a few-kilobyte model; on a desktop it is the same interface with a larger
 * one behind it — and the same interface is what makes them swappable.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_WAKEWORD_HPP
#    define LPL_SATELLITE_WAKEWORD_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::satellite {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::satellite

#endif // LPL_SATELLITE_WAKEWORD_HPP
