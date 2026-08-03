/**
 * @file Protocol.hpp
 * @brief The satellite wire format, in one place.
 *
 * Raw PCM16 mono at 16 kHz, an `END` datagram to close an utterance, and the
 * return channel interleaving `TXT:` segments, audio frames, then `AEND` — or
 * `STOP` when the sovereign cut in. Written once and shared, because a hosted
 * satellite, a bare-metal one and a microcontroller must not each hold their own
 * opinion about what closes an utterance.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_PROTOCOL_HPP
#    define LPL_SATELLITE_PROTOCOL_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::satellite {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::satellite

#endif // LPL_SATELLITE_PROTOCOL_HPP
