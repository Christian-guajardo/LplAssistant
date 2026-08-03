/**
 * @file Duplex.hpp
 * @brief Listening while speaking, and not answering oneself.
 *
 * A satellite that stops listening while it talks cannot be interrupted, which is
 * the one thing a voice assistant most needs to allow. So it listens through its own
 * playback — and then has to recognise its own voice coming back. Not by DSP echo
 * cancellation but by CONTENT: an utterance arriving inside the playback window whose
 * words are mostly the words just spoken is an echo. Cheap, and it needs no second
 * microphone.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_DUPLEX_HPP
#    define LPL_SATELLITE_DUPLEX_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::satellite {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::satellite

#endif // LPL_SATELLITE_DUPLEX_HPP
