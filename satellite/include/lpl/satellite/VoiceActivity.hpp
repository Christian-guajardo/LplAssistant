/**
 * @file VoiceActivity.hpp
 * @brief Deciding that someone is speaking, on energy alone.
 *
 * Hysteresis plus a hangover window: two thresholds so a pause inside a sentence
 * does not end the utterance, and a tail so the last syllable survives. No model,
 * no allocation — this has to run on a microcontroller with a few hundred kilobytes
 * and it has to run all the time.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_VOICEACTIVITY_HPP
#    define LPL_SATELLITE_VOICEACTIVITY_HPP

#    include <lpl/Foundation.hpp>

namespace lpl::satellite {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::satellite

#endif // LPL_SATELLITE_VOICEACTIVITY_HPP
