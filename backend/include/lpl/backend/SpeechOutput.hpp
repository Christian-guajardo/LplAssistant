/**
 * @file SpeechOutput.hpp
 * @brief Text to speech, with a graceful fallback.
 *
 * Prefers a natural neural voice and falls back to a robotic formant synthesiser
 * when it is absent, because an assistant that cannot speak at all is worse than
 * one that speaks plainly. Markdown is stripped before synthesis: the voice should
 * read the content, not the formatting.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_BACKEND_SPEECHOUTPUT_HPP
#    define LPL_BACKEND_SPEECHOUTPUT_HPP

#    include <cstdint>
#    include <string>
#    include <vector>

namespace lpl::backend {

// Synthèse vocale via espeak-ng (moteur externe, comme laplace-stt pour le
// STT). Retourne du PCM16 mono 16 kHz, prêt à streamer vers un satellite.
// Vide si la synthèse échoue.
std::vector<int16_t> synthesizeSpeech(const std::string& text);

} // namespace lpl::backend

#endif // LPL_BACKEND_SPEECHOUTPUT_HPP
