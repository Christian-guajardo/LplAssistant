#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace laplace {

// Synthèse vocale via espeak-ng (moteur externe, comme laplace-stt pour le
// STT). Retourne du PCM16 mono 16 kHz, prêt à streamer vers un satellite.
// Vide si la synthèse échoue.
std::vector<int16_t> tts_synthesize_16k(const std::string& text);

} // namespace laplace
