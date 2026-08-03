/**
 * @file SpeechInput.hpp
 * @brief Speech to text, in its own address space.
 *
 * Lives behind a separate binary, and that is not architectural taste: the tensor
 * library bundled with the transcription engine is link-incompatible with the one
 * bundled with the language model, and mixing them segfaults. Two processes is the
 * cheap, honest fix — and it happens to match what a satellite will do anyway.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_BACKEND_SPEECHINPUT_HPP
#    define LPL_BACKEND_SPEECHINPUT_HPP

#    include <memory>
#    include <string>

namespace lpl::backend {

// Transcription vocale via whisper.cpp. Entrée : WAV PCM 16 bits mono 16 kHz
// (le format que les satellites ESP32 enverront en UDP à terme).
class SpeechInput {
public:
    SpeechInput(const std::string& modelPath, int threadCount);
    ~SpeechInput();

    std::string transcribeWave(const std::string& wavePath,
                               const std::string& language = "fr");

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace lpl::backend

#endif // LPL_BACKEND_SPEECHINPUT_HPP
