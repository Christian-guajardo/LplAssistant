#pragma once
#include <memory>
#include <string>

namespace laplace {

// Transcription vocale via whisper.cpp. Entrée : WAV PCM 16 bits mono 16 kHz
// (le format que les satellites ESP32 enverront en UDP à terme).
class Stt {
public:
    Stt(const std::string& model_path, int n_threads);
    ~Stt();

    std::string transcribe_wav(const std::string& wav_path,
                               const std::string& language = "fr");

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace laplace
