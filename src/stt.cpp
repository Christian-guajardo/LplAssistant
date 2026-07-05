#include "stt.h"
#include <whisper.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace laplace {

struct Stt::Impl {
    whisper_context* ctx = nullptr;
    int n_threads;
    ~Impl() { if (ctx) whisper_free(ctx); }
};

Stt::Stt(const std::string& model_path, int n_threads) : impl(new Impl()) {
    whisper_context_params cp = whisper_context_default_params();
    impl->ctx = whisper_init_from_file_with_params(model_path.c_str(), cp);
    if (!impl->ctx)
        throw std::runtime_error("Stt: impossible de charger " + model_path);
    impl->n_threads = n_threads;
}

Stt::~Stt() = default;

// Lecteur WAV minimal : PCM 16 bits, downmix mono + rééchantillonnage 16 kHz.
static std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Stt: fichier introuvable: " + path);

    char riff[4], wave[4];
    uint32_t size;
    f.read(riff, 4); f.read((char*)&size, 4); f.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) || std::memcmp(wave, "WAVE", 4))
        throw std::runtime_error("Stt: pas un fichier WAV: " + path);

    uint16_t channels = 0, bits = 0;
    uint32_t rate = 0;
    std::vector<int16_t> pcm;
    char id[4];
    uint32_t chunk_size;
    while (f.read(id, 4) && f.read((char*)&chunk_size, 4)) {
        if (!std::memcmp(id, "fmt ", 4)) {
            uint16_t fmt;
            f.read((char*)&fmt, 2); f.read((char*)&channels, 2);
            f.read((char*)&rate, 4); f.seekg(6, std::ios::cur);
            f.read((char*)&bits, 2);
            f.seekg(chunk_size - 16, std::ios::cur);
            if (fmt != 1 || bits != 16)
                throw std::runtime_error("Stt: seul le PCM 16 bits est géré");
        } else if (!std::memcmp(id, "data", 4)) {
            pcm.resize(chunk_size / 2);
            f.read((char*)pcm.data(), chunk_size);
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
    }
    if (channels == 0 || rate == 0 || pcm.empty())
        throw std::runtime_error("Stt: WAV invalide ou vide: " + path);

    // Downmix vers mono (moyenne des canaux).
    std::vector<float> mono;
    mono.reserve(pcm.size() / channels);
    for (size_t i = 0; i + channels <= pcm.size(); i += channels) {
        int32_t acc = 0;
        for (int c = 0; c < channels; ++c) acc += pcm[i + c];
        mono.push_back((float)acc / (channels * 32768.0f));
    }
    if (rate == 16000) return mono;

    // Rééchantillonnage linéaire vers 16 kHz (suffisant pour du STT).
    double ratio = (double)rate / 16000.0;
    std::vector<float> samples;
    samples.reserve((size_t)(mono.size() / ratio) + 1);
    for (double pos = 0.0; pos < (double)(mono.size() - 1); pos += ratio) {
        size_t i = (size_t)pos;
        float  frac = (float)(pos - (double)i);
        samples.push_back(mono[i] * (1.0f - frac) + mono[i + 1] * frac);
    }
    return samples;
}

std::string Stt::transcribe_wav(const std::string& wav_path,
                                const std::string& language) {
    std::vector<float> samples = load_wav_16k_mono(wav_path);

    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.n_threads       = impl->n_threads;
    p.language        = language.c_str();
    p.print_progress  = false;
    p.print_realtime  = false;
    p.print_special   = false;

    if (whisper_full(impl->ctx, p, samples.data(), (int)samples.size()) != 0)
        throw std::runtime_error("Stt: échec de la transcription");

    std::string text;
    int n = whisper_full_n_segments(impl->ctx);
    for (int i = 0; i < n; ++i)
        text += whisper_full_get_segment_text(impl->ctx, i);
    // Trim des espaces de tête ajoutés par whisper
    size_t start = text.find_first_not_of(' ');
    return start == std::string::npos ? "" : text.substr(start);
}

} // namespace laplace
