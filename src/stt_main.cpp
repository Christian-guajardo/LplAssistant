// Binaire STT isolé : whisper.cpp embarque son propre ggml qui entre en
// conflit de symboles avec celui de llama.cpp en cas de link commun.
// Le processus séparé garantit l'isolation (et préfigure le micro-service
// STT de l'architecture cible).
#include "stt.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <fichier.wav> [langue]\n", argv[0]);
        return 1;
    }
    const char* model = std::getenv("LAPLACE_STT_MODEL");
    std::string model_path = model ? model : "models/ggml-base.bin";
    const char* env_threads = std::getenv("LAPLACE_N_THREADS");
    int n_threads = env_threads ? std::atoi(env_threads) : 8;

    try {
        laplace::Stt stt(model_path, n_threads);
        std::string text = stt.transcribe_wav(argv[1], argc > 2 ? argv[2] : "fr");
        std::printf("%s\n", text.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[laplace-stt] erreur: %s\n", e.what());
        return 1;
    }
    return 0;
}
