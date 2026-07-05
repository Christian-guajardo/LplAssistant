// laplace-mic : satellite audio de test (fallback local WSL).
// Capte le micro par défaut (PulseAudio/WSLg = micro Windows via RDPSource),
// détecte la parole par énergie, et streame le PCM16 mono 16 kHz en UDP vers
// le serveur `LplAssistant --listen`. Reproduit le comportement cible d'un
// ESP32 : petits paquets à cadence temps réel + "END" en fin d'énoncé.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pulse/error.h>
#include <pulse/simple.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>

namespace {

constexpr int    RATE          = 16000;
constexpr int    FRAME_MS      = 40;                  // paquet = 40 ms
constexpr int    FRAME_SAMPLES = RATE * FRAME_MS / 1000;
constexpr float  START_RMS     = 0.020f;              // seuil début de parole
constexpr float  KEEP_RMS      = 0.012f;              // seuil maintien (hystérésis)
constexpr int    HANGOVER_MS   = 700;                 // silence toléré avant "END"
constexpr int    PREROLL       = 4;                   // paquets gardés avant l'attaque

std::atomic<bool> g_run{true};
void on_sigint(int) { g_run = false; }

float frame_rms(const int16_t* s, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) { float v = s[i] / 32768.0f; acc += v * v; }
    return (float)std::sqrt(acc / n);
}

// Réception de la réponse : "TXT:<texte>" (affiché), paquets PCM16 16 kHz
// (accumulés), "AEND" (fin). Puis lecture sur le haut-parleur par défaut.
// Le mode "parle" et le mode "écoute" alternent dans la même boucle : la
// connexion UDP reste la même, aucun re-connect.
void receive_and_play_reply(int sock, const pa_sample_spec& ss) {
    std::vector<int16_t> voice;
    char buf[65536];
    bool got_any = false;
    while (g_run) {
        ssize_t n = recvfrom(sock, buf, sizeof(buf) - 1, 0, nullptr, nullptr);
        if (n < 0) {
            if (!got_any)
                std::fprintf(stderr, "[laplace-mic] pas de réponse (timeout)\n");
            return;
        }
        got_any = true;
        if (n == 3 && !std::memcmp(buf, "NOP", 3)) return; // ignoré (pas de mot d'appel)
        if (n == 4 && !std::memcmp(buf, "AEND", 4)) break;
        if (n > 4 && !std::memcmp(buf, "TXT:", 4)) {
            buf[n] = '\0';
            std::printf("Laplace: %s\n", buf + 4);
            std::fflush(stdout);
            continue;
        }
        voice.insert(voice.end(), (const int16_t*)buf,
                     (const int16_t*)buf + n / 2);
    }
    if (voice.empty()) return;

    int err = 0;
    pa_simple* out = pa_simple_new(nullptr, "laplace-mic", PA_STREAM_PLAYBACK,
                                   nullptr, "réponse", &ss, nullptr, nullptr,
                                   &err);
    if (!out) {
        std::fprintf(stderr, "[laplace-mic] lecture impossible: %s\n",
                     pa_strerror(err));
        return;
    }
    pa_simple_write(out, voice.data(), voice.size() * 2, &err);
    pa_simple_drain(out, &err);
    pa_simple_free(out);
}

} // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 7777;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!std::strcmp(argv[i], "--port") && i + 1 < argc) port = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr,
                "laplace-mic : satellite micro -> UDP\n"
                "Usage: %s [--host <ip>] [--port <p>]  (défaut 127.0.0.1:7777)\n",
                argv[0]);
            return 1;
        }
    }

    // Socket UDP vers le serveur.
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { std::perror("socket"); return 1; }
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port   = htons((uint16_t)port);
    if (inet_pton(AF_INET, host.c_str(), &dst.sin_addr) != 1) {
        std::fprintf(stderr, "hôte invalide: %s\n", host.c_str());
        return 1;
    }
    int rcv_timeo_us = 0;
    timeval tv{}; tv.tv_sec = 30;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    (void)rcv_timeo_us;

    // Capture PulseAudio : s16le mono 16 kHz.
    pa_sample_spec ss{};
    ss.format = PA_SAMPLE_S16LE;
    ss.channels = 1;
    ss.rate = RATE;
    int err = 0;
    pa_simple* pa = pa_simple_new(nullptr, "laplace-mic", PA_STREAM_RECORD,
                                  nullptr, "capture", &ss, nullptr, nullptr, &err);
    if (!pa) {
        std::fprintf(stderr, "pa_simple_new a échoué: %s\n", pa_strerror(err));
        return 1;
    }

    std::signal(SIGINT, on_sigint);
    std::fprintf(stderr,
        "[laplace-mic] micro -> %s:%d — parlez (Ctrl-C pour quitter)\n",
        host.c_str(), port);

    std::vector<int16_t> frame(FRAME_SAMPLES);
    std::vector<std::vector<int16_t>> preroll; // paquets pré-attaque
    bool speaking = false;
    int  silence_ms = 0;

    while (g_run) {
        if (pa_simple_read(pa, frame.data(), FRAME_SAMPLES * 2, &err) < 0) {
            std::fprintf(stderr, "pa_simple_read: %s\n", pa_strerror(err));
            break;
        }
        float rms = frame_rms(frame.data(), FRAME_SAMPLES);

        if (!speaking) {
            preroll.push_back(frame);
            if ((int)preroll.size() > PREROLL) preroll.erase(preroll.begin());
            if (rms >= START_RMS) {
                speaking = true;
                silence_ms = 0;
                std::fprintf(stderr, "[laplace-mic] parole détectée...\n");
                for (auto& f : preroll)
                    sendto(sock, f.data(), FRAME_SAMPLES * 2, 0,
                           (sockaddr*)&dst, sizeof(dst));
                preroll.clear();
            }
            continue;
        }

        // En cours de parole : on streame chaque paquet.
        sendto(sock, frame.data(), FRAME_SAMPLES * 2, 0,
               (sockaddr*)&dst, sizeof(dst));
        silence_ms = (rms < KEEP_RMS) ? silence_ms + FRAME_MS : 0;
        if (silence_ms >= HANGOVER_MS) {
            sendto(sock, "END", 3, 0, (sockaddr*)&dst, sizeof(dst));
            speaking = false;
            std::fprintf(stderr, "[laplace-mic] fin d'énoncé, attente réponse...\n");
            receive_and_play_reply(sock, ss);
            // Purge ce que le micro a capté pendant que Laplace parlait
            // (sinon le satellite s'écoute lui-même : larsen de dialogue).
            pa_simple_flush(pa, &err);
        }
    }

    pa_simple_free(pa);
    close(sock);
    std::fprintf(stderr, "\n[laplace-mic] arrêt.\n");
    return 0;
}
