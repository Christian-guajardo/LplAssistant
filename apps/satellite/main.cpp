/**
 * @file main.cpp
 * @brief The development node: a microphone and a speaker in one room.
 *
 *  
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

// laplace-mic : satellite audio de test (fallback local WSL).
// Capte le micro par défaut (PulseAudio/WSLg = micro Windows via RDPSource),
// détecte la parole par énergie, et streame le PCM16 mono 16 kHz en UDP vers
// le serveur `LplAssistant --listen`. Reproduit le comportement cible d'un
// ESP32 : petits paquets à cadence temps réel + "END" en fin d'énoncé.
//
// Full-duplex : l'écoute et la parole sont actives EN MÊME TEMPS. Un thread
// reçoit la réponse (texte + voix + AEND / NOP / STOP), un thread joue la
// voix, la boucle micro ne s'arrête jamais. On peut donc parler à Laplace
// pendant qu'il parle ("Laplace, stop" l'interrompt : le serveur renvoie
// STOP et la lecture est coupée net). Côté serveur, l'empreinte de la voix
// TTS évite que Laplace ne s'écoute lui-même via le micro.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pulse/error.h>
#include <pulse/simple.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int    RATE          = 16000;
constexpr int    FRAME_MS      = 40;                  // paquet = 40 ms
constexpr int    FRAME_SAMPLES = RATE * FRAME_MS / 1000;
constexpr float  START_RMS     = 0.020f;              // seuil début de parole
constexpr float  KEEP_RMS      = 0.012f;              // seuil maintien (hystérésis)
constexpr float  PLAY_BOOST    = 2.5f;                // seuils relevés pendant la lecture
constexpr int    HANGOVER_MS   = 700;                 // silence toléré avant "END"
constexpr int    PREROLL       = 4;                   // paquets gardés avant l'attaque

std::atomic<bool> g_run{true};
void on_sigint(int) { g_run = false; }

// La réponse arrive maintenant en LIVE, phrase par phrase (plusieurs "TXT:").
// On préfixe "Laplace:" une seule fois par réponse, puis on colle les segments.
std::atomic<bool> g_need_prefix{true};

// File de lecture partagée : le thread réseau remplit, le thread audio joue.
std::mutex                       g_play_mtx;
std::condition_variable          g_play_cv;
std::deque<std::vector<int16_t>> g_play_q;
std::atomic<bool>                g_playing{false};   // du son sort du HP
std::atomic<bool>                g_stop_play{false}; // interruption demandée

float frame_rms(const int16_t* s, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) { float v = s[i] / 32768.0f; acc += v * v; }
    return (float)std::sqrt(acc / n);
}

// Thread lecture : joue la file paquet par paquet (40 ms), pour pouvoir
// couper immédiatement sur STOP même si toute la réponse est déjà arrivée.
void playback_thread(const pa_sample_spec ss) {
    int err = 0;
    pa_simple* out = pa_simple_new(nullptr, "laplace-mic", PA_STREAM_PLAYBACK,
                                   nullptr, "réponse", &ss, nullptr, nullptr,
                                   &err);
    if (!out) {
        std::fprintf(stderr, "[laplace-mic] lecture impossible: %s\n",
                     pa_strerror(err));
        return;
    }
    while (g_run) {
        std::vector<int16_t> chunk;
        {
            std::unique_lock<std::mutex> lk(g_play_mtx);
            g_play_cv.wait_for(lk, std::chrono::milliseconds(200), [] {
                return !g_play_q.empty() || g_stop_play || !g_run;
            });
            if (g_stop_play) {
                g_play_q.clear();
                g_stop_play = false;
                g_playing = false;
                lk.unlock();
                pa_simple_flush(out, &err); // coupe ce qui joue encore
                std::fprintf(stderr, "[laplace-mic] lecture interrompue.\n");
                continue;
            }
            if (g_play_q.empty()) { g_playing = false; continue; }
            chunk = std::move(g_play_q.front());
            g_play_q.pop_front();
            g_playing = true;
        }
        pa_simple_write(out, chunk.data(), chunk.size() * 2, &err);
    }
    pa_simple_free(out);
}

// Thread réseau : reçoit en continu — la boucle micro n'attend jamais.
void receive_thread(int sock) {
    char buf[65536];
    while (g_run) {
        ssize_t n = recvfrom(sock, buf, sizeof(buf) - 1, 0, nullptr, nullptr);
        if (n <= 0) continue; // timeout : on repart écouter
        if (n == 3 && !std::memcmp(buf, "NOP", 3)) continue; // pas de mot d'appel
        if (n == 4 && !std::memcmp(buf, "AEND", 4)) {          // fin de réponse
            std::printf("\n");
            std::fflush(stdout);
            g_need_prefix = true;
            continue;
        }
        if (n == 4 && !std::memcmp(buf, "STOP", 4)) {
            g_stop_play = true;
            g_play_cv.notify_one();
            g_need_prefix = true;
            continue;
        }
        if (n > 4 && !std::memcmp(buf, "TXT:", 4)) {
            buf[n] = '\0';
            // Segments successifs de la même réponse : préfixe une seule fois.
            if (g_need_prefix.exchange(false)) std::printf("\nLaplace: ");
            std::printf("%s", buf + 4);
            std::fflush(stdout);
            continue;
        }
        // Sinon : un paquet de voix -> file de lecture.
        {
            std::lock_guard<std::mutex> lk(g_play_mtx);
            g_play_q.emplace_back((const int16_t*)buf,
                                  (const int16_t*)buf + n / 2);
            g_playing = true;
        }
        g_play_cv.notify_one();
    }
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
                "laplace-mic : satellite micro <-> UDP (écoute + parole)\n"
                "Usage: %s [--host <ip>] [--port <p>]  (défaut 127.0.0.1:7777)\n",
                argv[0]);
            return 1;
        }
    }

    // Socket UDP vers le serveur (une seule, partagée émission/réception).
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { std::perror("socket"); return 1; }
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port   = htons((uint16_t)port);
    if (inet_pton(AF_INET, host.c_str(), &dst.sin_addr) != 1) {
        std::fprintf(stderr, "hôte invalide: %s\n", host.c_str());
        return 1;
    }
    timeval tv{}; tv.tv_sec = 1; // le thread réseau se réveille pour g_run
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

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
        "[laplace-mic] micro <-> %s:%d — parlez, même pendant la réponse "
        "(« Laplace, stop » l'interrompt ; Ctrl-C pour quitter)\n",
        host.c_str(), port);

    std::thread rx(receive_thread, sock);
    std::thread px(playback_thread, ss);

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
        // Pendant que Laplace parle, le HP excite le micro : on relève les
        // seuils pour ne déclencher que sur une voix proche (la nôtre). Le
        // serveur filtre en plus par empreinte ce qui reste de sa voix.
        float boost = g_playing ? PLAY_BOOST : 1.0f;

        if (!speaking) {
            preroll.push_back(frame);
            if ((int)preroll.size() > PREROLL) preroll.erase(preroll.begin());
            if (rms >= START_RMS * boost) {
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
        silence_ms = (rms < KEEP_RMS * boost) ? silence_ms + FRAME_MS : 0;
        if (silence_ms >= HANGOVER_MS) {
            sendto(sock, "END", 3, 0, (sockaddr*)&dst, sizeof(dst));
            speaking = false;
            std::fprintf(stderr, "[laplace-mic] fin d'énoncé.\n");
        }
    }

    g_run = false;
    g_play_cv.notify_all();
    rx.join();
    px.join();
    pa_simple_free(pa);
    close(sock);
    std::fprintf(stderr, "\n[laplace-mic] arrêt.\n");
    return 0;
}
