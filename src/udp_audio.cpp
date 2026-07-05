#include "udp_audio.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace laplace {

static constexpr int    SAMPLE_RATE = 16000;
static constexpr int    SILENCE_MS  = 800;             // fin d'énoncé sans "END"
static constexpr int    FRAME_MS    = 40;              // paquets audio de réponse
static constexpr size_t FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS / 1000;
static constexpr size_t MIN_SAMPLES = SAMPLE_RATE / 3;  // < 0.3 s : bruit, ignoré
static constexpr size_t MAX_SAMPLES = SAMPLE_RATE * 60; // garde-fou 60 s

using Clock = std::chrono::steady_clock;

// Écrit un WAV PCM16 mono 16 kHz (format que laplace-stt sait lire).
static void write_wav_16k_mono(const std::string& path,
                               const std::vector<int16_t>& pcm) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("udp: impossible d'écrire " + path);
    uint32_t data_size = (uint32_t)(pcm.size() * 2);
    uint32_t riff_size = 36 + data_size;
    uint16_t fmt = 1, channels = 1, bits = 16, block = 2;
    uint32_t rate = SAMPLE_RATE, byte_rate = rate * block;
    f.write("RIFF", 4); f.write((char*)&riff_size, 4); f.write("WAVE", 4);
    f.write("fmt ", 4);
    uint32_t fmt_size = 16;
    f.write((char*)&fmt_size, 4);
    f.write((char*)&fmt, 2); f.write((char*)&channels, 2);
    f.write((char*)&rate, 4); f.write((char*)&byte_rate, 4);
    f.write((char*)&block, 2); f.write((char*)&bits, 2);
    f.write("data", 4); f.write((char*)&data_size, 4);
    f.write((const char*)pcm.data(), data_size);
}

namespace {

// Énoncé en cours de réception pour un satellite donné.
struct Session {
    std::vector<int16_t> pcm;
    sockaddr_in          addr{};
    Clock::time_point    last_packet;
};

// Énoncé complet, en attente de traitement (STT + LLM + TTS).
struct Job {
    std::string          wav_path;
    std::string          peer_key;
    std::vector<int16_t> pcm;
    sockaddr_in          addr{};
};

std::string peer_key_of(const sockaddr_in& a) {
    char ip[INET_ADDRSTRLEN] = {};
    inet_ntop(AF_INET, &a.sin_addr, ip, sizeof(ip));
    return std::string(ip) + ":" + std::to_string(ntohs(a.sin_port));
}

} // namespace

void run_udp_audio_server(
    int port,
    const std::function<std::string(const std::string&,
                                    const std::vector<int16_t>&,
                                    const std::string&)>& on_utterance,
    const std::function<std::vector<int16_t>(const std::string&)>& synth) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) throw std::runtime_error("udp: socket() a échoué");

    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t)port);
    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        throw std::runtime_error("udp: bind() a échoué sur le port " +
                                 std::to_string(port));
    }

    // Timeout court permanent : il sert d'horloge pour détecter la fin
    // d'énoncé (silence) de chaque satellite indépendamment.
    timeval tv{};
    tv.tv_usec = 100 * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::fprintf(stderr,
                 "[laplace] écoute UDP sur 0.0.0.0:%d — multi-satellites "
                 "(PCM16LE mono 16 kHz, fin d'énoncé: \"END\" ou %d ms de silence)\n",
                 port, SILENCE_MS);

    // File de travaux : le thread récepteur ne bloque jamais, le traitement
    // (STT + LLM + TTS) se fait ici, séquentiellement, énoncé par énoncé.
    std::mutex              mtx;
    std::condition_variable cv;
    std::deque<Job>         jobs;

    std::thread receiver([&] {
        std::unordered_map<std::string, Session> sessions;
        std::vector<char> buf(65536);
        long utt_counter = 0;

        while (true) {
            sockaddr_in from{};
            socklen_t   fromlen = sizeof(from);
            ssize_t n = recvfrom(sock, buf.data(), buf.size(), 0,
                                 (sockaddr*)&from, &fromlen);
            auto now = Clock::now();

            if (n > 0) {
                std::string key = peer_key_of(from);
                Session& s = sessions[key];
                s.addr = from;
                bool end = (n == 3 && !std::memcmp(buf.data(), "END", 3));
                if (!end && n > 1) {
                    size_t n_samples = (size_t)n / 2;
                    if (s.pcm.size() + n_samples <= MAX_SAMPLES) {
                        s.pcm.insert(s.pcm.end(), (const int16_t*)buf.data(),
                                     (const int16_t*)buf.data() + n_samples);
                        s.last_packet = now;
                        end = false;
                    } else {
                        end = true; // garde-fou : on traite ce qu'on a
                    }
                }
                if (end) // fin explicite → flush immédiat ci-dessous
                    s.last_packet = now - std::chrono::milliseconds(SILENCE_MS);
            } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                continue;
            }

            // Vérifie tous les satellites : énoncés terminés (END ou silence).
            for (auto it = sessions.begin(); it != sessions.end();) {
                Session& s = it->second;
                auto idle = std::chrono::duration_cast<std::chrono::milliseconds>(
                                now - s.last_packet).count();
                if (s.pcm.empty() || idle < SILENCE_MS) { ++it; continue; }

                if (s.pcm.size() >= MIN_SAMPLES) {
                    std::string wav = "/tmp/laplace-utt-" +
                                      std::to_string(utt_counter++) + ".wav";
                    write_wav_16k_mono(wav, s.pcm);
                    std::fprintf(stderr,
                                 "[laplace] énoncé reçu: %.1f s depuis %s\n",
                                 (double)s.pcm.size() / SAMPLE_RATE,
                                 it->first.c_str());
                    {
                        std::lock_guard<std::mutex> lk(mtx);
                        jobs.push_back({wav, it->first, std::move(s.pcm),
                                        s.addr});
                    }
                    cv.notify_one();
                }
                it = sessions.erase(it);
            }
        }
    });
    receiver.detach();

    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&] { return !jobs.empty(); });
            job = std::move(jobs.front());
            jobs.pop_front();
        }

        try {
            std::string answer = on_utterance(job.wav_path, job.pcm,
                                              job.peer_key);
            unlink(job.wav_path.c_str());
            if (answer.empty()) {
                // Mot d'appel absent : on se tait, mais on libère le
                // satellite qui attend une réponse.
                sendto(sock, "NOP", 3, 0, (sockaddr*)&job.addr, sizeof(job.addr));
                continue;
            }

            // Texte d'abord (affichage satellite)...
            std::string txt = "TXT:" + answer;
            sendto(sock, txt.data(), txt.size(), 0,
                   (sockaddr*)&job.addr, sizeof(job.addr));

            // ... puis la voix, en paquets de 40 ms légèrement espacés pour
            // ne pas déborder le buffer de réception du satellite.
            std::vector<int16_t> voice = synth(answer);
            for (size_t i = 0; i < voice.size(); i += FRAME_SAMPLES) {
                size_t len = std::min(FRAME_SAMPLES, voice.size() - i);
                sendto(sock, voice.data() + i, len * 2, 0,
                       (sockaddr*)&job.addr, sizeof(job.addr));
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            sendto(sock, "AEND", 4, 0, (sockaddr*)&job.addr, sizeof(job.addr));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[laplace] erreur sur l'énoncé de %s: %s\n",
                         job.peer_key.c_str(), e.what());
        }
    }
}

} // namespace laplace
