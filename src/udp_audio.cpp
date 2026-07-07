#include "udp_audio.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
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

static constexpr int    SAMPLE_RATE   = 16000;
static constexpr int    SILENCE_MS    = 800;            // fin d'énoncé sans "END"
static constexpr int    FRAME_MS      = 40;             // paquets audio de réponse
static constexpr size_t FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS / 1000;
static constexpr size_t MIN_SAMPLES   = SAMPLE_RATE / 3;  // < 0.3 s : bruit, ignoré
static constexpr size_t MAX_SAMPLES   = SAMPLE_RATE * 60; // garde-fou 60 s

// Segmentation du texte pour le TTS live : on découpe sur la ponctuation, mais
// pas trop court (éviter "M." isolé) ni trop long (garder la latence basse).
static constexpr size_t MIN_SEG = 24;
static constexpr size_t MAX_SEG = 180;

// Tampons du pipeline (double ring buffer). Bornés = mémoire bornée + effet de
// contre-pression : le LLM ne prend pas 10 s d'avance sur la voix.
static constexpr size_t SEG_CAP   = 8;   // segments de phrase en attente de TTS
static constexpr size_t FRAME_CAP = 50;  // ~2 s de trames audio en attente d'envoi

// L'envoi est cadencé légèrement plus vite que le temps réel (34 ms pour 40 ms
// d'audio) : le satellite garde un petit coussin sans jamais accumuler des
// secondes de retard, et un « stop » reste quasi instantané.
static constexpr int SEND_PACE_MS = 34;

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

// File bornée, fermable et abandonnable — l'anneau du pipeline. `close()`
// signale une fin propre (les lecteurs vident puis sortent) ; `abort()` coupe
// tout de suite (barge-in) : lecteurs et écrivains bloqués sortent aussitôt.
template <class T>
class Channel {
public:
    explicit Channel(size_t cap) : cap_(cap) {}

    // Renvoie false si le canal a été abandonné.
    bool push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_ || abort_; });
        if (abort_) return false;
        q_.push_back(std::move(v));
        not_empty_.notify_one();
        return true;
    }

    // Renvoie false si vidé+fermé, ou abandonné.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_ || abort_; });
        if (abort_ || q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }
    void abort() {
        std::lock_guard<std::mutex> lk(m_);
        abort_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    size_t                  cap_;
    std::deque<T>           q_;
    std::mutex              m_;
    std::condition_variable not_empty_, not_full_;
    bool                    closed_ = false, abort_ = false;
};

// Énoncé en cours de réception pour un satellite donné.
struct Session {
    std::vector<int16_t> pcm;
    sockaddr_in          addr{};
    Clock::time_point    last_packet;
};

// Énoncé complet, prêt pour le triage (STT + routage).
struct Job {
    std::string          wav_path;
    std::string          peer_key;
    std::vector<int16_t> pcm;
    sockaddr_in          addr{};
};

// Travail décidé par le triage, à exécuter par le thread de génération.
struct Task {
    UdpAction   action = UdpAction::Reply;
    std::string session;
    std::string question;
    std::string text;
    std::string peer;
    sockaddr_in addr{};
};

std::string peer_key_of(const sockaddr_in& a) {
    char ip[INET_ADDRSTRLEN] = {};
    inet_ntop(AF_INET, &a.sin_addr, ip, sizeof(ip));
    return std::string(ip) + ":" + std::to_string(ntohs(a.sin_port));
}

bool is_sentence_end(char c) {
    return c == '.' || c == '!' || c == '?' || c == ';' || c == ':' || c == '\n';
}

} // namespace

void run_udp_audio_server(int port, const UdpHandlers& h) {
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

    timeval tv{};
    tv.tv_usec = 100 * 1000;             // horloge de détection de silence
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::fprintf(stderr,
                 "[laplace] écoute UDP sur 0.0.0.0:%d — multi-satellites, "
                 "réponse en live, interruptible (PCM16LE mono 16 kHz)\n", port);

    // --- Files entre les 3 étages ------------------------------------------
    std::mutex              raw_mtx;
    std::condition_variable raw_cv;
    std::deque<Job>         raw_jobs;   // récepteur -> triage

    std::mutex              task_mtx;
    std::condition_variable task_cv;
    std::deque<Task>        tasks;      // triage -> génération

    // État de la génération en cours (pour la préemption). `cancel` est lu en
    // boucle par le pipeline ; `peer`/`active` sont sous gen_mtx.
    std::mutex        gen_mtx;
    bool              gen_active = false;
    std::string       gen_peer;
    std::atomic<bool> gen_cancel{false};

    // --- Étage 1 : récepteur (démux + fin d'énoncé) ------------------------
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
                if (end)
                    s.last_packet = now - std::chrono::milliseconds(SILENCE_MS);
            } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                continue;
            }

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
                        std::lock_guard<std::mutex> lk(raw_mtx);
                        raw_jobs.push_back({wav, it->first, std::move(s.pcm),
                                            s.addr});
                    }
                    raw_cv.notify_one();
                }
                it = sessions.erase(it);
            }
        }
    });
    receiver.detach();

    // --- Étage 2 : triage (STT + routage), tourne pendant la génération ----
    std::thread triage([&] {
        while (true) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(raw_mtx);
                raw_cv.wait(lk, [&] { return !raw_jobs.empty(); });
                job = std::move(raw_jobs.front());
                raw_jobs.pop_front();
            }

            UdpRoute r;
            try {
                std::string text = h.transcribe(job.wav_path);
                unlink(job.wav_path.c_str());
                r = h.route(text, job.pcm, job.peer_key);
            } catch (const std::exception& e) {
                unlink(job.wav_path.c_str());
                std::fprintf(stderr, "[laplace] triage %s: %s\n",
                             job.peer_key.c_str(), e.what());
                continue;
            }

            if (r.action == UdpAction::Ignore) continue;

            if (r.action == UdpAction::Stop) {
                {
                    std::lock_guard<std::mutex> lk(gen_mtx);
                    if (gen_active && gen_peer == job.peer_key) gen_cancel = true;
                }
                sendto(sock, "STOP", 4, 0, (sockaddr*)&job.addr,
                       sizeof(job.addr));
                {   // annule aussi une réponse encore en file pour ce satellite
                    std::lock_guard<std::mutex> lk(task_mtx);
                    tasks.erase(std::remove_if(tasks.begin(), tasks.end(),
                        [&](const Task& t) { return t.peer == job.peer_key; }),
                        tasks.end());
                }
                continue;
            }

            // Speak / Reply : la nouvelle demande écrase l'ancienne du même
            // satellite (priorité au dernier énoncé) et préempte la génération.
            {
                std::lock_guard<std::mutex> lk(gen_mtx);
                if (gen_active && gen_peer == job.peer_key) gen_cancel = true;
            }
            {
                std::lock_guard<std::mutex> lk(task_mtx);
                tasks.erase(std::remove_if(tasks.begin(), tasks.end(),
                    [&](const Task& t) { return t.peer == job.peer_key; }),
                    tasks.end());
                Task t;
                t.action = r.action; t.session = r.session;
                t.question = r.question; t.text = r.text;
                t.peer = job.peer_key; t.addr = job.addr;
                tasks.push_back(std::move(t));
            }
            task_cv.notify_one();
        }
    });
    triage.detach();

    // --- Étage 3 : génération + diffusion live (pipeline à deux tampons) ----
    auto run_pipeline = [&](const Task& t) {
        {
            std::lock_guard<std::mutex> lk(gen_mtx);
            gen_cancel = false;
            gen_active = true;
            gen_peer   = t.peer;
        }
        auto cancelled = [&] { return gen_cancel.load(); };

        Channel<std::string>          seg_ch(SEG_CAP);    // LLM  -> TTS
        Channel<std::vector<int16_t>> frame_ch(FRAME_CAP);// TTS  -> envoi
        auto abort_all = [&] { seg_ch.abort(); frame_ch.abort(); };

        // Consommateur TTS : phrase -> texte au satellite + PCM en trames.
        std::thread tts([&] {
            std::string seg;
            while (seg_ch.pop(seg)) {
                if (cancelled()) { abort_all(); break; }
                std::string txt = "TXT:" + seg;
                sendto(sock, txt.data(), txt.size(), 0, (sockaddr*)&t.addr,
                       sizeof(t.addr));
                if (h.on_spoken) h.on_spoken(t.peer, seg);
                std::vector<int16_t> pcm = h.synth(seg);
                for (size_t i = 0; i < pcm.size(); i += FRAME_SAMPLES) {
                    if (cancelled()) { abort_all(); break; }
                    size_t len = std::min(FRAME_SAMPLES, pcm.size() - i);
                    if (!frame_ch.push(std::vector<int16_t>(
                            pcm.begin() + i, pcm.begin() + i + len)))
                        break;
                }
            }
            frame_ch.close();
        });

        // Émetteur réseau : trames -> UDP, cadencé ~temps réel.
        std::thread sender([&] {
            std::vector<int16_t> f;
            while (frame_ch.pop(f)) {
                if (cancelled()) { abort_all(); break; }
                sendto(sock, f.data(), f.size() * 2, 0, (sockaddr*)&t.addr,
                       sizeof(t.addr));
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(SEND_PACE_MS));
            }
        });

        // Producteur (ce thread) : tokens du LLM -> segments de phrase.
        std::string seg_buf;
        auto flush = [&](bool final) {
            size_t start = 0;
            for (size_t i = 0; i < seg_buf.size(); ++i) {
                size_t len = i - start + 1;
                if ((is_sentence_end(seg_buf[i]) && len >= MIN_SEG) ||
                    len >= MAX_SEG) {
                    seg_ch.push(seg_buf.substr(start, len));
                    start = i + 1;
                }
            }
            seg_buf.erase(0, start);
            if (final && !seg_buf.empty()) {
                seg_ch.push(seg_buf);
                seg_buf.clear();
            }
        };

        try {
            if (t.action == UdpAction::Reply) {
                h.generate(t.session, t.question,
                    [&](const std::string& piece) { seg_buf += piece; flush(false); },
                    cancelled);
            } else { // Speak : texte tout prêt
                seg_buf = t.text;
            }
            flush(true);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[laplace] génération %s: %s\n",
                         t.peer.c_str(), e.what());
            gen_cancel = true;
        }

        if (cancelled()) abort_all();
        else             seg_ch.close();

        tts.join();
        sender.join();

        // Fin de flux : STOP purge la lecture (barge-in), AEND clôt proprement.
        if (cancelled())
            sendto(sock, "STOP", 4, 0, (sockaddr*)&t.addr, sizeof(t.addr));
        else
            sendto(sock, "AEND", 4, 0, (sockaddr*)&t.addr, sizeof(t.addr));

        {
            std::lock_guard<std::mutex> lk(gen_mtx);
            gen_active = false;
        }
    };

    // Boucle du thread de génération (celui-ci — un seul LLM en RAM).
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lk(task_mtx);
            task_cv.wait(lk, [&] { return !tasks.empty(); });
            task = std::move(tasks.front());
            tasks.pop_front();
        }
        run_pipeline(task);
    }
}

} // namespace laplace
