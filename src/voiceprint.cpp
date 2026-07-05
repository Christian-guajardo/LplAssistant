#include "voiceprint.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace laplace {

namespace {

constexpr int   RATE      = 16000;
constexpr int   NFFT      = 512;   // fenêtre 32 ms
constexpr int   HOP       = 160;   // pas 10 ms
constexpr int   NBANDS    = 24;    // bandes log-espacées 80 Hz -> 7 kHz
constexpr float MIN_RMS   = 0.015f; // on n'analyse que les trames voisées

// FFT radix-2 itérative en place (réel/imaginaire séparés).
void fft(std::vector<float>& re, std::vector<float>& im) {
    int n = (int)re.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        float wr = std::cos(ang), wi = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; ++k) {
                float ur = re[i + k], ui = im[i + k];
                float vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                float vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr;           im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                float ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

// Pitch d'une trame par autocorrélation (60–400 Hz), 0 si non voisée.
float frame_pitch(const float* s, int n) {
    int lag_min = RATE / 400, lag_max = RATE / 60;
    if (n <= lag_max) return 0.0f;
    float e0 = 0.0f;
    for (int i = 0; i < n; ++i) e0 += s[i] * s[i];
    if (e0 <= 0.0f) return 0.0f;
    float best = 0.0f;
    int   best_lag = 0;
    for (int lag = lag_min; lag <= lag_max; ++lag) {
        float acc = 0.0f;
        for (int i = 0; i + lag < n; ++i) acc += s[i] * s[i + lag];
        if (acc > best) { best = acc; best_lag = lag; }
    }
    if (best_lag == 0 || best / e0 < 0.30f) return 0.0f;
    return (float)RATE / best_lag;
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return -1.0f;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na <= 0 || nb <= 0) return -1.0f;
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

} // namespace

std::vector<float> voiceprint(const std::vector<int16_t>& pcm) {
    if ((int)pcm.size() < NFFT * 2) return {};

    // Bornes des bandes log-espacées (en bins FFT), 80 Hz -> 7 kHz.
    float f_lo = 80.0f, f_hi = 7000.0f;
    int band_bin[NBANDS + 1];
    for (int b = 0; b <= NBANDS; ++b) {
        float f = f_lo * std::pow(f_hi / f_lo, (float)b / NBANDS);
        band_bin[b] = (int)(f * NFFT / RATE);
    }

    std::vector<double> band_sum(NBANDS, 0), band_sq(NBANDS, 0);
    std::vector<float>  pitches;
    int n_frames = 0;

    std::vector<float> win(NFFT), re(NFFT), im(NFFT);
    for (int i = 0; i < NFFT; ++i)
        win[i] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (NFFT - 1));

    for (size_t off = 0; off + NFFT <= pcm.size(); off += HOP) {
        float rms = 0.0f;
        for (int i = 0; i < NFFT; ++i) {
            re[i] = pcm[off + i] / 32768.0f;
            rms += re[i] * re[i];
        }
        rms = std::sqrt(rms / NFFT);
        if (rms < MIN_RMS) continue; // silence : n'apporte que du bruit

        float p = frame_pitch(re.data(), NFFT);
        if (p > 0.0f) pitches.push_back(p);

        for (int i = 0; i < NFFT; ++i) { re[i] *= win[i]; im[i] = 0.0f; }
        fft(re, im);

        // Log-énergies par bande, centrées par trame (retire le volume).
        float logs[NBANDS];
        float mean = 0.0f;
        for (int b = 0; b < NBANDS; ++b) {
            double e = 1e-10;
            for (int k = band_bin[b]; k < band_bin[b + 1] && k < NFFT / 2; ++k)
                e += (double)re[k] * re[k] + (double)im[k] * im[k];
            logs[b] = std::log((float)e);
            mean += logs[b];
        }
        mean /= NBANDS;
        for (int b = 0; b < NBANDS; ++b) {
            float v = logs[b] - mean;
            band_sum[b] += v;
            band_sq[b]  += (double)v * v;
        }
        ++n_frames;
    }
    if (n_frames < 20 || pitches.size() < 5) return {}; // trop court/bruité

    // Signature : moyenne + écart-type par bande (timbre) + pitch moyen/σ.
    std::vector<float> sig;
    sig.reserve(NBANDS * 2 + 2);
    for (int b = 0; b < NBANDS; ++b) {
        double m = band_sum[b] / n_frames;
        sig.push_back((float)m);
        sig.push_back((float)std::sqrt(std::max(0.0, band_sq[b] / n_frames - m * m)));
    }
    double pm = 0;
    for (float p : pitches) pm += p;
    pm /= pitches.size();
    double pv = 0;
    for (float p : pitches) pv += (p - pm) * (p - pm);
    // Pitch ramené à une échelle comparable aux log-énergies.
    sig.push_back((float)(pm / 50.0));
    sig.push_back((float)(std::sqrt(pv / pitches.size()) / 50.0));
    return sig;
}

VoiceRegistry::VoiceRegistry(const std::string& path) : path_(path) {
    std::ifstream f(path_);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        Profile p;
        if (!(is >> p.name >> p.count)) continue;
        float v;
        while (is >> v) p.mean.push_back(v);
        if (p.count > 0 && !p.mean.empty()) profiles_.push_back(std::move(p));
    }
    if (!profiles_.empty())
        std::fprintf(stderr, "[laplace] %zu profil(s) vocal(aux) chargé(s)\n",
                     profiles_.size());
}

std::string VoiceRegistry::identify(const std::vector<float>& sig,
                                    float threshold) const {
    std::string best_name;
    float best = threshold;
    for (const auto& p : profiles_) {
        float c = cosine(sig, p.mean);
        if (c >= best) { best = c; best_name = p.name; }
    }
    return best_name;
}

int VoiceRegistry::enroll(const std::string& name,
                          const std::vector<float>& sig) {
    for (auto& p : profiles_) {
        if (p.name != name) continue;
        if (p.mean.size() == sig.size())
            for (size_t i = 0; i < sig.size(); ++i)
                p.mean[i] = (p.mean[i] * p.count + sig[i]) / (p.count + 1);
        ++p.count;
        save();
        return p.count;
    }
    profiles_.push_back({name, 1, sig});
    save();
    return 1;
}

void VoiceRegistry::save() const {
    std::ofstream f(path_, std::ios::trunc);
    for (const auto& p : profiles_) {
        f << p.name << '\t' << p.count;
        for (float v : p.mean) f << '\t' << v;
        f << '\n';
    }
}

} // namespace laplace
