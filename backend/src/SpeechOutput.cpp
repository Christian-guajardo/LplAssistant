/**
 * @file SpeechOutput.cpp
 * @brief Implementation of speech synthesis.
 *
 *  
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

// TTS en sous-processus (comme laplace-stt pour le STT) : Piper si installé
// (voix naturelle, third_party/piper + modèle onnx), sinon espeak-ng en
// secours (robotique mais toujours présent). Seul ce fichier connaît les
// moteurs. Le texte passe par un fichier temporaire (pas d'injection shell).
#include <lpl/backend/SpeechOutput.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits.h>

namespace lpl::backend {

namespace {

bool file_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0;
}

// Racine du projet = dossier de l'exécutable, en remontant tant qu'on ne
// trouve pas third_party/ (l'exe vit dans build/linux/x86_64/release/).
std::string find_piper_dir() {
    if (const char* env = std::getenv("LAPLACE_PIPER_DIR")) return env;
    char self[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    std::string dir = n > 0 ? std::string(self, (size_t)n) : "./x";
    for (int up = 0; up < 6; ++up) {
        size_t slash = dir.find_last_of('/');
        if (slash == std::string::npos) break;
        dir.resize(slash);
        if (file_exists(dir + "/third_party/piper/piper"))
            return dir + "/third_party/piper";
    }
    return "";
}

std::string find_voice_model(const std::string& piper_dir) {
    if (const char* env = std::getenv("LAPLACE_TTS_VOICE")) return env;
    // modèle attendu à côté des autres : <racine>/models/
    std::string root = piper_dir.substr(0, piper_dir.find("/third_party"));
    std::string m = root + "/models/fr_FR-siwis-medium.onnx";
    return file_exists(m) ? m : "";
}

// Lance `cmd` et récupère tout stdout.
std::vector<uint8_t> run_capture(const std::string& cmd) {
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return {};
    std::vector<uint8_t> raw;
    uint8_t buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), p)) > 0)
        raw.insert(raw.end(), buf, buf + n);
    if (pclose(p) != 0) return {};
    return raw;
}

// PCM16 quelconque -> 16 kHz (linéaire, même approche que stt.cpp).
std::vector<int16_t> resample_to_16k(const int16_t* pcm, size_t n_in,
                                     uint32_t rate) {
    if (n_in < 2) return {};
    if (rate == 16000) return std::vector<int16_t>(pcm, pcm + n_in);
    double ratio = (double)rate / 16000.0;
    std::vector<int16_t> out;
    out.reserve((size_t)(n_in / ratio) + 1);
    for (double pos = 0.0; pos < (double)(n_in - 1); pos += ratio) {
        size_t i = (size_t)pos;
        float  frac = (float)(pos - (double)i);
        out.push_back((int16_t)(pcm[i] * (1.0f - frac) + pcm[i + 1] * frac));
    }
    return out;
}

// La voix ne doit pas lire les décorations markdown que le LLM produit
// parfois : puces, gras, code, liens. On garde le texte, on jette la syntaxe.
std::string strip_markdown(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    bool line_start = true;
    for (size_t i = 0; i < in.size(); ++i) {
        char c = in[i];
        if (line_start) {
            // en-têtes "#", citations ">", puces "-"/"*", numéros gardés
            while (i < in.size() && (in[i] == '#' || in[i] == '>' ||
                                     in[i] == ' ' || in[i] == '\t'))
                ++i;
            if (i + 1 < in.size() && (in[i] == '-' || in[i] == '*') &&
                in[i + 1] == ' ')
                i += 2;
            if (i >= in.size()) break;
            c = in[i];
            line_start = false;
        }
        if (c == '\n') { line_start = true; out += '\n'; continue; }
        if (c == '*' || c == '_' || c == '`' || c == '~') continue; // emphase/code
        if (c == '[') { // lien [texte](url) -> texte
            size_t close = in.find(']', i);
            if (close != std::string::npos && close + 1 < in.size() &&
                in[close + 1] == '(') {
                size_t paren = in.find(')', close);
                if (paren != std::string::npos) {
                    out += in.substr(i + 1, close - i - 1);
                    i = paren;
                    continue;
                }
            }
        }
        out += c;
    }
    return out;
}

} // namespace

std::vector<int16_t> synthesizeSpeech(const std::string& raw_text) {
    std::string text = strip_markdown(raw_text);
    if (text.empty()) return {};

    char txt_path[] = "/tmp/laplace-tts-XXXXXX";
    int fd = mkstemp(txt_path);
    if (fd < 0) return {};
    {
        std::ofstream f(txt_path);
        f << text;
    }
    close(fd);

    static const std::string piper_dir = find_piper_dir();
    static const std::string voice =
        piper_dir.empty() ? "" : find_voice_model(piper_dir);
    static bool announced = false;
    if (!announced) {
        announced = true;
        std::fprintf(stderr, "[laplace] voix: %s\n",
                     !voice.empty() ? "piper (fr_FR-siwis)"
                                    : "espeak-ng (secours)");
    }

    std::vector<uint8_t> raw;
    uint32_t rate = 22050;
    if (!voice.empty()) {
        // Piper sort du PCM16 brut au taux du modèle (22050 pour siwis).
        std::string cmd = "LD_LIBRARY_PATH='" + piper_dir + "' '" +
                          piper_dir + "/piper' --model '" + voice +
                          "' --output_raw < '" + txt_path + "' 2>/dev/null";
        raw = run_capture(cmd);
        unlink(txt_path);
        if (raw.size() < 4) return {};
        return resample_to_16k((const int16_t*)raw.data(), raw.size() / 2,
                               rate);
    }

    // Secours : espeak-ng streame un WAV (taille RIFF factice) — on saute
    // l'en-tête jusqu'au chunk "data".
    raw = run_capture("espeak-ng -v fr -s 160 --stdout -f '" +
                      std::string(txt_path) + "' 2>/dev/null");
    unlink(txt_path);
    if (raw.size() < 44) return {};
    size_t data_off = 0;
    for (size_t i = 12; i + 8 <= raw.size();) {
        uint32_t sz;
        std::memcpy(&sz, &raw[i + 4], 4);
        if (!std::memcmp(&raw[i], "fmt ", 4) && i + 12 <= raw.size())
            std::memcpy(&rate, &raw[i + 12], 4);
        if (!std::memcmp(&raw[i], "data", 4)) { data_off = i + 8; break; }
        i += 8 + sz + (sz & 1);
    }
    if (data_off == 0 || rate == 0) return {};
    return resample_to_16k((const int16_t*)&raw[data_off],
                           (raw.size() - data_off) / 2, rate);
}

} // namespace lpl::backend
