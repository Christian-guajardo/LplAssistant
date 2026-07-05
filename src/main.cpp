#include "agent.h"
#include "config.h"
#include "db.h"
#include "embedder.h"
#include "llm.h"
#include "tts.h"
#include "udp_audio.h"
#include "voiceprint.h"
#include <algorithm>
#include <cctype>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include <array>
#include <stdexcept>
#include <unistd.h>
#include <limits.h>

using namespace laplace;

// Le STT tourne dans un binaire séparé (laplace-stt) : le ggml embarqué de
// whisper.cpp est incompatible au link avec celui de llama.cpp.
static std::string transcribe_via_subprocess(const std::string& wav) {
    char self[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    std::string dir = ".";
    if (n > 0) {
        self[n] = '\0';
        std::string s(self);
        dir = s.substr(0, s.find_last_of('/'));
    }
    std::string cmd = dir + "/laplace-stt '" + wav + "' 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) throw std::runtime_error("impossible de lancer laplace-stt");
    std::string out;
    std::array<char, 512> buf;
    while (fgets(buf.data(), buf.size(), p)) out += buf.data();
    if (pclose(p) != 0)
        throw std::runtime_error("laplace-stt a échoué sur " + wav);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

// Mot d'appel : Laplace ne répond que si l'énoncé commence par "laplace"
// (vérifié sur la transcription — pas de TinyML tant qu'il n'y a pas
// d'ESP32). Retourne la question sans le mot d'appel, ou "" si absent.
static std::string strip_wake_word(const std::string& text) {
    std::string low;
    for (char c : text) low += (char)std::tolower((unsigned char)c);
    // whisper transcrit parfois "la place" / "la passe" / "laplasse"...
    for (const char* w : {"laplace", "laplasse", "la place", "la-place",
                          "la passe", "la plasse"}) {
        size_t pos = low.find(w);
        if (pos != std::string::npos && pos <= 2) { // tolère ponctuation de tête
            size_t after = pos + std::strlen(w);
            while (after < text.size() &&
                   (std::isspace((unsigned char)text[after]) ||
                    text[after] == ',' || text[after] == '.' ||
                    text[after] == '!' || text[after] == '?' ||
                    text[after] == ':'))
                ++after;
            return text.substr(after);
        }
    }
    return "";
}

static void print_usage(const char* prog) {
    std::printf(
        "Laplace — assistant personnel local\n"
        "Usage: %s [options]\n"
        "  --wav <fichier>   transcrit un WAV PCM16 (mono/stéréo, tout taux) et l'utilise comme question\n"
        "  --ask <texte>     pose une question unique puis quitte\n"
        "  --listen [port]   serveur UDP audio pour satellites (défaut 7777)\n"
        "  (sans option)     REPL interactif. Commandes: /mem /forget /quit\n",
        prog);
}

int main(int argc, char** argv) {
    std::string wav_path, one_shot;
    int listen_port = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--wav") && i + 1 < argc) wav_path = argv[++i];
        else if (!std::strcmp(argv[i], "--ask") && i + 1 < argc) one_shot = argv[++i];
        else if (!std::strcmp(argv[i], "--listen")) {
            listen_port = (i + 1 < argc && argv[i + 1][0] != '-')
                          ? std::atoi(argv[++i]) : 7777;
            if (listen_port <= 0) { print_usage(argv[0]); return 1; }
        }
        else { print_usage(argv[0]); return !std::strcmp(argv[i], "--help") ? 0 : 1; }
    }

    Config cfg = Config::from_env();
    try {
        std::fprintf(stderr, "[laplace] chargement des modèles...\n");
        Embedder embedder(cfg.embed_model, cfg.n_threads);
        Llm llm(cfg.llm_model, cfg.n_ctx, cfg.n_threads);
        Db db(cfg.db_conn, embedder.dim());
        Agent agent(db, embedder, llm, cfg.top_k_memories,
                    cfg.min_similarity, cfg.max_new_tokens);
        std::fprintf(stderr, "[laplace] prêt (embeddings %dd)\n", embedder.dim());

        auto stream = [](const std::string& piece) {
            std::fputs(piece.c_str(), stdout);
            std::fflush(stdout);
        };

        if (listen_port > 0) {
            // Sessions par LOCUTEUR (empreinte vocale), pas par satellite :
            // une conversation suit la personne de pièce en pièce. Voix non
            // reconnue -> contexte partagé "global". Calibrage par la voix :
            // "Laplace, calibration <prénom>" (répéter 2-3 fois pour affiner).
            VoiceRegistry voices("voiceprints.tsv");
            float voice_threshold = 0.80f;
            if (const char* t = std::getenv("LAPLACE_VOICE_THRESHOLD"))
                voice_threshold = (float)std::atof(t);
            std::map<std::string, Agent> agents;
            auto agent_for = [&](const std::string& key) -> Agent& {
                auto it = agents.find(key);
                if (it == agents.end())
                    it = agents.emplace(key,
                             Agent(db, embedder, llm, cfg.top_k_memories,
                                   cfg.min_similarity,
                                   cfg.max_new_tokens)).first;
                return it->second;
            };

            run_udp_audio_server(listen_port,
                [&](const std::string& wav, const std::vector<int16_t>& pcm,
                    const std::string& peer) -> std::string {
                    std::string text = transcribe_via_subprocess(wav);
                    if (text.empty()) return "";
                    std::string question = strip_wake_word(text);
                    if (question.empty()) {
                        std::fprintf(stderr,
                            "[laplace] %s (ignoré, pas de mot d'appel): %s\n",
                            peer.c_str(), text.c_str());
                        return "";
                    }

                    std::vector<float> sig = voiceprint(pcm);

                    // Calibrage : "calibration <prénom> ..." — détection
                    // tolérante aux fautes de transcription (1er mot en
                    // "cali..."), prénom = mot suivant uniquement.
                    std::string low;
                    for (char c : question)
                        low += (char)std::tolower((unsigned char)c);
                    if (low.rfind("cali", 0) == 0) {
                        auto is_sep = [](char c) {
                            return std::isspace((unsigned char)c) ||
                                   c == ',' || c == '.' || c == '!' ||
                                   c == '?' || c == ':' || c == ';';
                        };
                        size_t i = 0;
                        while (i < question.size() && !is_sep(question[i])) ++i;
                        while (i < question.size() && is_sep(question[i])) ++i;
                        size_t j = i;
                        while (j < question.size() && !is_sep(question[j])) ++j;
                        std::string name = question.substr(i, j - i);
                        if (name.empty())
                            return "Dis : Laplace, calibration, puis ton prénom.";
                        if (sig.empty())
                            return "Énoncé trop court pour le calibrage, "
                                   "répète en parlant un peu plus longtemps.";
                        int n = voices.enroll(name, sig);
                        std::fprintf(stderr,
                            "[laplace] profil vocal '%s' : %d échantillon(s)\n",
                            name.c_str(), n);
                        return "Profil vocal de " + name + " enregistré, "
                               "échantillon numéro " + std::to_string(n) +
                               ". Répète pour affiner.";
                    }

                    std::string who =
                        sig.empty() ? "" : voices.identify(sig, voice_threshold);
                    std::string session = who.empty() ? "global" : who;
                    std::printf("Vous (%s @ %s): %s\nLaplace: ",
                                who.empty() ? "?" : who.c_str(), peer.c_str(),
                                question.c_str());
                    std::string answer = agent_for(session).ask(question, stream);
                    std::printf("\n");
                    return answer;
                },
                tts_synthesize_16k);
            return 0; // (boucle infinie, on n'arrive ici jamais)
        }
        if (!wav_path.empty()) {
            std::string text = transcribe_via_subprocess(wav_path);
            std::printf("Vous (audio): %s\nLaplace: ", text.c_str());
            agent.ask(text, stream);
            std::printf("\n");
            return 0;
        }
        if (!one_shot.empty()) {
            std::printf("Laplace: ");
            agent.ask(one_shot, stream);
            std::printf("\n");
            return 0;
        }

        std::printf("Laplace prêt. /mem /forget /quit\n");
        std::string line;
        while (true) {
            std::printf("\nVous> ");
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;
            if (line == "/quit" || line == "/exit") break;
            if (line == "/mem") {
                for (const auto& m : db.recent(10))
                    std::printf("[%lld|%s|%s] %.120s\n", m.id, m.category.c_str(),
                                m.created_at.substr(0, 19).c_str(), m.content.c_str());
                continue;
            }
            if (line == "/forget") {
                db.forget_all();
                std::printf("Mémoire effacée.\n");
                continue;
            }
            std::printf("Laplace: ");
            agent.ask(line, stream);
            std::printf("\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[laplace] erreur fatale: %s\n", e.what());
        return 1;
    }
    return 0;
}
