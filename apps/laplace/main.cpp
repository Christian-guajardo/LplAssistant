/**
 * @file main.cpp
 * @brief The assistant: prompt, single question, audio file, or listening for nodes.
 *
 *  
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include "Identity.hpp"

#include <lpl/mind/Conversation.hpp>
#include <lpl/backend/Settings.hpp>
#include <lpl/backend/VectorStore.hpp>
#include <lpl/backend/Embedder.hpp>
#include <lpl/backend/HostInference.hpp>
#include <lpl/research/Engine.hpp>
#include <lpl/research/LanguageModelClient.hpp>
#include <lpl/backend/SpeechOutput.hpp>
#include <lpl/backend/SatelliteLink.hpp>
#include <lpl/voice/Registry.hpp>
#include <lpl/voice/Voiceprint.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <array>
#include <stdexcept>
#include <unistd.h>
#include <limits.h>


// Le STT tourne dans un binaire séparé (lpl-stt) : le ggml embarqué de
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
    std::string cmd = dir + "/lpl-stt '" + wav + "' 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) throw std::runtime_error("impossible de lancer lpl-stt");
    std::string out;
    std::array<char, 512> buf;
    while (fgets(buf.data(), buf.size(), p)) out += buf.data();
    if (pclose(p) != 0)
        throw std::runtime_error("lpl-stt a échoué sur " + wav);
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
        "  --research <sujet> lance une recherche profonde et écrit un rapport md\n"
        "  --version         affiche la version, le commit et le build, et le LplPlugin utilisé\n"
        "  (sans option)     REPL interactif. Commandes: /mem /forget /research /quit\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "--version")) {
        lpl::apps::printIdentity(stdout, "lpl-assistant");
        return 0;
    }
    std::string wavePath, one_shot, research_topic;
    int listen_port = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--wav") && i + 1 < argc) wavePath = argv[++i];
        else if (!std::strcmp(argv[i], "--ask") && i + 1 < argc) one_shot = argv[++i];
        else if (!std::strcmp(argv[i], "--research") && i + 1 < argc) research_topic = argv[++i];
        else if (!std::strcmp(argv[i], "--listen")) {
            listen_port = (i + 1 < argc && argv[i + 1][0] != '-')
                          ? std::atoi(argv[++i]) : 7777;
            if (listen_port <= 0) { print_usage(argv[0]); return 1; }
        }
        else { print_usage(argv[0]); return !std::strcmp(argv[i], "--help") ? 0 : 1; }
    }

    lpl::backend::Settings cfg = lpl::backend::Settings::fromEnvironment();

    // Mode recherche : n'exige ni la base ni l'embedder (le LLM suffit — et
    // même pas lui si un llama-server est configuré).
    if (!research_topic.empty()) {
        try {
            std::unique_ptr<lpl::backend::HostInference> local;
            if (!std::getenv("LAPLACE_RESEARCH_LLM_URL")) {
                std::fprintf(stderr, "[laplace] chargement du modèle LLM...\n");
                local = std::make_unique<lpl::backend::HostInference>(cfg.languageModelPath, cfg.contextLength, cfg.threadCount);
            }
            auto client = lpl::research::make_llm_client(local.get());
            lpl::research::Engine engine(*client, lpl::research::ResearchOptions::fromEnvironment());
            std::printf("%s\n", engine.run(research_topic).c_str());
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[laplace] erreur fatale: %s\n", e.what());
            return 1;
        }
    }
    try {
        std::fprintf(stderr, "[laplace] chargement des modèles...\n");
        lpl::backend::Embedder embedder(cfg.embeddingModelPath, cfg.threadCount);
        lpl::backend::HostInference llm(cfg.languageModelPath, cfg.contextLength, cfg.threadCount);
        lpl::backend::VectorStore db(cfg.databaseConnection, embedder.dim());
        lpl::mind::Conversation agent(db, embedder, llm, cfg.topMemoryCount,
                    cfg.minimumSimilarity, cfg.maximumNewTokens);
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
            lpl::voice::Registry voices("voiceprints.tsv");
            float voice_threshold = lpl::voice::kDefaultThreshold;
            if (const char* t = std::getenv("LAPLACE_VOICE_THRESHOLD"))
                voice_threshold = (float)std::atof(t);

            // Anti-écho : quand un satellite joue la réponse, son micro peut
            // capter la voix de Laplace. Pas d'annulation d'écho DSP ni de
            // seuil acoustique (l'empreinte maison ne sépare pas assez sa voix
            // TTS d'une voix humaine) : Laplace sait CE qu'il vient de dire à
            // chaque satellite — si un énoncé arrive pendant la fenêtre de
            // lecture et que ses mots sont presque tous dans ce qu'il vient de
            // prononcer, c'est son propre écho, il l'ignore. On peut donc lui
            // parler pendant qu'il parle (barge-in). `onSpoken` alimente cette
            // fenêtre segment par segment, depuis le thread de génération ;
            // `route` la lit depuis le thread de triage -> un mutex.
            struct LastReply {
                std::vector<std::string>              words;
                std::chrono::steady_clock::time_point until{};
            };
            std::map<std::string, LastReply> last_reply; // par satellite
            std::mutex                       reply_mtx;
            auto words_of = [](const std::string& s) {
                std::vector<std::string> w;
                std::string cur;
                for (char c : s) {
                    if (std::isalnum((unsigned char)c))
                        cur += (char)std::tolower((unsigned char)c);
                    else if (!cur.empty()) { w.push_back(cur); cur.clear(); }
                }
                if (!cur.empty()) w.push_back(cur);
                return w;
            };
            auto is_echo = [&](const std::string& peer,
                               const std::string& text) {
                std::lock_guard<std::mutex> lk(reply_mtx);
                auto it = last_reply.find(peer);
                if (it == last_reply.end() ||
                    std::chrono::steady_clock::now() > it->second.until)
                    return false;
                std::vector<std::string> w = words_of(text);
                if (w.size() < 3) return false;
                size_t hit = 0;
                for (const auto& x : w)
                    if (std::find(it->second.words.begin(),
                                  it->second.words.end(), x) !=
                        it->second.words.end())
                        ++hit;
                return hit * 10 >= w.size() * 7; // >= 70 % de mots communs
            };
            std::map<std::string, lpl::mind::Conversation> agents;
            auto agent_for = [&](const std::string& key) -> lpl::mind::Conversation& {
                auto it = agents.find(key);
                if (it == agents.end())
                    it = agents.emplace(key,
                             lpl::mind::Conversation(db, embedder, llm, cfg.topMemoryCount,
                                   cfg.minimumSimilarity,
                                   cfg.maximumNewTokens)).first;
                return it->second;
            };
            auto contains = [](const std::string& s, const char* w) {
                return s.find(w) != std::string::npos;
            };
            // mot suivant le repère `after` (ex. "profil") dans `s`.
            auto word_after = [](const std::string& s, const char* after) {
                auto is_sep = [](char c) {
                    return std::isspace((unsigned char)c) || c == ',' ||
                           c == '.' || c == '!' || c == '?' || c == ':' ||
                           c == ';' || c == '\'';
                };
                size_t p = s.find(after);
                if (p == std::string::npos) return std::string();
                p += std::strlen(after);
                while (p < s.size() && !is_sep(s[p])) ++p; // fin du repère
                while (p < s.size() && is_sep(s[p])) ++p;   // séparateurs
                size_t e = p;
                while (e < s.size() && !is_sep(s[e])) ++e;
                return s.substr(p, e - p);
            };

            lpl::backend::LinkHandlers h;

            // STT + minusculisation intégrale du prompt capturé (simplifie tout
            // le reste : mot d'appel, écho, commandes, calibrage).
            h.transcribe = [&](const std::string& wav) -> std::string {
                std::string text;
                try { text = transcribe_via_subprocess(wav); }
                catch (const std::exception& e) {
                    std::fprintf(stderr, "[laplace] STT: %s\n", e.what());
                    return "";
                }
                for (auto& c : text) c = (char)std::tolower((unsigned char)c);
                return text;
            };

            // Routage : décide écho / mot d'appel / commandes / réponse.
            h.route = [&](const std::string& text,
                          const std::vector<int16_t>& pcm,
                          const std::string& peer) -> lpl::backend::Route {
                lpl::backend::Route r; // Ignore par défaut
                if (text.empty()) return r;
                if (is_echo(peer, text)) {
                    std::fprintf(stderr,
                        "[laplace] %s : mon propre écho, ignoré: %s\n",
                        peer.c_str(), text.c_str());
                    return r;
                }
                std::string q = strip_wake_word(text); // texte déjà minuscule
                if (q.empty()) {
                    std::fprintf(stderr,
                        "[laplace] %s (ignoré, pas de mot d'appel): %s\n",
                        peer.c_str(), text.c_str());
                    return r;
                }

                // Interruption : "stop / arrête / tais-toi / chut / silence".
                for (const char* w : {"stop", "arrête", "arrete", "tais-toi",
                                      "tais toi", "chut", "silence"}) {
                    if (q.rfind(w, 0) == 0) {
                        std::fprintf(stderr,
                            "[laplace] %s : interruption demandée\n",
                            peer.c_str());
                        r.action = lpl::backend::RouteAction::Stop;
                        return r;
                    }
                }

                lpl::voice::Signature sig = lpl::voice::computeVoiceprint(pcm);

                // Suppression de profils vocaux. Priorité au « tout effacer ».
                bool wipe_word = contains(q, "profil") &&
                    (contains(q, "tous") || contains(q, "tout") ||
                     contains(q, "toutes") || contains(q, "réinitialise") ||
                     contains(q, "reinitialise") || contains(q, "reset"));
                bool del_word = contains(q, "profil") &&
                    (q.rfind("supprime", 0) == 0 || q.rfind("efface", 0) == 0 ||
                     q.rfind("oublie", 0) == 0 || q.rfind("retire", 0) == 0);
                if (wipe_word) {
                    int n = voices.clearAll();
                    std::fprintf(stderr,
                        "[laplace] %d profil(s) vocal(aux) supprimé(s)\n", n);
                    r.action = lpl::backend::RouteAction::Speak;
                    r.text = n ? "J'ai supprimé les " + std::to_string(n) +
                                 " profils vocaux."
                               : "Il n'y avait aucun profil à supprimer.";
                    return r;
                }
                if (del_word) {
                    std::string name = word_after(q, "profil");
                    r.action = lpl::backend::RouteAction::Speak;
                    if (name.empty()) {
                        int n = voices.clearAll();
                        r.text = "J'ai supprimé les " + std::to_string(n) +
                                 " profils vocaux.";
                    } else if (voices.remove(name)) {
                        r.text = "Profil de " + name + " supprimé.";
                    } else {
                        r.text = "Je n'ai pas trouvé de profil au nom de " +
                                 name + ".";
                    }
                    return r;
                }

                // Calibrage : "calibration <prénom> ..." (1er mot en "cali").
                if (q.rfind("cali", 0) == 0) {
                    std::string name = word_after(q, "cali");
                    r.action = lpl::backend::RouteAction::Speak;
                    if (name.empty())
                        r.text = "Dis : Laplace, calibration, puis ton prénom.";
                    else if (sig.empty())
                        r.text = "Énoncé trop court pour le calibrage, répète "
                                 "en parlant un peu plus longtemps.";
                    else {
                        int n = voices.enroll(name, sig);
                        std::fprintf(stderr,
                            "[laplace] profil vocal '%s' : %d échantillon(s)\n",
                            name.c_str(), n);
                        r.text = "Profil vocal de " + name + " enregistré, "
                                 "échantillon numéro " + std::to_string(n) +
                                 ". Répète pour affiner.";
                    }
                    return r;
                }

                std::string who =
                    sig.empty() ? "" : voices.identify(sig, voice_threshold);
                r.action   = lpl::backend::RouteAction::Reply;
                r.session  = who.empty() ? "global" : who;
                r.question = q;
                std::fprintf(stderr, "[laplace] Vous (%s @ %s): %s\n",
                             who.empty() ? "?" : who.c_str(), peer.c_str(),
                             q.c_str());
                return r;
            };

            // Génération streaming (annulable), thread de génération.
            h.generate = [&](const std::string& session,
                             const std::string& question,
                             const std::function<void(const std::string&)>& emit,
                             const std::function<bool()>& shouldCancel)
                -> std::string {
                return agent_for(session).ask(question, emit, shouldCancel);
            };

            h.synth = lpl::backend::synthesizeSpeech;

            // Alimente la fenêtre anti-écho au fil des phrases prononcées.
            h.onSpoken = [&](const std::string& peer, const std::string& seg) {
                std::vector<std::string> w = words_of(seg);
                std::lock_guard<std::mutex> lk(reply_mtx);
                LastReply& lr = last_reply[peer];
                lr.words.insert(lr.words.end(), w.begin(), w.end());
                if (lr.words.size() > 400)
                    lr.words.erase(lr.words.begin(), lr.words.end() - 400);
                lr.until = std::chrono::steady_clock::now() +
                           std::chrono::seconds(8);
            };

            lpl::backend::runSatelliteLink(listen_port, h);
            return 0; // (boucle infinie, on n'arrive ici jamais)
        }
        if (!wavePath.empty()) {
            std::string text = transcribe_via_subprocess(wavePath);
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

        std::printf("Laplace prêt. /mem /forget /research /quit\n");
        std::string line;
        while (true) {
            std::printf("\nVous> ");
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;
            if (line == "/quit" || line == "/exit") break;
            if (line.rfind("/research ", 0) == 0) {
                // Recherche profonde synchrone dans le REPL (le mode
                // fire-and-forget vocal viendra avec l'intégration UDP).
                try {
                    lpl::research::LocalLanguageModelClient client(llm);
                    lpl::research::Engine engine(client, lpl::research::ResearchOptions::fromEnvironment());
                    std::string path = engine.run(line.substr(10));
                    std::printf("Rapport : %s\n", path.c_str());
                } catch (const std::exception& e) {
                    std::printf("Recherche échouée : %s\n", e.what());
                }
                continue;
            }
            if (line == "/mem") {
                for (const auto& m : db.recent(10))
                    std::printf("[%lld|%s|%s] %.120s\n", m.id, m.category.c_str(),
                                m.createdAt.substr(0, 19).c_str(), m.content.c_str());
                continue;
            }
            if (line == "/forget") {
                db.forgetAll();
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
