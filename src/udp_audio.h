#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace laplace {

// Serveur UDP pour satellites audio. Chaque satellite (identifié par son
// couple ip:port source) a son propre buffer d'énoncé : plusieurs personnes
// peuvent parler à des satellites différents en même temps.
//
// Architecture interne (3 étages, découplés pour la réactivité) :
//   1. récepteur   : lit les datagrammes, démux par satellite, détecte la fin
//                    d'énoncé ("END" ou silence) -> file d'énoncés bruts.
//   2. triage      : STT + routage (echo/mot d'appel/commandes/empreinte) EN
//                    PARALLÈLE de la génération -> peut donc annuler la
//                    réponse en cours (barge-in) et donner la priorité à la
//                    nouvelle demande.
//   3. génération  : un seul LLM en RAM, mais la réponse est produite et
//                    diffusée EN LIVE via un pipeline à deux tampons
//                    (LLM -> segments de phrase -> TTS -> trames audio ->
//                    envoi cadencé). Le premier son part dès la 1re phrase,
//                    pas après tout le paragraphe. Annulable à tout instant.

enum class UdpAction {
    Ignore, // rien à faire (pas de mot d'appel, écho, bruit)
    Stop,   // interruption : couper la lecture du satellite + annuler la génération
    Speak,  // dire un texte tout prêt (confirmation, calibrage) — pas de LLM
    Reply,  // générer une réponse au LLM et la diffuser en live
};

struct UdpRoute {
    UdpAction   action = UdpAction::Ignore;
    std::string session;  // Reply : clé d'agent (locuteur reconnu ou "global")
    std::string question; // Reply : la question, mot d'appel retiré
    std::string text;     // Speak : le texte à synthétiser tel quel
};

// Callbacks fournis par main.cpp (le transport ignore tout du LLM/STT/TTS).
struct UdpHandlers {
    // WAV (PCM16 16 kHz mono) -> transcription (déjà nettoyée/minusculisée).
    std::function<std::string(const std::string& wav_path)> transcribe;

    // Décide quoi faire d'une transcription (echo, mot d'appel, commandes de
    // profils, calibrage, empreinte vocale). `pcm` = énoncé brut pour
    // l'empreinte. Appelé depuis le thread de triage.
    std::function<UdpRoute(const std::string& text,
                           const std::vector<int16_t>& pcm,
                           const std::string& peer)> route;

    // Génération streaming : `emit` reçoit les morceaux de texte au fil de
    // l'eau, `should_cancel` est consulté pour abandonner (barge-in). Retourne
    // le texte complet produit. Appelé depuis le thread de génération.
    std::function<std::string(const std::string& session,
                              const std::string& question,
                              const std::function<void(const std::string&)>& emit,
                              const std::function<bool()>& should_cancel)> generate;

    // Texte -> PCM16 mono 16 kHz.
    std::function<std::vector<int16_t>(const std::string& text)> synth;

    // (optionnel) Notifie qu'un segment vient d'être prononcé vers `peer`
    // (sert à l'anti-écho par contenu). Appelé depuis le thread de génération.
    std::function<void(const std::string& peer, const std::string& segment)> on_spoken;
};

void run_udp_audio_server(int port, const UdpHandlers& handlers);

} // namespace laplace
