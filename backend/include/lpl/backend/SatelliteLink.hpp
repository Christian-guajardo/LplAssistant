/**
 * @file SatelliteLink.hpp
 * @brief The server side of the household nodes.
 *
 * Three decoupled stages, and the decoupling is the whole design. Reception demuxes
 * by source address so several people can speak to different nodes at once. Triage
 * (transcription, wake word, echo rejection, speaker identification) runs IN PARALLEL
 * with generation, which is what lets a new utterance cancel the answer in flight.
 * Generation streams: the first sentence is spoken while the rest is still being
 * produced, with bounded buffers providing back-pressure so the model never runs more
 * than a couple of seconds ahead of the voice.
 * 
 * The transport knows nothing of the model, the transcriber or the synthesiser — all
 * four arrive as callbacks. That is what will let the same orchestration serve a
 * microcontroller node later without changing shape.
 * 
 * TODO(refactor): the WIRE FORMAT (frame size, END, STOP, TXT:, AEND, silence
 * timeout) is still embedded in the implementation below. It belongs in the
 * freestanding lpl::satellite::Protocol, because it has three consumers and two of
 * them will not be hosted. Extracting it is the prerequisite for the kernel's
 * satellite profile, and it is deliberately left visible here rather than quietly
 * duplicated.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_BACKEND_SATELLITELINK_HPP
#    define LPL_BACKEND_SATELLITELINK_HPP

#    include <cstdint>
#    include <functional>
#    include <string>
#    include <vector>

namespace lpl::backend {

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

enum class RouteAction {
    Ignore, // rien à faire (pas de mot d'appel, écho, bruit)
    Stop,   // interruption : couper la lecture du satellite + annuler la génération
    Speak,  // dire un texte tout prêt (confirmation, calibrage) — pas de LLM
    Reply,  // générer une réponse au LLM et la diffuser en live
};

struct Route {
    RouteAction   action = RouteAction::Ignore;
    std::string session;  // Reply : clé d'agent (locuteur reconnu ou "global")
    std::string question; // Reply : la question, mot d'appel retiré
    std::string text;     // Speak : le texte à synthétiser tel quel
};

// Callbacks fournis par main.cpp (le transport ignore tout du LLM/STT/TTS).
struct LinkHandlers {
    // WAV (PCM16 16 kHz mono) -> transcription (déjà nettoyée/minusculisée).
    std::function<std::string(const std::string& wavePath)> transcribe;

    // Décide quoi faire d'une transcription (echo, mot d'appel, commandes de
    // profils, calibrage, empreinte vocale). `pcm` = énoncé brut pour
    // l'empreinte. Appelé depuis le thread de triage.
    std::function<Route(const std::string& text,
                           const std::vector<int16_t>& pcm,
                           const std::string& peer)> route;

    // Génération streaming : `emit` reçoit les morceaux de texte au fil de
    // l'eau, `shouldCancel` est consulté pour abandonner (barge-in). Retourne
    // le texte complet produit. Appelé depuis le thread de génération.
    std::function<std::string(const std::string& session,
                              const std::string& question,
                              const std::function<void(const std::string&)>& emit,
                              const std::function<bool()>& shouldCancel)> generate;

    // Texte -> PCM16 mono 16 kHz.
    std::function<std::vector<int16_t>(const std::string& text)> synth;

    // (optionnel) Notifie qu'un segment vient d'être prononcé vers `peer`
    // (sert à l'anti-écho par contenu). Appelé depuis le thread de génération.
    std::function<void(const std::string& peer, const std::string& segment)> onSpoken;
};

void runSatelliteLink(int port, const LinkHandlers& handlers);

} // namespace lpl::backend

#endif // LPL_BACKEND_SATELLITELINK_HPP
