#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace laplace {

// Serveur UDP pour satellites audio. Chaque satellite (identifié par son
// couple ip:port source) a son propre buffer d'énoncé : plusieurs personnes
// peuvent parler à des satellites différents en même temps, les énoncés sont
// traités un par un (un seul LLM) mais jamais mélangés.
//
// on_utterance(wav_path, pcm, peer_key) -> texte de réponse ("" = ignorer,
// aucune réponse envoyée — sert au filtrage par mot d'appel). `pcm` est
// l'énoncé brut (16 kHz mono), utile pour l'empreinte vocale.
// synth(texte) -> PCM16 mono 16 kHz à streamer vers le satellite (vide = pas
// d'audio, seul le texte est envoyé).
//
// Réponse envoyée au satellite émetteur uniquement :
//   1 datagramme "TXT:<texte>", puis l'audio en paquets de 40 ms, puis "AEND".
void run_udp_audio_server(
    int port,
    const std::function<std::string(const std::string& wav_path,
                                    const std::vector<int16_t>& pcm,
                                    const std::string& peer_key)>& on_utterance,
    const std::function<std::vector<int16_t>(const std::string&)>& synth);

} // namespace laplace
