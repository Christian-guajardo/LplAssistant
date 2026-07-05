#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace laplace {

// Empreinte vocale légère (pas de modèle neuronal — besoin avant solution) :
// statistiques spectrales long terme (timbre) + hauteur de voix (pitch),
// invariantes au volume, comparées en cosinus. Suffisant pour distinguer
// quelques personnes d'un foyer, pas pour de la biométrie de sécurité.
std::vector<float> voiceprint(const std::vector<int16_t>& pcm_16k_mono);

// Registre persistant (fichier TSV) : nom -> empreinte moyenne.
class VoiceRegistry {
public:
    explicit VoiceRegistry(const std::string& path);

    // Meilleur profil si similarité cosinus >= threshold, sinon "".
    std::string identify(const std::vector<float>& sig, float threshold) const;

    // Ajoute un échantillon au profil (créé si absent, moyenne glissante).
    // Retourne le nombre d'échantillons accumulés.
    int enroll(const std::string& name, const std::vector<float>& sig);

private:
    struct Profile {
        std::string        name;
        int                count = 0;
        std::vector<float> mean;
    };
    std::string          path_;
    std::vector<Profile> profiles_;
    void save() const;
};

} // namespace laplace
