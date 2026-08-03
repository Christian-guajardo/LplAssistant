/**
 * @file Registry.hpp
 * @brief Named voice profiles, persisted, refined by use.
 *
 * Enrolment keeps a running mean, so a profile sharpens as its owner keeps talking
 * rather than being pinned to whatever the first sample happened to capture.
 *
 * Deleting a profile — one, or all of them — is reachable by voice, because a member
 * of a household who wants to be forgotten should not have to open a terminal. Name
 * matching for deletion is case-insensitive for the same reason: the name arrives
 * from a speech transcript, not from a keyboard.
 *
 * The store is a flat tab-separated file. That is not laziness: a profile is a few
 * dozen floats and a name, the file is inspectable and hand-editable, and a corrupt
 * line degrades to one lost profile instead of an unreadable database.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOICE_REGISTRY_HPP
#    define LPL_VOICE_REGISTRY_HPP

#    include <lpl/voice/Similarity.hpp>

#    include <string>
#    include <vector>

namespace lpl::voice {

class Registry {
  public:
    explicit Registry(std::string path);

    /// Best matching profile name when its similarity reaches @p threshold,
    /// otherwise an empty string. An empty result means "unknown speaker", which is
    /// a normal outcome and routes the utterance to the shared conversation.
    [[nodiscard]] std::string identify(const Signature &signature, float threshold) const;

    /// Adds a sample to @p name, creating the profile when absent. Returns the number
    /// of samples now accumulated, which is what tells a caller whether the profile is
    /// worth trusting yet.
    int enroll(const std::string &name, const Signature &signature);

    /// Removes every profile. Returns how many were removed.
    int clearAll();

    /// Removes one profile, matching @p name case-insensitively.
    bool remove(const std::string &name);

    [[nodiscard]] std::size_t profileCount() const noexcept { return _profiles.size(); }

  private:
    struct Profile {
        std::string name;
        int sampleCount = 0;
        Signature mean;
    };

    void save() const;

    std::string _path;
    std::vector<Profile> _profiles;
};

} // namespace lpl::voice

#endif // LPL_VOICE_REGISTRY_HPP
