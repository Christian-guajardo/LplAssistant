/**
 * @file Registry.cpp
 * @brief Implementation of the persistent voice profile registry.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voice/Registry.hpp>

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <utility>

namespace lpl::voice {

namespace {

bool equalsIgnoringCase(const std::string &left, const std::string &right)
{
    if (left.size() != right.size())
        return false;

    for (std::size_t index = 0; index < left.size(); ++index)
    {
        if (std::tolower(static_cast<unsigned char>(left[index])) !=
            std::tolower(static_cast<unsigned char>(right[index])))
            return false;
    }
    return true;
}

} // namespace

Registry::Registry(std::string path) : _path(std::move(path))
{
    std::ifstream file(_path);
    std::string line;

    while (std::getline(file, line))
    {
        std::istringstream stream(line);
        Profile profile;
        if (!(stream >> profile.name >> profile.sampleCount))
            continue;

        float coefficient = 0.0f;
        while (stream >> coefficient)
            profile.mean.push_back(coefficient);

        // A malformed line costs one profile, not the whole registry.
        if (profile.sampleCount > 0 && !profile.mean.empty())
            _profiles.push_back(std::move(profile));
    }

    if (!_profiles.empty())
        std::fprintf(stderr, "[laplace] %zu profil(s) vocal(aux) chargé(s)\n", _profiles.size());
}

std::string Registry::identify(const Signature &signature, float threshold) const
{
    std::string bestName;
    float best = threshold;

    for (const Profile &profile : _profiles)
    {
        const float similarity = cosineSimilarity(signature, profile.mean);
        if (similarity >= best)
        {
            best = similarity;
            bestName = profile.name;
        }
    }
    return bestName;
}

int Registry::enroll(const std::string &name, const Signature &signature)
{
    for (Profile &profile : _profiles)
    {
        if (profile.name != name)
            continue;

        // Sizes can disagree across builds; skip the blend rather than corrupt the
        // profile, but still count the sample so the caller sees progress.
        if (profile.mean.size() == signature.size())
        {
            for (std::size_t index = 0; index < signature.size(); ++index)
                profile.mean[index] = (profile.mean[index] * static_cast<float>(profile.sampleCount) +
                                       signature[index]) /
                                      static_cast<float>(profile.sampleCount + 1);
        }
        ++profile.sampleCount;
        save();
        return profile.sampleCount;
    }

    _profiles.push_back({name, 1, signature});
    save();
    return 1;
}

int Registry::clearAll()
{
    const int removed = static_cast<int>(_profiles.size());
    _profiles.clear();
    save();
    return removed;
}

bool Registry::remove(const std::string &name)
{
    for (auto it = _profiles.begin(); it != _profiles.end(); ++it)
    {
        if (equalsIgnoringCase(it->name, name))
        {
            _profiles.erase(it);
            save();
            return true;
        }
    }
    return false;
}

void Registry::save() const
{
    std::ofstream file(_path, std::ios::trunc);
    for (const Profile &profile : _profiles)
    {
        file << profile.name << '\t' << profile.sampleCount;
        for (const float coefficient : profile.mean)
            file << '\t' << coefficient;
        file << '\n';
    }
}

} // namespace lpl::voice
