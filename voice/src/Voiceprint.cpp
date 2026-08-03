/**
 * @file Voiceprint.cpp
 * @brief Implementation of vocal signature extraction.
 *
 * Three decisions carry the quality of the result, and none of them is obvious:
 *
 *   - silent frames are SKIPPED rather than averaged in. Silence has no timbre, and
 *     including it drags every profile towards the same point;
 *   - per-frame log energies are centred before accumulation, which is what removes
 *     loudness from the signature — the same voice at two distances must match;
 *   - pitch is divided by fifty before storage, so it lands on a scale comparable to
 *     the log energies. Without that the cosine would be dominated by pitch alone,
 *     and two people with similar voices at different pitches would never match on
 *     timbre.
 *
 * @author Christian-guajardo, MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/voice/Voiceprint.hpp>

#include <algorithm>
#include <cmath>

namespace lpl::voice {

namespace {

constexpr int kFastFourierTransformSize = 512; // 32 ms window
constexpr int kHopSamples = 160;               // 10 ms step
constexpr int kBandCount = 24;                 // log-spaced, 80 Hz to 7 kHz
constexpr float kMinimumRootMeanSquare = 0.015f;
constexpr int kMinimumVoicedFrames = 20;
constexpr int kMinimumPitchSamples = 5;
constexpr float kPitchScale = 50.0f;

/// In-place iterative radix-2 transform, real and imaginary parts kept separate.
void fastFourierTransform(std::vector<float> &real, std::vector<float> &imaginary)
{
    const int size = static_cast<int>(real.size());

    for (int i = 1, j = 0; i < size; ++i)
    {
        int bit = size >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
        {
            std::swap(real[i], real[j]);
            std::swap(imaginary[i], imaginary[j]);
        }
    }

    for (int length = 2; length <= size; length <<= 1)
    {
        const float angle = -2.0f * static_cast<float>(M_PI) / static_cast<float>(length);
        const float stepReal = std::cos(angle);
        const float stepImaginary = std::sin(angle);

        for (int base = 0; base < size; base += length)
        {
            float twiddleReal = 1.0f;
            float twiddleImaginary = 0.0f;

            for (int k = 0; k < length / 2; ++k)
            {
                const int lo = base + k;
                const int hi = lo + length / 2;

                const float upperReal = real[lo];
                const float upperImaginary = imaginary[lo];
                const float lowerReal = real[hi] * twiddleReal - imaginary[hi] * twiddleImaginary;
                const float lowerImaginary = real[hi] * twiddleImaginary + imaginary[hi] * twiddleReal;

                real[lo] = upperReal + lowerReal;
                imaginary[lo] = upperImaginary + lowerImaginary;
                real[hi] = upperReal - lowerReal;
                imaginary[hi] = upperImaginary - lowerImaginary;

                const float nextReal = twiddleReal * stepReal - twiddleImaginary * stepImaginary;
                twiddleImaginary = twiddleReal * stepImaginary + twiddleImaginary * stepReal;
                twiddleReal = nextReal;
            }
        }
    }
}

/// Autocorrelation pitch over 60-400 Hz; zero when the frame is unvoiced.
float framePitch(const float *samples, int count)
{
    const int minimumLag = kSampleRate / 400;
    const int maximumLag = kSampleRate / 60;
    if (count <= maximumLag)
        return 0.0f;

    float energy = 0.0f;
    for (int i = 0; i < count; ++i)
        energy += samples[i] * samples[i];
    if (energy <= 0.0f)
        return 0.0f;

    float best = 0.0f;
    int bestLag = 0;
    for (int lag = minimumLag; lag <= maximumLag; ++lag)
    {
        float accumulator = 0.0f;
        for (int i = 0; i + lag < count; ++i)
            accumulator += samples[i] * samples[i + lag];
        if (accumulator > best)
        {
            best = accumulator;
            bestLag = lag;
        }
    }

    if (bestLag == 0 || best / energy < 0.30f)
        return 0.0f;

    return static_cast<float>(kSampleRate) / static_cast<float>(bestLag);
}

} // namespace

Signature computeVoiceprint(const std::vector<std::int16_t> &pulseCodeModulation)
{
    if (static_cast<int>(pulseCodeModulation.size()) < kFastFourierTransformSize * 2)
        return {};

    // Log-spaced band edges, expressed in transform bins.
    constexpr float kLowFrequency = 80.0f;
    constexpr float kHighFrequency = 7000.0f;
    int bandBin[kBandCount + 1];
    for (int band = 0; band <= kBandCount; ++band)
    {
        const float frequency = kLowFrequency * std::pow(kHighFrequency / kLowFrequency,
                                                         static_cast<float>(band) / kBandCount);
        bandBin[band] = static_cast<int>(frequency * kFastFourierTransformSize / kSampleRate);
    }

    std::vector<double> bandSum(kBandCount, 0.0);
    std::vector<double> bandSquare(kBandCount, 0.0);
    std::vector<float> pitches;
    int voicedFrames = 0;

    std::vector<float> window(kFastFourierTransformSize);
    std::vector<float> real(kFastFourierTransformSize);
    std::vector<float> imaginary(kFastFourierTransformSize);
    for (int i = 0; i < kFastFourierTransformSize; ++i)
        window[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * static_cast<float>(i) /
                                           (kFastFourierTransformSize - 1));

    for (std::size_t offset = 0;
         offset + kFastFourierTransformSize <= pulseCodeModulation.size();
         offset += kHopSamples)
    {
        float rootMeanSquare = 0.0f;
        for (int i = 0; i < kFastFourierTransformSize; ++i)
        {
            real[i] = static_cast<float>(pulseCodeModulation[offset + i]) / 32768.0f;
            rootMeanSquare += real[i] * real[i];
        }
        rootMeanSquare = std::sqrt(rootMeanSquare / kFastFourierTransformSize);

        // Silence has no timbre; averaging it in drags every profile together.
        if (rootMeanSquare < kMinimumRootMeanSquare)
            continue;

        const float pitch = framePitch(real.data(), kFastFourierTransformSize);
        if (pitch > 0.0f)
            pitches.push_back(pitch);

        for (int i = 0; i < kFastFourierTransformSize; ++i)
        {
            real[i] *= window[i];
            imaginary[i] = 0.0f;
        }
        fastFourierTransform(real, imaginary);

        // Centring per frame is what removes loudness from the signature.
        float logEnergy[kBandCount];
        float mean = 0.0f;
        for (int band = 0; band < kBandCount; ++band)
        {
            double energy = 1e-10;
            for (int bin = bandBin[band];
                 bin < bandBin[band + 1] && bin < kFastFourierTransformSize / 2; ++bin)
                energy += static_cast<double>(real[bin]) * real[bin] +
                          static_cast<double>(imaginary[bin]) * imaginary[bin];
            logEnergy[band] = std::log(static_cast<float>(energy));
            mean += logEnergy[band];
        }
        mean /= kBandCount;

        for (int band = 0; band < kBandCount; ++band)
        {
            const float centred = logEnergy[band] - mean;
            bandSum[band] += centred;
            bandSquare[band] += static_cast<double>(centred) * centred;
        }
        ++voicedFrames;
    }

    if (voicedFrames < kMinimumVoicedFrames ||
        pitches.size() < static_cast<std::size_t>(kMinimumPitchSamples))
        return {};

    Signature signature;
    signature.reserve(kBandCount * 2 + 2);
    for (int band = 0; band < kBandCount; ++band)
    {
        const double bandMean = bandSum[band] / voicedFrames;
        signature.push_back(static_cast<float>(bandMean));
        signature.push_back(static_cast<float>(
            std::sqrt(std::max(0.0, bandSquare[band] / voicedFrames - bandMean * bandMean))));
    }

    double pitchMean = 0.0;
    for (const float pitch : pitches)
        pitchMean += pitch;
    pitchMean /= static_cast<double>(pitches.size());

    double pitchVariance = 0.0;
    for (const float pitch : pitches)
        pitchVariance += (pitch - pitchMean) * (pitch - pitchMean);

    // Scaled so pitch cannot dominate the cosine over the timbre coefficients.
    signature.push_back(static_cast<float>(pitchMean / kPitchScale));
    signature.push_back(static_cast<float>(
        std::sqrt(pitchVariance / static_cast<double>(pitches.size())) / kPitchScale));

    return signature;
}

} // namespace lpl::voice
