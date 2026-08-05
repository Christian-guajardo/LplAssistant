/**
 * @file test_satellite_parity.cpp
 * @brief One protocol, three implementations, one set of decisions.
 *
 * The gate for the satellite floor. What it guards is not an algorithm but an
 * AGREEMENT: a hosted development node, the kernel's satellite profile and an
 * eventual microcontroller firmware share no register, no allocator and no
 * instruction set, and they must still decide the same thing about the same audio —
 * when to start sending, when to stop, whether the word was heard, and whether the
 * node is hearing itself.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/satellite/Duplex.hpp>
#include <lpl/satellite/Parity.hpp>
#include <lpl/satellite/PowerState.hpp>
#include <lpl/satellite/Protocol.hpp>
#include <lpl/satellite/VoiceActivity.hpp>
#include <lpl/satellite/WakeWord.hpp>

#include <cstdio>

namespace {

int gFailures = 0;
int gChecks = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++gFailures;
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== satellite: one protocol, three machines ==\n");

    // ── The wire format ──────────────────────────────────────────────────────
    std::printf("\n-- a header, not a tag --\n");
    {
        core::u8 buffer[satellite::kHeaderBytes + satellite::kFramePayloadBytes];
        core::i16 samples[satellite::kFrameSamples] = {};
        for (core::u32 i = 0u; i < satellite::kFrameSamples; ++i)
            samples[i] = static_cast<core::i16>(i * 37u);

        const core::u32 written =
            satellite::encode(satellite::Datagram::Audio, 7u, reinterpret_cast<const core::u8 *>(samples),
                              satellite::kFramePayloadBytes, buffer, sizeof(buffer));
        check(written == satellite::kHeaderBytes + satellite::kFramePayloadBytes, "an audio datagram encodes whole");

        satellite::Frame decoded{};
        check(satellite::decode(buffer, written, decoded), "and decodes");
        check(decoded.kind == satellite::Datagram::Audio && decoded.sequence == 7u,
              "with its kind and sequence intact");
        check(decoded.payloadBytes == satellite::kFramePayloadBytes, "and its whole payload");

        bool identical = decoded.payload != nullptr;
        for (core::u32 i = 0u; identical && i < satellite::kFramePayloadBytes; ++i)
            identical = decoded.payload[i] == reinterpret_cast<const core::u8 *>(samples)[i];
        check(identical, "byte for byte");

        // The defect the header removes. Under the tag-based format these four bytes
        // ARE the discriminator, so a loud frame that happened to start with them was
        // printed as text and never played — at random, silently.
        core::u8 payload[satellite::kFramePayloadBytes];
        for (core::u32 i = 0u; i < satellite::kFramePayloadBytes; ++i)
            payload[i] = static_cast<core::u8>(i & 0xFFu);
        payload[0] = 'T';
        payload[1] = 'X';
        payload[2] = 'T';
        payload[3] = ':';
        const core::u32 tagged = satellite::encode(satellite::Datagram::Audio, 0u, payload,
                                                   satellite::kFramePayloadBytes, buffer, sizeof(buffer));
        satellite::Frame taggedFrame{};
        check(satellite::decode(buffer, tagged, taggedFrame) && taggedFrame.kind == satellite::Datagram::Audio,
              "audio whose first bytes spell \"TXT:\" is still audio");

        // And the other direction: something that is not ours is refused rather than
        // guessed at. A satellite port receives stray traffic.
        const core::u8 stray[8] = {'H', 'T', 'T', 'P', '/', '1', '.', '1'};
        satellite::Frame strayFrame{};
        check(!satellite::decode(stray, sizeof(stray), strayFrame), "a stray datagram is refused, not guessed at");

        const core::u8 truncated[satellite::kHeaderBytes + 3u] = {satellite::kMagicFirst, satellite::kMagicSecond,
                                                                  1u, 0u, 1u, 2u, 3u};
        satellite::Frame oddFrame{};
        check(!satellite::decode(truncated, sizeof(truncated), oddFrame),
              "an audio payload cut mid-sample is refused rather than clicked through");

        check(satellite::missedBetween(254u, 3u) == 4u, "sequence gaps count across the wrap");
        check(satellite::missedBetween(9u, 10u) == 0u, "and the expected successor loses nothing");
    }

    // ── Voice activity ───────────────────────────────────────────────────────
    std::printf("\n-- hysteresis, and why there are two thresholds --\n");
    {
        satellite::VoiceActivityParams params{};
        satellite::VoiceActivity detector{params};

        core::i16 quiet[satellite::kFrameSamples] = {};
        core::i16 loud[satellite::kFrameSamples];
        for (core::u32 i = 0u; i < satellite::kFrameSamples; ++i)
            loud[i] = (i % 8u < 4u) ? 8000 : -8000;

        check(satellite::frameLevel(quiet, satellite::kFrameSamples) == math::Fixed32::zero(),
              "silence measures zero");
        check(satellite::frameLevel(loud, satellite::kFrameSamples) > params.startLevel,
              "a loud frame is above the start level");

        check(detector.observe(quiet, satellite::kFrameSamples, false) == satellite::VoiceEvent::Silence,
              "silence starts nothing");
        check(detector.observe(loud, satellite::kFrameSamples, false) == satellite::VoiceEvent::UtteranceBegan,
              "a loud frame opens an utterance");

        // A pause inside a sentence must not close it. That is what the gap between
        // the two thresholds is for, and it is the whole reason there are two.
        core::u32 pauses = 0u;
        while (detector.observe(quiet, satellite::kFrameSamples, false) == satellite::VoiceEvent::Speaking)
            ++pauses;
        const core::u32 expected = params.hangoverMilliseconds / satellite::kFrameMilliseconds;
        std::printf("    the utterance survived %u silent frames, hangover allows %u\n", pauses, expected);
        check(pauses >= expected - 1u && pauses <= expected + 1u,
              "the utterance closes after the hangover and not before");
        check(detector.utterances() == 1u, "and exactly one utterance was counted");

        // While the node's own speaker is on, both levels rise: only a nearby voice
        // gets through. Same frame, different answer.
        satellite::VoiceActivity guarded{params};
        core::i16 middling[satellite::kFrameSamples];
        for (core::u32 i = 0u; i < satellite::kFrameSamples; ++i)
            middling[i] = (i % 8u < 4u) ? 1200 : -1200;
        check(guarded.observe(middling, satellite::kFrameSamples, false) == satellite::VoiceEvent::UtteranceBegan,
              "a middling frame opens an utterance in a quiet room");
        satellite::VoiceActivity boosted{params};
        check(boosted.observe(middling, satellite::kFrameSamples, true) == satellite::VoiceEvent::Silence,
              "the same frame does not, while the node is speaking");
    }

    // ── Features, the wake word, and the echo ────────────────────────────────
    std::printf("\n-- one feature extractor, three consumers --\n");
    {
        core::i16 high[satellite::kFrameSamples];
        core::i16 low[satellite::kFrameSamples];
        core::i16 silent[satellite::kFrameSamples] = {};
        for (core::u32 i = 0u; i < satellite::kFrameSamples; ++i)
        {
            high[i] = (i % 2u == 0u) ? 8000 : -8000;
            low[i] = (i % 128u < 64u) ? 8000 : -8000;
        }

        satellite::FeatureFrame highShape{};
        satellite::FeatureFrame lowShape{};
        satellite::FeatureFrame silentShape{};
        satellite::extractFeatures(high, satellite::kFrameSamples, highShape);
        satellite::extractFeatures(low, satellite::kFrameSamples, lowShape);
        satellite::extractFeatures(silent, satellite::kFrameSamples, silentShape);

        core::u32 total = 0u;
        for (core::u32 b = 0u; b < satellite::kBandCount; ++b)
            total += highShape.bands[b];
        check(total == satellite::kBandScale, "the bands sum to exactly the scale, with no rounding drift");

        std::printf("    high tone %u/%u/%u/%u\n", highShape.bands[0], highShape.bands[1], highShape.bands[2],
                    highShape.bands[3]);
        std::printf("    low tone  %u/%u/%u/%u\n", lowShape.bands[0], lowShape.bands[1], lowShape.bands[2],
                    lowShape.bands[3]);
        check(highShape.bands[0] > lowShape.bands[0], "a high tone puts its energy in the top octave");
        check(lowShape.bands[3] > highShape.bands[3], "and a low tone in the residual");
        check(satellite::shapeDistance(highShape, lowShape) > 512u, "the two are far apart");

        // Level-independence is the property that makes a template usable from
        // across a room rather than only into the microphone.
        core::i16 faint[satellite::kFrameSamples];
        for (core::u32 i = 0u; i < satellite::kFrameSamples; ++i)
            faint[i] = static_cast<core::i16>(high[i] / 16);
        satellite::FeatureFrame faintShape{};
        satellite::extractFeatures(faint, satellite::kFrameSamples, faintShape);
        const core::u32 levelDrift = satellite::shapeDistance(highShape, faintShape);
        std::printf("    the same tone sixteen times quieter is %u away\n", levelDrift);
        check(levelDrift < 32u, "the shape survives a sixteenfold drop in level");

        check(satellite::shapeDistance(silentShape, highShape) > 512u,
              "silence has no shape, and so matches nothing");
    }

    // ── The canonical exchange ───────────────────────────────────────────────
    std::printf("\n-- the exchange the kernel must reproduce --\n");
    satellite::SatelliteFoldResult folded{};
    satellite::foldSatelliteState(folded);

    check(folded.detections == 1u, "the wake word fired exactly once");
    check(folded.wakeFrame == satellite::parityWakeFirstFrame() + satellite::parityWakeFrames() - 1u,
          "on the frame that completed it, not before and not after");
    check(folded.wakeDistance <= satellite::parityWakeTolerance(), "within tolerance");

    // The claim that matters, and the one a gate checking only the match would miss:
    // ordinary speech must be FAR. A detector that matched everything would satisfy
    // every check above.
    std::printf("    the word matched at %u; broadband speech sits at %u, tolerance is %u\n", folded.wakeDistance,
                folded.speechDistance, satellite::parityWakeTolerance());
    check(folded.speechDistance > satellite::parityWakeTolerance() * 4u,
          "and ordinary speech is nowhere near it");

    check(folded.utterances == 1u, "one utterance was opened and closed");
    check(folded.framesEmitted > 0u, "datagrams left the node");
    check(folded.echoesRejected > 0u, "and the node recognised its own voice coming back");
    check(folded.taggedAudioIsAudio == 1u, "audio is audio whatever its first four bytes spell");

    check(folded.idlePermille > 0u && folded.idlePermille < 1000u,
          "the node was idle for part of the timeline and busy for the rest");
    // In microseconds the two are exactly complementary — "awake" is defined as
    // "not idle". In thousandths they are two independent floors of complementary
    // fractions, so each may lose a unit and the sum lands on 999 rather than 1000.
    // Asserting the exact identity here was wrong: it demanded of the reporting a
    // precision the reporting does not claim.
    const core::u32 accounted = folded.dutyPermille + folded.idlePermille;
    std::printf("    idle %u‰ + awake %u‰ = %u‰\n", folded.idlePermille, folded.dutyPermille, accounted);
    check(accounted >= 999u && accounted <= 1000u,
          "awake and idle account for the whole timeline, up to the rounding of two floors");

    satellite::SatelliteFoldResult again{};
    satellite::foldSatelliteState(again);
    check(again.featureSignature == folded.featureSignature && again.wireSignature == folded.wireSignature &&
              again.stateSignature == folded.stateSignature,
          "running the exchange twice decides the same things");

    std::printf("\n-- signatures the kernel must reproduce --\n");
    std::printf("  feature_sig  = 0x%08X\n", folded.featureSignature);
    std::printf("  level_sig    = 0x%08X\n", folded.levelSignature);
    std::printf("  event_sig    = 0x%08X\n", folded.eventSignature);
    std::printf("  wire_sig     = 0x%08X\n", folded.wireSignature);
    std::printf("  state_sig    = 0x%08X\n", folded.stateSignature);
    std::printf("  template_sig = 0x%08X\n", folded.templateSignature);
    std::printf("  emitted      = %u\n", folded.framesEmitted);
    std::printf("  utterances   = %u\n", folded.utterances);
    std::printf("  detections   = %u\n", folded.detections);
    std::printf("  wake_frame   = %u\n", folded.wakeFrame);
    std::printf("  wake_dist    = %u\n", folded.wakeDistance);
    std::printf("  speech_dist  = %u\n", folded.speechDistance);
    std::printf("  echoes       = %u\n", folded.echoesRejected);
    std::printf("  transitions  = %u\n", folded.transitions);
    std::printf("  idle_permille= %u\n", folded.idlePermille);
    std::printf("  duty_permille= %u\n", folded.dutyPermille);
    std::printf("  tagged_audio = %u\n", folded.taggedAudioIsAudio);

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
