#include <lpl/satellite/Duplex.hpp>
#include <lpl/satellite/Parity.hpp>
#include <lpl/satellite/PowerState.hpp>
#include <lpl/satellite/Protocol.hpp>
#include <lpl/satellite/VoiceActivity.hpp>
#include <lpl/satellite/WakeWord.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(satellite);

namespace {

/**
 * @brief Fills @p samples with a square wave of @p amplitude whose half period is @p halfPeriod samples.
 */
void squareWave(lpl::core::i16 (&samples)[lpl::satellite::kFrameSamples], lpl::core::i16 amplitude,
                lpl::core::u32 halfPeriod)
{
    for (lpl::core::u32 index = 0u; index < lpl::satellite::kFrameSamples; ++index)
        samples[index] = (index % (2u * halfPeriod) < halfPeriod) ? amplitude : static_cast<lpl::core::i16>(-amplitude);
}

} // namespace

/**
 * @brief The wire format: an audio datagram encodes and decodes whole, its kind is a header field
 *        rather than a tag its payload could imitate, and stray or truncated traffic is refused
 *        rather than guessed at.
 */
LPL_TEST(a_datagram_is_a_header_not_a_tag)
{
    lpl::core::u8 buffer[lpl::satellite::kHeaderBytes + lpl::satellite::kFramePayloadBytes];
    lpl::core::i16 samples[lpl::satellite::kFrameSamples] = {};

    for (lpl::core::u32 index = 0u; index < lpl::satellite::kFrameSamples; ++index)
        samples[index] = static_cast<lpl::core::i16>(index * 37u);

    const lpl::core::u8 *const sampleBytes = reinterpret_cast<const lpl::core::u8 *>(samples);
    const lpl::core::u32 written = lpl::satellite::encode(lpl::satellite::Datagram::Audio, 7u, sampleBytes,
                                                          lpl::satellite::kFramePayloadBytes, buffer, sizeof(buffer));

    test.check(written == lpl::satellite::kHeaderBytes + lpl::satellite::kFramePayloadBytes,
               "an audio datagram encodes whole");

    lpl::satellite::Frame decoded{};

    test.check(lpl::satellite::decode(buffer, written, decoded), "and decodes");
    test.check(decoded.kind == lpl::satellite::Datagram::Audio && decoded.sequence == 7u,
               "with its kind and sequence intact");
    test.check(decoded.payloadBytes == lpl::satellite::kFramePayloadBytes, "and its whole payload");

    bool identical = (decoded.payload != nullptr);

    for (lpl::core::u32 index = 0u; identical && index < lpl::satellite::kFramePayloadBytes; ++index)
        identical = (decoded.payload[index] == sampleBytes[index]);
    test.check(identical, "byte for byte");

    lpl::core::u8 payload[lpl::satellite::kFramePayloadBytes];

    for (lpl::core::u32 index = 0u; index < lpl::satellite::kFramePayloadBytes; ++index)
        payload[index] = static_cast<lpl::core::u8>(index & 0xFFu);
    payload[0] = 'T';
    payload[1] = 'X';
    payload[2] = 'T';
    payload[3] = ':';

    const lpl::core::u32 tagged = lpl::satellite::encode(lpl::satellite::Datagram::Audio, 0u, payload,
                                                         lpl::satellite::kFramePayloadBytes, buffer, sizeof(buffer));
    lpl::satellite::Frame taggedFrame{};

    test.check(lpl::satellite::decode(buffer, tagged, taggedFrame) &&
                   taggedFrame.kind == lpl::satellite::Datagram::Audio,
               "audio whose first bytes spell \"TXT:\" is still audio");

    const lpl::core::u8 stray[8] = {'H', 'T', 'T', 'P', '/', '1', '.', '1'};
    lpl::satellite::Frame strayFrame{};

    test.check(!lpl::satellite::decode(stray, sizeof(stray), strayFrame) &&
                   strayFrame.refusal == lpl::satellite::Refusal::NotSatellite,
               "a stray datagram is refused, not guessed at");

    const lpl::core::u8 truncated[lpl::satellite::kHeaderBytes + 3u] = {lpl::satellite::kMagicFirst,
                                                                        lpl::satellite::kMagicSecond,
                                                                        lpl::satellite::kProtocolVersion,
                                                                        1u,
                                                                        0u,
                                                                        1u,
                                                                        2u,
                                                                        3u};
    lpl::satellite::Frame truncatedFrame{};

    test.check(!lpl::satellite::decode(truncated, sizeof(truncated), truncatedFrame) &&
                   truncatedFrame.refusal == lpl::satellite::Refusal::PartialSample,
               "an audio payload cut mid-sample is refused rather than clicked through");

    const lpl::core::u8 headerCut[] = {lpl::satellite::kMagicFirst, lpl::satellite::kMagicSecond,
                                       lpl::satellite::kProtocolVersion, 1u};
    lpl::satellite::Frame headerCutFrame{};

    test.check(!lpl::satellite::decode(headerCut, sizeof(headerCut), headerCutFrame) &&
                   headerCutFrame.refusal == lpl::satellite::Refusal::ShortHeader,
               "a header cut short is refused as such, not taken for someone else's traffic");

    const lpl::core::u8 unknownKind[] = {lpl::satellite::kMagicFirst, lpl::satellite::kMagicSecond,
                                         lpl::satellite::kProtocolVersion, 0x2Au, 0u};
    lpl::satellite::Frame unknownKindFrame{};

    test.check(!lpl::satellite::decode(unknownKind, sizeof(unknownKind), unknownKindFrame) &&
                   unknownKindFrame.refusal == lpl::satellite::Refusal::UnknownKind,
               "a kind the protocol does not define is refused as such");
    test.check(lpl::satellite::missedBetween(254u, 3u) == 4u, "sequence gaps count across the wrap");
    test.check(lpl::satellite::missedBetween(9u, 10u) == 0u, "and the expected successor loses nothing");
}

/**
 * @brief A reader speaks one version of the format: a datagram of another version, or of the
 *        unversioned format before it, is refused, and the refusal names the version it got.
 */
LPL_TEST(an_unknown_version_is_refused_by_name)
{
    lpl::core::u8 buffer[lpl::satellite::kHeaderBytes];
    const lpl::core::u32 written =
        lpl::satellite::encode(lpl::satellite::Datagram::EndOfUtterance, 9u, nullptr, 0u, buffer, sizeof(buffer));
    lpl::satellite::Frame current{};

    test.check(written == lpl::satellite::kHeaderBytes && buffer[2] == lpl::satellite::kProtocolVersion,
               "a node writes its version right after the magic");
    test.check(lpl::satellite::decode(buffer, written, current) &&
                   current.version == lpl::satellite::kProtocolVersion &&
                   current.refusal == lpl::satellite::Refusal::None,
               "and a reader of the same version accepts it");

    const lpl::core::u8 foreign[] = {lpl::satellite::kMagicFirst, lpl::satellite::kMagicSecond, 0x2Au,
                                     static_cast<lpl::core::u8>(lpl::satellite::Datagram::EndOfUtterance), 9u};
    lpl::satellite::Frame foreignFrame{};

    test.check(!lpl::satellite::decode(foreign, sizeof(foreign), foreignFrame),
               "a datagram of a version the reader does not know is refused");
    test.check(foreignFrame.refusal == lpl::satellite::Refusal::UnknownVersion && foreignFrame.version == 0x2Au,
               "and the refusal names the version it got");

    const lpl::core::u8 unversioned[] = {lpl::satellite::kMagicFirst,
                                         lpl::satellite::kMagicSecond,
                                         static_cast<lpl::core::u8>(lpl::satellite::Datagram::Audio),
                                         3u,
                                         0x10u,
                                         0x20u};
    lpl::satellite::Frame unversionedFrame{};

    test.check(!lpl::satellite::decode(unversioned, sizeof(unversioned), unversionedFrame) &&
                   unversionedFrame.refusal == lpl::satellite::Refusal::UnknownVersion &&
                   unversionedFrame.version == static_cast<lpl::core::u8>(lpl::satellite::Datagram::Audio),
               "a datagram from before the version is refused, its kind named as the version it got");

    bool everyUnversionedHeaderIsNamed = true;

    for (lpl::core::u8 unversionedKind = 1u; everyUnversionedHeaderIsNamed && unversionedKind <= 6u; ++unversionedKind)
    {
        const lpl::core::u8 unversionedHeader[] = {lpl::satellite::kMagicFirst, lpl::satellite::kMagicSecond,
                                                   unversionedKind, 9u};
        lpl::satellite::Frame unversionedHeaderFrame{};

        everyUnversionedHeaderIsNamed =
            !lpl::satellite::decode(unversionedHeader, sizeof(unversionedHeader), unversionedHeaderFrame) &&
            unversionedHeaderFrame.refusal == lpl::satellite::Refusal::UnknownVersion &&
            unversionedHeaderFrame.version == unversionedKind;
    }
    test.check(everyUnversionedHeaderIsNamed,
               "every four-byte header of the unversioned format is refused by the version it seems to carry");
}

/**
 * @brief Voice activity has two thresholds: a loud frame opens an utterance, which survives the
 *        pauses of its hangover and no more, and a middling frame that opens one in a quiet room
 *        does not while the node is speaking.
 */
LPL_TEST(two_thresholds_make_one_utterance)
{
    lpl::satellite::VoiceActivityParams params{};
    lpl::satellite::VoiceActivity detector{params};
    lpl::core::i16 quiet[lpl::satellite::kFrameSamples] = {};
    lpl::core::i16 loud[lpl::satellite::kFrameSamples];

    squareWave(loud, 8000, 4u);
    test.check(lpl::satellite::frameLevel(quiet, lpl::satellite::kFrameSamples) == lpl::math::Fixed32::zero(),
               "silence measures zero");
    test.check(lpl::satellite::frameLevel(loud, lpl::satellite::kFrameSamples) > params.startLevel,
               "a loud frame is above the start level");
    test.check(detector.observe(quiet, lpl::satellite::kFrameSamples, false) == lpl::satellite::VoiceEvent::Silence,
               "silence starts nothing");
    test.check(detector.observe(loud, lpl::satellite::kFrameSamples, false) ==
                   lpl::satellite::VoiceEvent::UtteranceBegan,
               "a loud frame opens an utterance");

    lpl::core::u32 pauses = 0u;

    while (detector.observe(quiet, lpl::satellite::kFrameSamples, false) == lpl::satellite::VoiceEvent::Speaking)
        ++pauses;

    const lpl::core::u32 hangoverFrames = params.hangoverMilliseconds / lpl::satellite::kFrameMilliseconds;

    test.check(pauses + 1u >= hangoverFrames && pauses <= hangoverFrames + 1u,
               "the utterance closes after the hangover and not before");
    test.check(detector.utterances() == 1u, "and exactly one utterance was counted");
    test.measure("silent_frames_survived", pauses);

    lpl::core::i16 middling[lpl::satellite::kFrameSamples];
    lpl::satellite::VoiceActivity quietRoom{params};
    lpl::satellite::VoiceActivity speakingNode{params};

    squareWave(middling, 1200, 4u);
    test.check(quietRoom.observe(middling, lpl::satellite::kFrameSamples, false) ==
                   lpl::satellite::VoiceEvent::UtteranceBegan,
               "a middling frame opens an utterance in a quiet room");
    test.check(speakingNode.observe(middling, lpl::satellite::kFrameSamples, true) ==
                   lpl::satellite::VoiceEvent::Silence,
               "the same frame does not, while the node is speaking");
}

/**
 * @brief One feature extractor serves the hosted node, the kernel and a firmware: its bands sum to
 *        the scale exactly, tell a high tone from a low one, survive a sixteenfold drop in level,
 *        and give silence no shape to match.
 */
LPL_TEST(one_feature_extractor_serves_three_machines)
{
    lpl::core::i16 high[lpl::satellite::kFrameSamples];
    lpl::core::i16 low[lpl::satellite::kFrameSamples];
    lpl::core::i16 faint[lpl::satellite::kFrameSamples];
    lpl::core::i16 silent[lpl::satellite::kFrameSamples] = {};

    squareWave(high, 8000, 1u);
    squareWave(low, 8000, 64u);
    for (lpl::core::u32 index = 0u; index < lpl::satellite::kFrameSamples; ++index)
        faint[index] = static_cast<lpl::core::i16>(high[index] / 16);

    lpl::satellite::FeatureFrame highShape{};
    lpl::satellite::FeatureFrame lowShape{};
    lpl::satellite::FeatureFrame faintShape{};
    lpl::satellite::FeatureFrame silentShape{};

    lpl::satellite::extractFeatures(high, lpl::satellite::kFrameSamples, highShape);
    lpl::satellite::extractFeatures(low, lpl::satellite::kFrameSamples, lowShape);
    lpl::satellite::extractFeatures(faint, lpl::satellite::kFrameSamples, faintShape);
    lpl::satellite::extractFeatures(silent, lpl::satellite::kFrameSamples, silentShape);

    lpl::core::u32 total = 0u;

    for (lpl::core::u32 band = 0u; band < lpl::satellite::kBandCount; ++band)
        total += highShape.bands[band];
    test.check(total == lpl::satellite::kBandScale, "the bands sum to exactly the scale, with no rounding drift");
    test.check(highShape.bands[0] > lowShape.bands[0], "a high tone puts its energy in the top octave");
    test.check(lowShape.bands[3] > highShape.bands[3], "and a low tone in the residual");
    test.check(lpl::satellite::shapeDistance(highShape, lowShape) > 512u, "the two are far apart");

    const lpl::core::u32 levelDrift = lpl::satellite::shapeDistance(highShape, faintShape);

    test.check(levelDrift < 32u, "the shape survives a sixteenfold drop in level");
    test.check(lpl::satellite::shapeDistance(silentShape, highShape) > 512u,
               "silence has no shape, and so matches nothing");
    test.measure("level_drift", levelDrift);
}

/**
 * @brief Gate P15 satellite: a hosted node, the kernel and a firmware share no instruction set and
 *        still decide the same exchange: when to start sending, when to stop, whether the word was
 *        heard, whether the node is hearing itself, and which datagrams are of a version it does not
 *        speak.
 */
LPL_TEST(three_machines_decide_the_same_exchange)
{
    lpl::satellite::SatelliteFoldResult folded{};

    lpl::satellite::foldSatelliteState(folded);
    test.check(folded.detections == 1u, "the wake word fired exactly once");
    test.check(folded.wakeFrame == lpl::satellite::parityWakeFirstFrame() + lpl::satellite::parityWakeFrames() - 1u,
               "on the frame that completed it, not before and not after");
    test.check(folded.wakeDistance <= lpl::satellite::parityWakeTolerance(), "within tolerance");
    test.check(folded.speechDistance > lpl::satellite::parityWakeTolerance() * 4u,
               "and ordinary speech is nowhere near it");
    test.check(folded.utterances == 1u, "one utterance was opened and closed");
    test.check(folded.framesEmitted > 0u, "datagrams left the node");
    test.check(folded.echoesRejected > 0u, "and the node recognised its own voice coming back");
    test.check(folded.taggedAudioIsAudio == 1u, "audio is audio whatever its first four bytes spell");
    test.check(folded.refusedVersion == lpl::satellite::kProtocolVersion + 1u,
               "a datagram one version ahead is refused, by its version");
    test.check(folded.idlePermille > 0u && folded.idlePermille < 1000u,
               "the node was idle for part of the timeline and busy for the rest");

    const lpl::core::u32 accounted = folded.dutyPermille + folded.idlePermille;

    test.check(accounted >= 999u && accounted <= 1000u,
               "awake and idle account for the whole timeline, up to the rounding of two floors");

    lpl::satellite::SatelliteFoldResult again{};

    lpl::satellite::foldSatelliteState(again);
    test.check(again.featureSignature == folded.featureSignature && again.wireSignature == folded.wireSignature &&
                   again.stateSignature == folded.stateSignature,
               "running the exchange twice decides the same things");

    test.measureHexadecimal("feature_signature", folded.featureSignature);
    test.measureHexadecimal("level_signature", folded.levelSignature);
    test.measureHexadecimal("event_signature", folded.eventSignature);
    test.measureHexadecimal("wire_signature", folded.wireSignature);
    test.measureHexadecimal("state_signature", folded.stateSignature);
    test.measureHexadecimal("template_signature", folded.templateSignature);
    test.measure("frames_emitted", folded.framesEmitted);
    test.measure("wake_frame", folded.wakeFrame);
    test.measure("wake_distance", folded.wakeDistance);
    test.measure("speech_distance", folded.speechDistance);
    test.measure("echoes_rejected", folded.echoesRejected);
    test.measure("transitions", folded.transitions);
    test.measure("idle_permille", folded.idlePermille);
    test.measure("duty_permille", folded.dutyPermille);
    test.measure("refused_version", folded.refusedVersion);
}
