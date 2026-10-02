/**
 * @file Parity.cpp
 * @brief The canonical satellite exchange, folded stage by stage.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/satellite/Parity.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/math/Random.hpp>
#    include <lpl/satellite/Duplex.hpp>
#    include <lpl/satellite/PowerState.hpp>
#    include <lpl/satellite/VoiceActivity.hpp>
#    include <lpl/satellite/WakeWord.hpp>

namespace lpl::satellite {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Folds one word into a running FNV-1a hash.
 * @param hash Running value.
 * @param word Word to absorb.
 */
void foldWord(core::u32 &hash, core::u32 word) noexcept { hash = (hash ^ word) * kFnv1aPrime; }

/// Amplitude of everything that is not silence, comfortably above the start level.
constexpr core::i16 kLoud = 8000;

/// First frame of the spoken utterance, right after the wake word.
constexpr core::u32 kSpeechFirstFrame = 16u;

/// One past the last frame of the utterance.
constexpr core::u32 kSpeechEndFrame = 30u;

/// First frame of the reply playing back into the room.
constexpr core::u32 kPlaybackFirstFrame = 48u;

/// One past the last frame of the reply.
constexpr core::u32 kPlaybackEndFrame = 64u;

/**
 * @brief A square wave of a given period, in samples.
 *
 * Square rather than sine for the reason the whole module is integer: a sine needs a
 * transcendental, and what this timeline has to produce is not a pleasant tone but
 * two sounds with reliably DIFFERENT spectral shapes. A square wave's period places
 * its energy in a known octave, which is exactly what the Haar cascade measures.
 *
 * @param index  Sample position.
 * @param period Samples per cycle.
 * @return The sample.
 */
core::i16 square(core::u32 index, core::u32 period) noexcept
{
    return ((index % period) < (period / 2u)) ? kLoud : static_cast<core::i16>(-kLoud);
}

} // namespace

bool parityFrameIsPlayback(core::u32 frame) noexcept
{
    return frame >= kPlaybackFirstFrame && frame < kPlaybackEndFrame;
}

void synthesiseFrame(core::u32 frame, core::i16 *out, core::u32 count) noexcept
{
    if (out == nullptr)
        return;

    const core::u32 wakeEnd = parityWakeFirstFrame() + parityWakeFrames();

    for (core::u32 i = 0u; i < count; ++i)
        out[i] = 0;

    if (frame >= parityWakeFirstFrame() && frame < wakeEnd)
    {
        // The wake word: a fixed alternation of two octaves, so its shape is the
        // same every time it is uttered — which is what a template is.
        const core::u32 period = ((frame - parityWakeFirstFrame()) % 2u == 0u) ? 4u : 32u;
        for (core::u32 i = 0u; i < count; ++i)
            out[i] = square(i, period);
        return;
    }

    if (frame >= kSpeechFirstFrame && frame < kSpeechEndFrame)
    {
        // The utterance: broadband noise. Its shape must be far from the wake word's
        // or the gate would fire on ordinary speech, which the fold asserts.
        math::Random stream = math::deriveStream(0x5A7Eu, frame);
        for (core::u32 i = 0u; i < count; ++i)
            out[i] = static_cast<core::i16>(static_cast<core::i32>(stream.below(2u * kLoud)) - kLoud);
        return;
    }

    if (parityFrameIsPlayback(frame))
    {
        // The reply, in a third octave. The node plays this AND hears it, which is
        // what the duplex rejection has to survive.
        for (core::u32 i = 0u; i < count; ++i)
            out[i] = square(i, 12u);
        return;
    }
}

void foldSatelliteState(SatelliteFoldResult &out)
{
    out = SatelliteFoldResult{};

    core::i16 frame[kFrameSamples];
    core::u8 datagram[kHeaderBytes + kFramePayloadBytes];

    // ── Arm the wake word from its own utterance ─────────────────────────────
    // The template is RECORDED from the timeline rather than written out as a table
    // of numbers. A table would be a second statement of what the word sounds like,
    // and the day the synthesiser changed the two would part without either being
    // wrong on its own.
    FeatureFrame templateFrames[kMaxTemplateFrames];
    core::u32 templateHash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < parityWakeFrames(); ++i)
    {
        synthesiseFrame(parityWakeFirstFrame() + i, frame, kFrameSamples);
        extractFeatures(frame, kFrameSamples, templateFrames[i]);
        for (core::u32 b = 0u; b < kBandCount; ++b)
            foldWord(templateHash, templateFrames[i].bands[b]);
    }
    out.templateSignature = templateHash;

    WakeWord gate;
    if (!gate.arm(templateFrames, parityWakeFrames(), parityWakeTolerance()))
        return;

    VoiceActivity detector{VoiceActivityParams{}};
    Duplex duplex{parityEchoTolerance()};
    PowerState power;

    core::u32 featureHash = kFnv1aOffsetBasis;
    core::u32 levelHash = kFnv1aOffsetBasis;
    core::u32 eventHash = kFnv1aOffsetBasis;
    core::u32 wireHash = kFnv1aOffsetBasis;
    core::u32 stateHash = kFnv1aOffsetBasis;

    out.wakeFrame = parityFrameCount();
    out.speechDistance = kBandScale * 2u;
    core::u8 sequence = 0u;
    bool streaming = false;

    for (core::u32 index = 0u; index < parityFrameCount(); ++index)
    {
        synthesiseFrame(index, frame, kFrameSamples);
        const bool playing = parityFrameIsPlayback(index);

        // The node feeds its own playback to the duplex detector BEFORE listening,
        // because that is the order the hardware imposes: samples go to the codec,
        // then come back through the microphone one buffer later at the earliest.
        if (playing)
            duplex.notePlayed(frame, kFrameSamples);
        else
            duplex.clear();

        FeatureFrame shape{};
        extractFeatures(frame, kFrameSamples, shape);
        for (core::u32 b = 0u; b < kBandCount; ++b)
            foldWord(featureHash, shape.bands[b]);

        const bool echo = playing && duplex.isEcho(frame, kFrameSamples);

        const VoiceEvent event = detector.observe(frame, kFrameSamples, playing);
        foldWord(levelHash, static_cast<core::u32>(detector.lastLevel().raw()));
        foldWord(eventHash, static_cast<core::u32>(event));
        foldWord(eventHash, echo ? 1u : 0u);

        // An echo is not somebody speaking, so it must not move the state machine.
        if (!echo)
            power.onVoice(event);

        if (gate.observe(frame, kFrameSamples))
        {
            if (out.wakeFrame == parityFrameCount())
            {
                out.wakeFrame = index;
                out.wakeDistance = gate.matchDistance();
            }
            power.onWakeWord();
            streaming = true;
        }

        // How far ordinary speech sits from the template. Recorded rather than
        // asserted here: a gate that only checked the word MATCHED would pass on a
        // detector that matches everything.
        //
        // Frames where the window is not yet full are SKIPPED, not minimised over.
        // They report kNoMeasurement, and taking a minimum across the absence of a
        // measurement is how the first version of this reported a distance of zero
        // for broadband noise — a number that looked like a catastrophic false match
        // and was in fact no reading at all.
        if (index >= kSpeechFirstFrame && index < kSpeechEndFrame &&
            gate.lastDistance() != WakeWord::kNoMeasurement && gate.lastDistance() < out.speechDistance)
            out.speechDistance = gate.lastDistance();

        if (playing)
            power.onPlaybackBegan();
        else if (index == kPlaybackEndFrame)
            power.onPlaybackEnded();

        if (streaming && !echo && (event == VoiceEvent::UtteranceBegan || event == VoiceEvent::Speaking))
        {
            const core::u32 written = encode(Datagram::Audio, sequence, reinterpret_cast<const core::u8 *>(frame),
                                             kFramePayloadBytes, datagram, sizeof(datagram));
            wireHash = foldDatagram(wireHash, datagram, written);
            ++sequence;
            ++out.framesEmitted;
        }
        if (streaming && event == VoiceEvent::UtteranceEnded)
        {
            const core::u32 written = encode(Datagram::EndOfUtterance, sequence, nullptr, 0u, datagram,
                                             sizeof(datagram));
            wireHash = foldDatagram(wireHash, datagram, written);
            ++sequence;
            streaming = false;
        }

        // Awake exactly when there is something to do. Idle frames are the ones the
        // processor slept through, which is what makes the duty cycle a measurement
        // of this timeline rather than of the machine that ran it.
        const bool awake = power.state() != NodeState::Idle;
        power.advance(kFrameMilliseconds * 1000u, awake);
        foldWord(stateHash, static_cast<core::u32>(power.state()));
    }

    // ── The ambiguity the header exists to remove ────────────────────────────
    // An audio payload whose first four bytes spell "TXT:" — two perfectly ordinary
    // loud samples — must still decode as audio. Under the untagged format it did
    // not, and nothing reported it.
    {
        core::u8 payload[kFramePayloadBytes];
        for (core::u32 i = 0u; i < kFramePayloadBytes; ++i)
            payload[i] = static_cast<core::u8>(i & 0xFFu);
        payload[0] = 'T';
        payload[1] = 'X';
        payload[2] = 'T';
        payload[3] = ':';

        const core::u32 written = encode(Datagram::Audio, 0u, payload, kFramePayloadBytes, datagram,
                                         sizeof(datagram));
        Frame decoded{};
        out.taggedAudioIsAudio = (decode(datagram, written, decoded) && decoded.kind == Datagram::Audio) ? 1u : 0u;
    }

    out.featureSignature = featureHash;
    out.levelSignature = levelHash;
    out.eventSignature = eventHash;
    out.wireSignature = wireHash;
    out.stateSignature = stateHash;
    out.utterances = detector.utterances();
    out.detections = gate.detections();
    out.echoesRejected = duplex.rejected();
    out.transitions = power.transitions();
    out.dutyPermille = power.dutyCyclePermille();

    const core::u64 elapsed = power.elapsedMicroseconds();
    out.idlePermille =
        elapsed == 0u ? 0u : static_cast<core::u32>((power.microsecondsIn(NodeState::Idle) * 1000u) / elapsed);
}

} // namespace lpl::satellite

#endif // LPL_HAS_FOUNDATION
