/**
 * @file Protocol.hpp
 * @brief The satellite wire format, in one place.
 *
 * Raw PCM16 mono at 16 kHz, an `END` datagram to close an utterance, and the
 * return channel interleaving `TXT:` segments, audio frames, then `AEND` — or
 * `STOP` when the sovereign cut in. Written once and shared, because a hosted
 * satellite, a bare-metal one and a microcontroller must not each hold their own
 * opinion about what closes an utterance.
 *
 * ⚠ The format described above is the one the hosted node and the server speak
 * TODAY, and it has a defect that this module exists to remove. Its datagrams are
 * discriminated by their first bytes — a datagram beginning with `TXT:` is a
 * transcript, anything else is audio — and a PCM16 frame is arbitrary bytes. The
 * four bytes `T` `X` `T` `:` are 0x54 0x58 0x54 0x3A, which read as the little-endian
 * samples 22612 and 14932: two perfectly ordinary loud samples. A sufficiently loud
 * reply therefore gets printed as text and never played, at random, and nothing
 * reports an error.
 *
 * The fix is a four-byte HEADER rather than a longer tag, and the difference is not
 * cosmetic: a tag is read out of the payload's own bytes, so no tag can ever be
 * unambiguous; a header occupies bytes 0 to 3 and the payload starts at 4, so an
 * audio sample cannot be read as a kind whatever its value. Ambiguity removed by
 * construction, not made unlikely. The cost is four bytes on a 1280-byte frame.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SATELLITE_PROTOCOL_HPP
#    define LPL_SATELLITE_PROTOCOL_HPP

#    include <lpl/Foundation.hpp>

namespace lpl::satellite {

/// Sampling rate every node captures and plays at.
inline constexpr core::u32 kSampleRateHz = 16000u;

/**
 * @brief Milliseconds of audio in one datagram.
 *
 * Forty, which is a latency budget rather than a round number: a node must be
 * interruptible mid-sentence, so the playback queue is drained a datagram at a time
 * and forty milliseconds is the granularity at which `STOP` takes effect.
 */
inline constexpr core::u32 kFrameMilliseconds = 40u;

/// Samples in one datagram's payload.
inline constexpr core::u32 kFrameSamples = kSampleRateHz * kFrameMilliseconds / 1000u;

/// Payload bytes in one audio datagram.
inline constexpr core::u32 kFramePayloadBytes = kFrameSamples * 2u;

/// First header byte, 'L'.
inline constexpr core::u8 kMagicFirst = 0x4Cu;

/// Second header byte, 'S'.
inline constexpr core::u8 kMagicSecond = 0x53u;

/// Bytes before the payload: magic, kind, sequence.
inline constexpr core::u32 kHeaderBytes = 4u;

/**
 * @enum Datagram
 * @brief What a datagram is.
 *
 * One enumeration for both directions rather than two, because the node and the
 * server have to agree on every value anyway and a split would let one of them
 * renumber its half.
 */
enum class Datagram : core::u8 {
    Unknown = 0u,        ///< Not a satellite datagram at all.
    Audio = 1u,          ///< PCM16 mono payload, either direction.
    EndOfUtterance = 2u, ///< Node to server: the utterance is closed.
    NoWakeWord = 3u,     ///< Server to node: nothing here was addressed to us.
    Text = 4u,           ///< Server to node: a transcript segment, as bytes.
    ReplyEnd = 5u,       ///< Server to node: the reply is complete.
    Interrupt = 6u,      ///< Server to node: stop playing, now.
};

/**
 * @struct Frame
 * @brief A decoded datagram: what it is, and where its payload starts.
 *
 * A view, never a copy. A node has the datagram in the buffer the driver filled and
 * has nowhere to put a second one.
 */
struct Frame {
    Datagram kind{Datagram::Unknown};
    core::u8 sequence{0u};
    const core::u8 *payload{nullptr};
    core::u32 payloadBytes{0u};
};

/**
 * @brief Writes a header.
 *
 * The sequence number wraps at 256 and is not used for ordering — over a datagram
 * link a node cannot reassemble anyway. It is there so a receiver can COUNT what it
 * lost, which is the difference between a link that is degrading and one that is
 * merely quiet.
 *
 * @param kind     What follows.
 * @param sequence Wrapping counter.
 * @param out      Receives @ref kHeaderBytes bytes.
 * @return Bytes written.
 */
core::u32 encodeHeader(Datagram kind, core::u8 sequence, core::u8 *out) noexcept;

/**
 * @brief Writes a whole datagram.
 * @param kind     What this is.
 * @param sequence Wrapping counter.
 * @param payload  Payload, may be null when @p payloadBytes is zero.
 * @param payloadBytes Its length.
 * @param out      Receives header + payload.
 * @param capacity Room in @p out.
 * @return Bytes written, or 0 when the room is short.
 */
core::u32 encode(Datagram kind, core::u8 sequence, const core::u8 *payload, core::u32 payloadBytes, core::u8 *out,
                 core::u32 capacity) noexcept;

/**
 * @brief Reads a datagram.
 *
 * Anything without the magic is @ref Datagram::Unknown and is NOT guessed at. A
 * satellite link carries other traffic — a stray broadcast, a port scan, the tail of
 * something else — and a decoder that fell back to "probably audio" would play it.
 *
 * @param bytes The datagram.
 * @param count Its length.
 * @param out   Receives the decoding.
 * @return false when it is not one of ours, or is malformed.
 */
[[nodiscard]] bool decode(const core::u8 *bytes, core::u32 count, Frame &out) noexcept;

/**
 * @brief Missed datagrams between two sequence numbers.
 *
 * Wrapping subtraction, so 3 after 254 is five lost and not minus two hundred and
 * fifty one.
 *
 * @param previous Last sequence seen.
 * @param current  Sequence just received.
 * @return Datagrams that went missing, zero for the expected successor.
 */
[[nodiscard]] core::u32 missedBetween(core::u8 previous, core::u8 current) noexcept;

/**
 * @brief Folds a datagram into a running FNV-1a hash.
 * @param hash  Running value.
 * @param bytes The datagram.
 * @param count Its length.
 * @return The updated value.
 */
[[nodiscard]] core::u32 foldDatagram(core::u32 hash, const core::u8 *bytes, core::u32 count) noexcept;

} // namespace lpl::satellite

#endif // LPL_SATELLITE_PROTOCOL_HPP
