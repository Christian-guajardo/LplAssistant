/**
 * @file Protocol.cpp
 * @brief The satellite wire format, in one place.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/satellite/Protocol.hpp>

namespace lpl::satellite {

namespace {

constexpr core::u32 kFnv1aPrime = 0x01000193u;

constexpr core::u32 kVersionOffset = 2u;
constexpr core::u32 kKindOffset = 3u;
constexpr core::u32 kSequenceOffset = 4u;

static_assert(kSequenceOffset + 1u == kHeaderBytes, "the sequence is the last byte of the header");

/**
 * @brief Does the datagram begin with the satellite magic?
 * @param bytes The datagram, may be null.
 * @param count Its length.
 * @return true when its first two bytes are 'L' 'S'.
 */
bool hasMagic(const core::u8 *bytes, core::u32 count) noexcept
{
    return bytes != nullptr && count >= kVersionOffset && bytes[0] == kMagicFirst && bytes[1] == kMagicSecond;
}

/**
 * @brief Is this byte a kind the protocol defines?
 * @param value Fourth header byte.
 * @return true when it names a datagram.
 */
bool knownKind(core::u8 value) noexcept
{
    return value >= static_cast<core::u8>(Datagram::Audio) && value <= static_cast<core::u8>(Datagram::Interrupt);
}

/**
 * @brief Does the payload hold what its kind promises?
 *
 * An audio payload that is not a whole number of samples is a truncated datagram, and
 * playing it would emit a click at the seam. Refused rather than rounded down: a link
 * that truncates is a link to report, not to paper over.
 *
 * @param kind         What the header says the datagram is.
 * @param payloadBytes Bytes after the header.
 * @return false for an audio payload that is empty or cut mid-sample.
 */
bool payloadIsWhole(Datagram kind, core::u32 payloadBytes) noexcept
{
    return kind != Datagram::Audio || (payloadBytes != 0u && (payloadBytes % 2u) == 0u);
}

/**
 * @brief Records why a datagram was refused.
 * @param out    The decoding under way.
 * @param reason Why it stops.
 * @return false, what the refused decode returns.
 */
bool refuse(Frame &out, Refusal reason) noexcept
{
    out.refusal = reason;
    return false;
}

} // namespace

core::u32 encodeHeader(Datagram kind, core::u8 sequence, core::u8 *out) noexcept
{
    if (out == nullptr)
        return 0u;
    out[0] = kMagicFirst;
    out[1] = kMagicSecond;
    out[kVersionOffset] = kProtocolVersion;
    out[kKindOffset] = static_cast<core::u8>(kind);
    out[kSequenceOffset] = sequence;
    return kHeaderBytes;
}

core::u32 encode(Datagram kind, core::u8 sequence, const core::u8 *payload, core::u32 payloadBytes, core::u8 *out,
                 core::u32 capacity) noexcept
{
    if (out == nullptr || capacity < kHeaderBytes + payloadBytes)
        return 0u;
    if (payloadBytes != 0u && payload == nullptr)
        return 0u;

    encodeHeader(kind, sequence, out);
    for (core::u32 i = 0u; i < payloadBytes; ++i)
        out[kHeaderBytes + i] = payload[i];
    return kHeaderBytes + payloadBytes;
}

bool decode(const core::u8 *bytes, core::u32 count, Frame &out) noexcept
{
    out = Frame{};
    if (!hasMagic(bytes, count))
        return refuse(out, Refusal::NotSatellite);
    if (count <= kVersionOffset)
        return refuse(out, Refusal::ShortHeader);

    out.version = bytes[kVersionOffset];
    if (out.version != kProtocolVersion)
        return refuse(out, Refusal::UnknownVersion);
    if (count < kHeaderBytes)
        return refuse(out, Refusal::ShortHeader);
    if (!knownKind(bytes[kKindOffset]))
        return refuse(out, Refusal::UnknownKind);

    out.kind = static_cast<Datagram>(bytes[kKindOffset]);
    out.sequence = bytes[kSequenceOffset];
    out.payloadBytes = count - kHeaderBytes;
    out.payload = out.payloadBytes == 0u ? nullptr : bytes + kHeaderBytes;
    if (!payloadIsWhole(out.kind, out.payloadBytes))
        return refuse(out, Refusal::PartialSample);
    return true;
}

core::u32 missedBetween(core::u8 previous, core::u8 current) noexcept
{
    const core::u32 gap = static_cast<core::u32>(static_cast<core::u8>(current - previous));
    return gap == 0u ? 0u : gap - 1u;
}

core::u32 foldDatagram(core::u32 hash, const core::u8 *bytes, core::u32 count) noexcept
{
    if (bytes == nullptr)
        return hash;
    for (core::u32 i = 0u; i < count; ++i)
        hash = (hash ^ static_cast<core::u32>(bytes[i])) * kFnv1aPrime;
    return (hash ^ count) * kFnv1aPrime;
}

} // namespace lpl::satellite
