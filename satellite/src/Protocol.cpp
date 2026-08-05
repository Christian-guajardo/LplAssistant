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

/**
 * @brief Is this byte a kind the protocol defines?
 * @param value Third header byte.
 * @return true when it names a datagram.
 */
bool knownKind(core::u8 value) noexcept
{
    return value >= static_cast<core::u8>(Datagram::Audio) && value <= static_cast<core::u8>(Datagram::Interrupt);
}

} // namespace

core::u32 encodeHeader(Datagram kind, core::u8 sequence, core::u8 *out) noexcept
{
    if (out == nullptr)
        return 0u;
    out[0] = kMagicFirst;
    out[1] = kMagicSecond;
    out[2] = static_cast<core::u8>(kind);
    out[3] = sequence;
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
    if (bytes == nullptr || count < kHeaderBytes)
        return false;
    if (bytes[0] != kMagicFirst || bytes[1] != kMagicSecond || !knownKind(bytes[2]))
        return false;

    out.kind = static_cast<Datagram>(bytes[2]);
    out.sequence = bytes[3];
    out.payloadBytes = count - kHeaderBytes;
    out.payload = out.payloadBytes == 0u ? nullptr : bytes + kHeaderBytes;

    // An audio payload that is not a whole number of samples is a truncated
    // datagram, and playing it would emit a click at the seam. Refused rather than
    // rounded down: a link that truncates is a link to report, not to paper over.
    if (out.kind == Datagram::Audio && (out.payloadBytes == 0u || (out.payloadBytes % 2u) != 0u))
        return false;

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
