/**
 * @file Persona.cpp
 * @brief Implementation of who the demon is, as editable data.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/mind/Persona.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/Fold.hpp>

namespace lpl::mind {

namespace {

/**
 * @brief Copies bytes into a fixed slot, truncating at its capacity.
 *
 * @param destination Slot to fill.
 * @param capacity    Its size.
 * @param text        Bytes to copy; may be null when @p count is zero.
 * @param count       How many.
 * @return Bytes written.
 */
core::u32 storeBounded(char *destination, core::u32 capacity, const char *text, core::u32 count) noexcept
{
    const core::u32 stored = count < capacity ? count : capacity;
    for (core::u32 i = 0u; i < stored; ++i)
        destination[i] = text[i];
    return stored;
}

} // namespace

core::u32 personaSetName(Persona &persona, const char *text, core::u32 count) noexcept
{
    if (text == nullptr)
        count = 0u;
    persona.nameBytes = storeBounded(persona.name, kPersonaNameBytes, text, count);
    return persona.nameBytes;
}

bool personaAddDirective(Persona &persona, const char *text, core::u32 count) noexcept
{
    if (persona.directiveCount >= kMaxDirectives || text == nullptr)
        return false;

    const core::u32 slot = persona.directiveCount;
    persona.directiveBytes[slot] = storeBounded(persona.directives[slot], kDirectiveBytes, text, count);
    ++persona.directiveCount;
    return true;
}

core::u32 foldPersona(const Persona &persona, core::u32 hash) noexcept
{
    foldBytes(hash, reinterpret_cast<const core::u8 *>(persona.name), persona.nameBytes);
    foldWord(hash, persona.directiveCount);
    for (core::u32 i = 0u; i < persona.directiveCount; ++i)
    {
        foldWord(hash, persona.directiveBytes[i]);
        foldBytes(hash, reinterpret_cast<const core::u8 *>(persona.directives[i]), persona.directiveBytes[i]);
    }

    /* The traits are folded by their raw representation rather than by any decimal
       rendering of them. A signature exists to catch a target that computed a
       different number, and rendering would hide small differences behind rounding —
       which is the one class of difference worth catching. */
    foldWord(hash, static_cast<core::u32>(persona.caution.raw()));
    foldWord(hash, static_cast<core::u32>(persona.brevity.raw()));
    foldWord(hash, static_cast<core::u32>(persona.initiative.raw()));
    return hash;
}

} // namespace lpl::mind

#endif // LPL_HAS_FOUNDATION
