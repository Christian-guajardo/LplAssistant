/**
 * @file Persona.hpp
 * @brief Who the demon is, as editable data.
 *
 * Not a fine-tune. A personality baked into weights cannot be reviewed, diffed or
 * reverted, and the sovereign must be able to do all three.
 *
 * The consequence of that choice is what the type looks like: fixed-capacity text the
 * sovereign writes, plus three traits that actually change behaviour. The traits are
 * `Fixed32` and not float because the reasoner reads them on every step — a demon that
 * asked instead of acting on one target and acted instead of asking on the other would
 * be a different demon grown from the same file.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_MIND_PERSONA_HPP
#    define LPL_LPL_MIND_PERSONA_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/math/FixedPoint.hpp>

namespace lpl::mind {

/// Bytes a persona's name may occupy.
inline constexpr core::u32 kPersonaNameBytes = 32u;

/// Bytes one directive may occupy.
inline constexpr core::u32 kDirectiveBytes = 128u;

/// Directives a persona may carry.
inline constexpr core::u32 kMaxDirectives = 8u;

/**
 * @struct Persona
 * @brief The demon's identity, in a form the sovereign can read and edit.
 */
struct Persona {
    char name[kPersonaNameBytes]{};                     ///< What it answers to.
    core::u32 nameBytes{0u};                            ///< Bytes of @ref name in use.
    char directives[kMaxDirectives][kDirectiveBytes]{}; ///< Standing instructions, in order.
    core::u32 directiveBytes[kMaxDirectives]{};         ///< Bytes of each directive in use.
    core::u32 directiveCount{0u};                       ///< Directives in force.

    /**
     * How readily it asks rather than acts.
     *
     * At one it never acts on an intent it cannot resolve; at zero it never asks.
     * Authoritative because it decides which branch a step takes, and two targets that
     * rounded it differently would produce two transcripts from one question.
     */
    math::Fixed32 caution{};

    /// How little it says. At one, the shortest thing that is still an answer.
    math::Fixed32 brevity{};

    /// How readily it acts without being asked.
    math::Fixed32 initiative{};
};

/**
 * @brief Gives a persona its name.
 *
 * Truncates rather than refusing. A name is not a safety property, and a persona that
 * failed to load because somebody typed thirty-three characters would be a worse
 * outcome than one called something slightly shorter.
 *
 * @param persona Persona to name.
 * @param text    Bytes of the name.
 * @param count   How many.
 * @return Bytes actually stored.
 */
core::u32 personaSetName(Persona &persona, const char *text, core::u32 count) noexcept;

/**
 * @brief Appends a standing instruction.
 *
 * Order is kept, because directives are read in order and a later one is understood to
 * qualify an earlier one. Reordering them would change what the persona means without
 * changing what it says.
 *
 * @param persona Persona to extend.
 * @param text    Bytes of the directive.
 * @param count   How many.
 * @return true when it was stored; false when the persona is already full.
 */
bool personaAddDirective(Persona &persona, const char *text, core::u32 count) noexcept;

/**
 * @brief Folds a persona into a signature.
 *
 * @param persona Persona to fold.
 * @param hash    Running value.
 * @return The updated hash.
 */
[[nodiscard]] core::u32 foldPersona(const Persona &persona, core::u32 hash) noexcept;

} // namespace lpl::mind

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_MIND_PERSONA_HPP
