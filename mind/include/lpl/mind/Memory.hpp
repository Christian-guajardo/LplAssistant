/**
 * @file Memory.hpp
 * @brief The notes the demon writes to itself.
 *
 * On a salient event it emits a short record and files it; weeks later a lookup
 * surfaces it again. Long memory without retraining, and — because the notes are text —
 * memory the sovereign can audit and correct.
 *
 * The store is bounded, so the interesting question is not how to add a note but which
 * note dies when it is full. The rule here has two halves and the second is the one
 * that matters: the least salient note is evicted, ties broken by age and then by
 * index so two targets cannot disagree; AND a note is refused outright if everything
 * already stored is more salient than it is. Without that second half, a flood of
 * trivia quietly erases everything worth keeping, which is the failure mode of every
 * fixed-size memory that only ever drops its oldest entry.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_MIND_MEMORY_HPP
#    define LPL_LPL_MIND_MEMORY_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/math/FixedPoint.hpp>

namespace lpl::mind {

/// Bytes one note may occupy.
inline constexpr core::u32 kNoteBytes = 96u;

/// Notes the store holds before it has to choose.
inline constexpr core::u32 kMaxNotes = 32u;

/**
 * @struct MemoryNote
 * @brief One thing worth remembering.
 */
struct MemoryNote {
    char text[kNoteBytes]{}; ///< What happened, in the demon's own words.
    core::u32 bytes{0u};     ///< Bytes of @ref text in use.
    core::u32 topic{0u};     ///< What it is about, for the structured filter.
    core::u32 tick{0u};      ///< When it was filed, in turns rather than in time.

    /**
     * How much it matters.
     *
     * Authoritative: it decides which note survives a full store, so a rounding
     * difference between two targets would eventually give them different pasts.
     */
    math::Fixed32 salience{};
};

/**
 * @class MemoryStore
 * @brief A bounded set of notes with a deterministic eviction rule.
 */
class MemoryStore {
  public:
    /**
     * @brief Files a note.
     *
     * @param note Note to keep.
     * @return true when it was stored.
     */
    bool remember(const MemoryNote &note) noexcept;

    /// Notes currently held.
    [[nodiscard]] core::u32 count() const noexcept { return _count; }

    /**
     * @brief One held note.
     * @param index Below @ref count.
     * @return The note.
     */
    [[nodiscard]] const MemoryNote &note(core::u32 index) const noexcept { return _notes[index]; }

    /// Notes displaced to make room.
    [[nodiscard]] core::u32 evictions() const noexcept { return _evictions; }

    /// Notes refused because everything held mattered more.
    [[nodiscard]] core::u32 refusals() const noexcept { return _refusals; }

  private:
    MemoryNote _notes[kMaxNotes]{};
    core::u32 _count{0u};
    core::u32 _evictions{0u};
    core::u32 _refusals{0u};
};

/**
 * @brief Folds a store into a signature.
 *
 * @param store Store to fold.
 * @param hash  Running value.
 * @return The updated hash.
 */
[[nodiscard]] core::u32 foldMemory(const MemoryStore &store, core::u32 hash) noexcept;

} // namespace lpl::mind

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_MIND_MEMORY_HPP
