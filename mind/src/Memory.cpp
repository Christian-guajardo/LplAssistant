/**
 * @file Memory.cpp
 * @brief Implementation of the notes the demon writes to itself.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/mind/Memory.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/Fold.hpp>

namespace lpl::mind {

bool MemoryStore::remember(const MemoryNote &note) noexcept
{
    if (_count < kMaxNotes)
    {
        _notes[_count++] = note;
        return true;
    }

    /* Find the weakest held note. Ties go to the OLDER one, and a second tie to the
       lower index — not because either rule is obviously right, but because a rule
       that left the choice to iteration order would let two targets keep different
       notes from the same sequence of events, and every recall after that would
       diverge without anything looking broken. */
    core::u32 weakest = 0u;
    for (core::u32 i = 1u; i < kMaxNotes; ++i)
    {
        if (_notes[i].salience < _notes[weakest].salience)
        {
            weakest = i;
            continue;
        }
        if (_notes[i].salience == _notes[weakest].salience && _notes[i].tick < _notes[weakest].tick)
            weakest = i;
    }

    /* The half that stops a flood of trivia from erasing everything worth keeping. An
       incoming note has to be at least as salient as the weakest thing already held;
       otherwise the store is doing its job by saying no. */
    if (note.salience < _notes[weakest].salience)
    {
        ++_refusals;
        return false;
    }

    _notes[weakest] = note;
    ++_evictions;
    return true;
}

core::u32 foldMemory(const MemoryStore &store, core::u32 hash) noexcept
{
    foldWord(hash, store.count());
    for (core::u32 i = 0u; i < store.count(); ++i)
    {
        const MemoryNote &note = store.note(i);
        foldWord(hash, note.bytes);
        foldBytes(hash, reinterpret_cast<const core::u8 *>(note.text), note.bytes);
        foldWord(hash, note.topic);
        foldWord(hash, note.tick);
        foldWord(hash, static_cast<core::u32>(note.salience.raw()));
    }
    /* Both counters are folded. What a store REFUSED is as much a fact about it as
       what it holds — two stores with identical contents, one of which turned away a
       dozen notes, are not in the same state. */
    foldWord(hash, store.evictions());
    foldWord(hash, store.refusals());
    return hash;
}

} // namespace lpl::mind

#endif // LPL_HAS_FOUNDATION
