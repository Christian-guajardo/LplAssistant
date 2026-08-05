/**
 * @file TensorArena.hpp
 * @brief The one big preallocated region weights and activations live in.
 *
 * Sized at boot from the host profile, never grown. A server profile renders
 * nothing, and that is exactly the memory this arena claims.
 *
 * It is a thin typed layer over @c lpl::memory::ArenaAllocator rather than a second
 * bump allocator, and that is deliberate: the repository already ships one arena and
 * folds its byte accounting as a determinism gate, so a second implementation would
 * be a second answer to how many bytes a given sequence of claims consumes. What
 * changes between a host and ring 0 is where the block comes from, never how it is
 * carved.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_TENSORARENA_HPP
#    define LPL_LPL_INFER_TENSORARENA_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/memory/ArenaAllocator.hpp>

namespace lpl::infer {

/**
 * @class TensorArena
 * @brief Typed, bounded, never-grown storage for a forward pass.
 */
class TensorArena {
public:
    /**
     * @brief Claims @p bytes from the host allocator.
     * @param bytes Capacity.
     */
    explicit TensorArena(core::usize bytes) : _arena(bytes) {}

    /**
     * @brief Adopts a block somebody else owns.
     *
     * The ring-0 form: the kernel reserves a region once at boot and hands it over,
     * so nothing here ever touches the kernel heap during a forward pass.
     *
     * @param memory Block base; must outlive the arena.
     * @param bytes  Block size.
     */
    TensorArena(void *memory, core::usize bytes) noexcept : _arena(memory, bytes) {}

    /**
     * @brief Claims room for @p count objects of @p T.
     *
     * Value-initialises what it returns. An uninitialised weight matrix would
     * produce a plausible-looking model whose output depended on what the block
     * held before, which is the one failure a determinism gate cannot describe.
     *
     * @tparam T     Element type; trivially constructible.
     * @param  count Elements.
     * @return The run, or nullptr when the arena is exhausted.
     */
    template <typename T> [[nodiscard]] T *claim(core::usize count)
    {
        if (count == 0u)
            return nullptr;
        void *const block = _arena.allocate(sizeof(T) * count, alignof(T));
        if (block == nullptr)
            return nullptr;
        T *const typed = static_cast<T *>(block);
        for (core::usize i = 0u; i < count; ++i)
            typed[i] = T{};
        return typed;
    }

    /**
     * @brief Bytes handed out so far.
     * @return The high-water mark, which never falls without a @ref reset.
     */
    [[nodiscard]] core::usize used() const noexcept { return _arena.used(); }

    /**
     * @brief Bytes the arena was given.
     * @return The capacity.
     */
    [[nodiscard]] core::usize capacity() const noexcept { return _arena.capacity(); }

    /**
     * @brief Releases everything at once.
     *
     * Valid between generations, never within one: the weights live in the same
     * arena as the activations, so a reset mid-pass would hand a matrix's storage
     * to the vector that was about to read it.
     */
    void reset() noexcept { _arena.reset(); }

    /**
     * @brief Mebibytes, spelled out.
     * @param count Mebibytes wanted.
     * @return The byte count.
     */
    [[nodiscard]] static constexpr core::usize mebibytes(core::usize count) noexcept
    {
        return count * 1024u * 1024u;
    }

    /**
     * @brief Kibibytes, spelled out.
     * @param count Kibibytes wanted.
     * @return The byte count.
     */
    [[nodiscard]] static constexpr core::usize kibibytes(core::usize count) noexcept { return count * 1024u; }

private:
    memory::ArenaAllocator _arena;
};

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_TENSORARENA_HPP
