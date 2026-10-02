/**
 * @file Sampler.cpp
 * @brief Choosing the next token, reproducibly.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Sampler.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/infer/Attention.hpp>
#    include <lpl/infer/Vocab.hpp>

namespace lpl::infer {

core::u32 Sampler::next(VectorView logits, const bool *allowed) noexcept
{
    if (logits.count == 0u)
        return kNoToken;

    const bool greedy = _params.topK <= 1u || _params.temperature.raw() <= 0;

    core::u32 best = kNoToken;
    for (core::u32 t = 0u; t < logits.count; ++t)
    {
        if (allowed != nullptr && !allowed[t])
            continue;
        if (best == kNoToken || logits.at(t) > logits.at(best))
            best = t;
    }
    if (best == kNoToken || greedy)
        return best;

    // Shortlist by repeated selection. K is at most kMaxTopK, so this is a bounded
    // number of passes over the vocabulary rather than a sort of it — and a sort
    // would have to define an order for equal scores, which selection does not.
    core::u32 candidates[kMaxTopK];
    math::Fixed32 weights[kMaxTopK];
    const core::u32 wanted = _params.topK < kMaxTopK ? _params.topK : kMaxTopK;

    core::u32 kept = 0u;
    for (core::u32 rank = 0u; rank < wanted; ++rank)
    {
        core::u32 pick = kNoToken;
        for (core::u32 t = 0u; t < logits.count; ++t)
        {
            if (allowed != nullptr && !allowed[t])
                continue;
            bool taken = false;
            for (core::u32 i = 0u; i < kept; ++i)
                taken = taken || candidates[i] == t;
            if (taken)
                continue;
            if (pick == kNoToken || logits.at(t) > logits.at(pick))
                pick = t;
        }
        if (pick == kNoToken)
            break;
        candidates[kept] = pick;
        weights[kept] = logits.at(pick) / _params.temperature;
        ++kept;
    }
    if (kept == 0u)
        return kNoToken;
    if (kept == 1u)
        return candidates[0];

    VectorView shortlist{weights, kept};
    softmaxInPlace(shortlist);

    core::i64 total = 0;
    for (core::u32 i = 0u; i < kept; ++i)
        total += weights[i].raw();
    if (total <= 0)
        return candidates[0];

    ++_draws;
    const core::u32 draw = _stream.below(static_cast<core::u32>(total));
    core::i64 running = 0;
    for (core::u32 i = 0u; i < kept; ++i)
    {
        running += weights[i].raw();
        if (static_cast<core::i64>(draw) < running)
            return candidates[i];
    }
    // The cumulative sum rounds down, so the very last slice can fall a raw unit
    // short of the draw. The last candidate owns that remainder.
    return candidates[kept - 1u];
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
