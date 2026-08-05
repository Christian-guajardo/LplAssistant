/**
 * @file Inference.cpp
 * @brief The façade: prompt in, tokens out, within a budget.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Inference.hpp>

#if defined(LPL_HAS_FOUNDATION)

namespace lpl::infer {

bool Inference::initialise(TensorArena &arena, const Model &model, CachePolicy policy)
{
    if (!model.ready())
        return false;
    if (!_cache.allocate(arena, model.config(), policy))
        return false;
    if (!_transformer.allocate(arena, model))
        return false;

    math::Fixed32 *const logits = arena.claim<math::Fixed32>(model.config().vocabSize);
    bool *const allowed = arena.claim<bool>(model.config().vocabSize);
    if (logits == nullptr || allowed == nullptr)
        return false;

    _model = &model;
    _logits = VectorView{logits, model.config().vocabSize};
    _allowed = allowed;
    return true;
}

bool Inference::generate(const core::u32 *prompt, core::u32 promptCount, const GenerationParams &params,
                         const Grammar *grammar, core::u32 *out, core::u32 capacity,
                         GenerationReport &report) noexcept
{
    report = GenerationReport{};
    if (_model == nullptr || out == nullptr)
        return false;

    _cache.clear();
    Sampler sampler{params.sampler};
    GrammarState state = grammar != nullptr ? grammar->start() : GrammarState{};

    core::u32 position = 0u;
    for (core::u32 i = 0u; i < promptCount; ++i)
    {
        if (!_transformer.forward(prompt[i], position, _cache, _logits))
        {
            report.windowExhausted = true;
            report.promptTokens = i;
            return false;
        }
        ++position;
    }
    report.promptTokens = promptCount;

    // An empty prompt has no logits to sample from: the first forward pass has not
    // run, so the scores buffer holds whatever initialise left there. Refusing is
    // the honest answer; sampling zeros would produce a confident-looking token 0.
    if (promptCount == 0u)
        return false;

    const core::u32 ceiling = params.maxTokens < capacity ? params.maxTokens : capacity;
    while (report.generated < ceiling)
    {
        const bool *mask = nullptr;
        if (grammar != nullptr)
        {
            report.admittedLast = maskAllowedTokens(*grammar, state, _model->vocab(), _allowed);
            if (report.admittedLast == 0u)
            {
                report.grammarExhausted = true;
                break;
            }
            mask = _allowed;
        }

        const core::u32 token = sampler.next(_logits, mask);
        if (token == kNoToken)
        {
            report.grammarExhausted = grammar != nullptr;
            break;
        }

        if (grammar != nullptr)
        {
            core::u32 length = 0u;
            const char *const bytes = _model->vocab().text(token, length);
            GrammarState next{};
            // The mask was built from this same state, so a token it admitted must
            // advance it. Checking anyway keeps the two from being two statements of
            // the rule that could come apart.
            if (bytes == nullptr || !grammar->accepts(state, bytes, length, next))
            {
                report.grammarExhausted = true;
                break;
            }
            state = next;
        }

        out[report.generated++] = token;

        if (grammar != nullptr && grammar->complete(state))
        {
            report.grammarComplete = true;
            break;
        }

        if (!_transformer.forward(token, position, _cache, _logits))
        {
            report.windowExhausted = true;
            break;
        }
        ++position;
    }

    report.evictions = _cache.evictions();
    report.draws = sampler.draws();
    return true;
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
