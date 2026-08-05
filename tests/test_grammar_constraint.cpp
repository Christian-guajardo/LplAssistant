/**
 * @file test_grammar_constraint.cpp
 * @brief A forbidden tool call must be unrepresentable, not merely unlikely.
 *
 * Drive the sampler against a grammar that excludes an action and assert it can
 * never be emitted — the property that lets a small local model be trusted with
 * an engine.
 *
 * The demonstration is stronger than it looks because the model is UNTRAINED. Its
 * weights come from a seed, so left alone it babbles; under the grammar it emits a
 * valid tool call every time. That is the whole argument for constrained decoding
 * stated as an experiment: the guarantee comes from the language, not from the
 * model's competence, so it does not weaken when the model is small.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/infer/Inference.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/infer/Tokenizer.hpp>

#include <cstdio>

namespace {

int gFailures = 0;
int gChecks = 0;

void check(bool condition, const char *what)
{
    ++gChecks;
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
        ++gFailures;
}

/// Length of a null-terminated string, so the test needs no <cstring>.
lpl::core::u32 textLength(const char *text)
{
    lpl::core::u32 length = 0u;
    while (text[length] != '\0')
        ++length;
    return length;
}

/// Byte-for-byte equality, likewise.
bool sameText(const char *lhs, lpl::core::u32 lhsLength, const char *rhs)
{
    if (lhsLength != textLength(rhs))
        return false;
    for (lpl::core::u32 i = 0u; i < lhsLength; ++i)
        if (lhs[i] != rhs[i])
            return false;
    return true;
}

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== grammar: a malformed call is unrepresentable ==\n");

    infer::TensorArena arena{infer::parityArenaBytes()};

    infer::Vocab vocab;
    check(infer::buildParityVocab(arena, vocab), "the canonical vocabulary builds");

    core::u32 phraseCount = 0u;
    const char *const *phrases = infer::parityGrammarPhrases(phraseCount);
    infer::Grammar grammar;
    check(grammar.build(arena, phrases, phraseCount), "the grammar builds");
    check(grammar.size() == 2u, "it offers exactly the two calls the engine has");

    // ── The acceptor, on its own ─────────────────────────────────────────────
    std::printf("\n-- what the language admits, byte by byte --\n");
    {
        const auto walk = [&](const char *text) {
            infer::GrammarState state = grammar.start();
            const core::u32 length = textLength(text);
            for (core::u32 i = 0u; i < length; ++i)
            {
                infer::GrammarState next{};
                if (!grammar.accepts(state, text + i, 1u, next))
                    return false;
                state = next;
            }
            return grammar.complete(state);
        };

        check(walk("generate_world"), "generate_world is in the language");
        check(walk("place_entity"), "place_entity is in the language");
        check(!walk(infer::parityForbiddenPhrase()), "set_biome is NOT in the language");
        check(!walk("delete_everything"), "neither is anything else");
        check(!walk("generate_worlds"), "nor a valid call with one byte too many");

        infer::GrammarState partial = grammar.start();
        infer::GrammarState next{};
        check(grammar.accepts(partial, "gen", 3u, next), "a three-byte token can advance the state");
        check(!grammar.complete(next), "a prefix is not an accepted phrase");
    }

    // ── The mask ─────────────────────────────────────────────────────────────
    std::printf("\n-- which tokens the sampler is even offered --\n");
    {
        bool *const allowed = arena.claim<bool>(vocab.size());
        const core::u32 admitted = infer::maskAllowedTokens(grammar, grammar.start(), vocab, allowed);
        std::printf("    %u of %u tokens can begin a valid call\n", admitted, vocab.size());
        check(admitted > 0u && admitted < vocab.size(), "the mask is neither empty nor everything");

        // Every admitted token must actually start one of the phrases. A mask that
        // let something else through would make the guarantee a coincidence of this
        // particular vocabulary.
        bool consistent = true;
        for (core::u32 t = 0u; t < vocab.size() && consistent; ++t)
        {
            if (!allowed[t])
                continue;
            core::u32 length = 0u;
            const char *const bytes = vocab.text(t, length);
            bool startsOne = false;
            for (core::u32 p = 0u; p < phraseCount; ++p)
            {
                bool prefix = length <= textLength(phrases[p]);
                for (core::u32 b = 0u; prefix && b < length; ++b)
                    prefix = bytes[b] == phrases[p][b];
                startsOne = startsOne || prefix;
            }
            consistent = startsOne;
        }
        check(consistent, "every admitted token is a prefix of a phrase the grammar accepts");
    }

    // ── The model, driven ────────────────────────────────────────────────────
    std::printf("\n-- an untrained model, made to speak correctly --\n");
    {
        infer::ModelConfig shape = infer::parityModelConfig();
        shape.vocabSize = vocab.size();
        infer::Model model;
        check(model.synthesise(arena, shape, vocab, infer::parityModelSeed()), "the model builds");

        infer::Tokenizer tokenizer{vocab};
        const char *const prompt = infer::parityPrompt();
        core::u32 tokens[64];
        core::u32 skipped = 0u;
        const core::u32 count = tokenizer.encode(prompt, textLength(prompt), tokens, 64u, skipped);

        infer::Inference inference;
        check(inference.initialise(arena, model, infer::CachePolicy::Refuse), "the façade initialises");

        infer::GenerationParams params{};
        params.sampler.topK = 4u;
        params.maxTokens = 24u;

        // Several seeds, because one run proving it is a run, and the claim is about
        // every run.
        core::u32 valid = 0u;
        core::u32 attempts = 0u;
        for (core::u32 seed = 1u; seed <= 8u; ++seed)
        {
            params.sampler.seed = seed * 7919u;
            core::u32 out[32];
            infer::GenerationReport report{};
            if (!inference.generate(tokens, count, params, &grammar, out, 32u, report))
                continue;
            ++attempts;

            char text[128];
            const core::u32 bytes = tokenizer.decode(out, report.generated, text, 128u);
            const bool spelled = sameText(text, bytes, "generate_world") || sameText(text, bytes, "place_entity");
            if (report.grammarComplete && spelled)
                ++valid;
            if (seed <= 3u)
                std::printf("    seed %-6u -> \"%.*s\"\n", params.sampler.seed, static_cast<int>(bytes), text);
        }
        std::printf("    %u of %u runs emitted a whole, valid call\n", valid, attempts);
        check(attempts == 8u, "all eight runs completed");
        check(valid == attempts, "every run emitted a whole call the engine can execute");

        // And the negative: without the grammar, the same model on the same prompt
        // does not produce a valid call. Without this the test could pass on a model
        // that happened to be right anyway.
        core::u32 accidental = 0u;
        for (core::u32 seed = 1u; seed <= 8u; ++seed)
        {
            params.sampler.seed = seed * 7919u;
            core::u32 out[32];
            infer::GenerationReport report{};
            if (!inference.generate(tokens, count, params, nullptr, out, 32u, report))
                continue;
            char text[128];
            const core::u32 bytes = tokenizer.decode(out, report.generated, text, 128u);
            if (sameText(text, bytes, "generate_world") || sameText(text, bytes, "place_entity"))
                ++accidental;
        }
        std::printf("    %u of 8 unconstrained runs stumbled onto a valid call\n", accidental);
        check(accidental == 0u, "the guarantee comes from the grammar and not from the model");
    }

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
