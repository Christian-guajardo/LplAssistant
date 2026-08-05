/**
 * @file test_infer_parity.cpp
 * @brief The same prompt and seed must produce the same tokens on both targets.
 *
 * The gate for lot 8, and the one that decides whether a demon in ring 0 is a
 * real claim: an inference that cannot be replayed cannot be audited.
 *
 * The signatures at the end are the contract. Everything above them is what makes a
 * mismatch READABLE: quantisation, normalisation, the exponential, the rotary angles
 * and the sampling draw are each checked on their own, so a failing gate says which
 * layer moved instead of only that the answer changed.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/infer/Attention.hpp>
#include <lpl/infer/Inference.hpp>
#include <lpl/infer/Parity.hpp>
#include <lpl/infer/Tokenizer.hpp>

#include <lpl/math/FixedMath.hpp>

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

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== mind: the same seed thinks the same thought ==\n");

    // ── The exponential the softmax stands on ────────────────────────────────
    std::printf("\n-- an exponential without libm --\n");
    {
        check(math::fixedExp2(math::Fixed32::zero()) == math::Fixed32::one(), "2^0 is exactly one");
        check(math::fixedExp2(math::Fixed32::one()) == math::Fixed32::fromInt(2), "2^1 is exactly two");
        check(math::fixedExp2(math::Fixed32::fromInt(-1)) == math::Fixed32::half(), "2^-1 is exactly one half");
        check(math::fixedExp2(math::Fixed32::fromInt(10)) == math::Fixed32::fromInt(1024),
              "2^10 is exactly 1024 — the integer part is a shift, not an approximation");

        // The inverse pair actually inverts. Not a tautology: log2 is piecewise
        // linear and exp2 is a Taylor series, so agreeing at the powers of two is
        // the strongest claim the two can make about each other.
        //
        // Up to 2^14 and not further, and the bound is the format's rather than the
        // function's: Q16.16 tops out just below 32768, so 2^15 is not a value
        // Fixed32 can hold. exp2 saturates there, which is the documented behaviour
        // and not a failure to invert.
        bool inverts = true;
        for (core::u32 power = 0u; power <= 14u; ++power)
        {
            const core::u32 value = core::u32{1} << power;
            inverts = inverts && math::fixedExp2(math::fixedLog2(value)) == math::Fixed32::fromInt(
                                                                                static_cast<core::i32>(value));
        }
        check(inverts, "exp2 undoes log2 on every power of two Fixed32 can hold");

        check(math::fixedExp2(math::Fixed32::fromInt(-40)) == math::Fixed32::zero(),
              "a hopeless score underflows to zero rather than wrapping");
        check(math::fixedExp2(math::Fixed32::fromInt(40)) == math::Fixed32::max(),
              "an impossible score saturates rather than wrapping");
    }

    // ── Quantisation ─────────────────────────────────────────────────────────
    std::printf("\n-- eight bits, and what they cost --\n");
    {
        infer::QuantBlock block{};
        math::Fixed32 row[infer::kQuantBlockLanes];
        for (core::u32 i = 0u; i < infer::kQuantBlockLanes; ++i)
            row[i] = math::Fixed32::fromRaw(static_cast<core::i32>(i) * 1024 - 16384);

        infer::quantiseRow(row, infer::kQuantBlockLanes, &block);

        core::i32 worst = 0;
        for (core::u32 i = 0u; i < infer::kQuantBlockLanes; ++i)
        {
            const core::i32 error = (infer::dequantiseLane(block, i) - row[i]).abs().raw();
            if (error > worst)
                worst = error;
        }
        // Half a level of the block's own scale is the definition of correct
        // rounding. Anything larger is a bug, not a precision limit.
        const core::i32 halfLevel = block.scaleRaw / (2 * infer::kQuantLevels) + 1;
        std::printf("    worst lane error %d raw, half a level is %d raw\n", worst, halfLevel);
        check(worst <= halfLevel, "no lane is off by more than half a quantisation level");

        math::Fixed32 zeros[infer::kQuantBlockLanes] = {};
        infer::QuantBlock empty{};
        infer::quantiseRow(zeros, infer::kQuantBlockLanes, &empty);
        check(empty.scaleRaw == 0 && infer::dequantiseLane(empty, 0u) == math::Fixed32::zero(),
              "an all-zero block quantises to zero instead of dividing by its own scale");
    }

    // ── Normalisation and rotation ───────────────────────────────────────────
    std::printf("\n-- the two operations position depends on --\n");
    {
        math::Fixed32 values[4] = {math::Fixed32::fromInt(3), math::Fixed32::fromInt(-4), math::Fixed32::fromInt(0),
                                   math::Fixed32::fromInt(5)};
        math::Fixed32 gains[4] = {math::Fixed32::one(), math::Fixed32::one(), math::Fixed32::one(),
                                  math::Fixed32::one()};
        math::Fixed32 normed[4] = {};
        infer::rmsNorm(infer::ConstVectorView{values, 4u}, infer::ConstVectorView{gains, 4u},
                       infer::VectorView{normed, 4u});

        core::i64 sum = 0;
        for (core::u32 i = 0u; i < 4u; ++i)
            sum += (static_cast<core::i64>(normed[i].raw()) * normed[i].raw()) >> 16;
        const math::Fixed32 rms = math::fixedSqrt(math::Fixed32::fromRaw(static_cast<core::i32>(sum / 4)));
        std::printf("    root mean square after normalisation: %.4f\n", static_cast<double>(rms.toFloat()));
        check((rms - math::Fixed32::one()).abs() < math::Fixed32::fromRaw(math::Fixed32::kOne / 16),
              "rmsNorm leaves a vector with unit root mean square");

        // The rotation is what makes attention see RELATIVE position: two vectors
        // rotated by the same amount keep their dot product.
        math::Fixed32 first[4] = {math::Fixed32::half(), math::Fixed32::fromRaw(math::Fixed32::kOne / 4),
                                  math::Fixed32::fromRaw(-math::Fixed32::kOne / 8), math::Fixed32::half()};
        math::Fixed32 second[4] = {math::Fixed32::fromRaw(math::Fixed32::kOne / 3), math::Fixed32::half(),
                                   math::Fixed32::half(), math::Fixed32::fromRaw(-math::Fixed32::kOne / 5)};

        const auto dot = [](const math::Fixed32 *a, const math::Fixed32 *b) {
            core::i64 total = 0;
            for (core::u32 i = 0u; i < 4u; ++i)
                total += (static_cast<core::i64>(a[i].raw()) * b[i].raw()) >> 16;
            return math::Fixed32::fromRaw(static_cast<core::i32>(total));
        };

        const math::Fixed32 before = dot(first, second);
        infer::applyRotaryEmbedding(infer::VectorView{first, 4u}, 1u, 4u, 5u);
        infer::applyRotaryEmbedding(infer::VectorView{second, 4u}, 1u, 4u, 5u);
        const math::Fixed32 after = dot(first, second);
        std::printf("    dot product before %.4f, after a shared rotation %.4f\n",
                    static_cast<double>(before.toFloat()), static_cast<double>(after.toFloat()));
        check((before - after).abs() < math::Fixed32::fromRaw(math::Fixed32::kOne / 64),
              "rotating both vectors by the same position leaves their dot product alone");
    }

    // ── Softmax ──────────────────────────────────────────────────────────────
    std::printf("\n-- scores become a distribution --\n");
    {
        math::Fixed32 scores[5] = {math::Fixed32::fromInt(1), math::Fixed32::fromInt(3), math::Fixed32::fromInt(2),
                                   math::Fixed32::fromInt(-1), math::Fixed32::fromInt(0)};
        infer::softmaxInPlace(infer::VectorView{scores, 5u});

        core::i64 total = 0;
        for (core::u32 i = 0u; i < 5u; ++i)
            total += scores[i].raw();
        check(total > math::Fixed32::kOne - 64 && total <= math::Fixed32::kOne,
              "the weights sum to one, up to the format's resolution");
        check(scores[1] > scores[2] && scores[2] > scores[0] && scores[0] > scores[4] && scores[4] > scores[3],
              "the order of the scores survives");

        // The subtraction of the maximum is not cosmetic: without it these two
        // land on the same clamped value and become equally likely.
        math::Fixed32 extreme[2] = {math::Fixed32::fromInt(20), math::Fixed32::fromInt(40)};
        infer::softmaxInPlace(infer::VectorView{extreme, 2u});
        check(extreme[1] > extreme[0], "a score of forty still beats a score of twenty after the exponential");
    }

    // ── The canonical case ───────────────────────────────────────────────────
    std::printf("\n-- the run the kernel must reproduce --\n");
    infer::MindFoldResult folded{};
    infer::foldMindState(folded);

    check(folded.vocabSize == 121u, "the canonical vocabulary is 121 tokens");
    check(folded.weightSignature != 0u && folded.tokenSignature != 0u, "the run produced signatures");
    check(folded.generated == infer::parityGeneratedTokens(), "the free run produced its whole budget of tokens");
    check(folded.draws == folded.generated, "one sampling draw per token — the stream is not consumed elsewhere");
    check(folded.blobReopened == 1u, "the model written out and read back has the same weights, bit for bit");

    // Occupancy is checked against the capacity and NOT against the kernel's figure.
    // A bump allocator's byte count only matches across targets when everything it
    // carves is the same size on both, and LayerWeights holds nine views — a pointer
    // each — so it is 144 bytes here and 100 in ring 0. Comparing the two would be a
    // gate that fails for a reason that is not a defect.
    check(folded.arenaBytes < infer::parityArenaBytes(), "the whole run fits inside the arena it declared");
    std::printf("    arena occupancy %u of %u bytes (a per-target figure, not a signature)\n",
                folded.arenaBytes, static_cast<core::u32>(infer::parityArenaBytes()));

    // Replay: the property the whole project is built on.
    infer::MindFoldResult again{};
    infer::foldMindState(again);
    check(again.weightSignature == folded.weightSignature && again.tokenSignature == folded.tokenSignature &&
              again.residualSignature == folded.residualSignature,
          "running it twice produces the same mind");

    // A different seed must produce a different mind, or the seed is decoration.
    {
        infer::TensorArena arena{infer::parityArenaBytes()};
        infer::Vocab vocab;
        infer::buildParityVocab(arena, vocab);
        infer::ModelConfig shape = infer::parityModelConfig();
        shape.vocabSize = vocab.size();
        infer::Model other;
        const bool built = other.synthesise(arena, shape, vocab, infer::parityModelSeed() + 1u);
        check(built && other.fold(0x811C9DC5u) != folded.weightSignature,
              "a different seed derives a different model");
    }

    std::printf("\n-- what it actually said --\n");
    {
        infer::TensorArena arena{infer::parityArenaBytes()};
        infer::Vocab vocab;
        infer::buildParityVocab(arena, vocab);
        infer::ModelConfig shape = infer::parityModelConfig();
        shape.vocabSize = vocab.size();
        infer::Model model;
        const bool built = model.synthesise(arena, shape, vocab, infer::parityModelSeed());
        check(built, "the canonical model builds");

        infer::Tokenizer tokenizer{vocab};
        const char *const prompt = infer::parityPrompt();
        core::u32 tokens[64];
        core::u32 skipped = 0u;
        const core::u32 count = tokenizer.encode(prompt, textLength(prompt), tokens, 64u, skipped);
        check(skipped == 0u, "the canonical prompt is spellable in the canonical vocabulary");

        char round[128];
        const core::u32 bytes = tokenizer.decode(tokens, count, round, 128u);
        bool same = bytes == textLength(prompt);
        for (core::u32 i = 0u; same && i < bytes; ++i)
            same = round[i] == prompt[i];
        check(same, "tokenising the prompt and decoding it back gives the prompt");

        std::printf("    prompt \"%s\" -> %u tokens\n", prompt, count);

        infer::Inference inference;
        const bool ready = inference.initialise(arena, model, infer::CachePolicy::Refuse);
        check(ready, "the inference façade claims its cache and scratch");

        infer::GenerationParams params{};
        params.sampler.topK = 4u;
        params.sampler.seed = infer::paritySamplerSeed();
        params.maxTokens = infer::parityGeneratedTokens();

        core::u32 out[32];
        infer::GenerationReport report{};
        const bool ran = inference.generate(tokens, count, params, nullptr, out, 32u, report);
        check(ran, "the free run completes");

        char text[256];
        const core::u32 written = tokenizer.decode(out, report.generated, text, 255u);
        text[written] = '\0';
        std::printf("    free      -> \"%s\"\n", text);
        std::printf("    (weights derived from a seed have learnt nothing, so babble is the correct\n"
                    "     output; what the gate asserts is that both targets babble identically)\n");
    }

    std::printf("\n-- signatures the kernel must reproduce --\n");
    std::printf("  weight_sig      = 0x%08X\n", folded.weightSignature);
    std::printf("  prompt_sig      = 0x%08X\n", folded.promptSignature);
    std::printf("  logit_sig       = 0x%08X\n", folded.logitSignature);
    std::printf("  residual_sig    = 0x%08X\n", folded.residualSignature);
    std::printf("  token_sig       = 0x%08X\n", folded.tokenSignature);
    std::printf("  constrained_sig = 0x%08X\n", folded.constrainedSignature);
    std::printf("  text_sig        = 0x%08X\n", folded.textSignature);
    std::printf("  vocab           = %u\n", folded.vocabSize);
    std::printf("  prompt_tokens   = %u\n", folded.promptTokens);
    std::printf("  generated       = %u\n", folded.generated);
    std::printf("  draws           = %u\n", folded.draws);
    std::printf("  constrained     = %u\n", folded.constrainedTokens);
    std::printf("  admitted_first  = %u\n", folded.admittedFirst);
    std::printf("  grammar_done    = %u\n", folded.grammarComplete);
    std::printf("  forbidden       = %u\n", folded.forbiddenReachable);
    std::printf("  blob_bytes      = %u\n", folded.blobBytes);
    std::printf("  blob_reopened   = %u\n", folded.blobReopened);
    std::printf("  arena_bytes     = %u\n", folded.arenaBytes);

    std::printf("\n%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
