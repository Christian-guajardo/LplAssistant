/**
 * @file Parity.cpp
 * @brief The canonical mind case, folded stage by stage.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/infer/Parity.hpp>

#if defined(LPL_HAS_FOUNDATION)

#    include <lpl/infer/Inference.hpp>
#    include <lpl/infer/Tokenizer.hpp>

namespace lpl::infer {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Folds one 32-bit word into a running FNV-1a hash.
 * @param hash Running value.
 * @param word Word to absorb.
 */
void foldWord(core::u32 &hash, core::u32 word) noexcept { hash = (hash ^ word) * kFnv1aPrime; }

/// First and last printable ASCII byte the vocabulary spells one at a time.
constexpr char kFirstPrintable = 0x20;
constexpr char kLastPrintable = 0x7E;

/**
 * @brief The merge list.
 *
 * Fragments, never whole tool names. A merge that spelled a call in one token would
 * make the constrained run a single forced choice, and the gate would then say
 * nothing about a byte-level acceptor crossing a token boundary — which is the only
 * hard part of constrained decoding.
 */
const char *const kMerges[] = {"gen",  "erate", "_wor", "ld",    "place", "_ent", "ity",  "set",  "_bio",
                               "me",   "del",   "ete",  "_every", "thing", "tool", "args", "seed", "true",
                               "false", "null",  "\":\"", "\",\"", "{\"",  "\"}",  "00"};

constexpr core::u32 kMergeCount = static_cast<core::u32>(sizeof(kMerges) / sizeof(kMerges[0]));

/// The two calls the constrained run may spell.
const char *const kGrammarPhrases[] = {"generate_world", "place_entity"};

/**
 * @brief Length of a null-terminated string.
 * @param text The string.
 * @return Its length.
 */
core::u32 textLength(const char *text) noexcept
{
    core::u32 length = 0u;
    while (text[length] != '\0')
        ++length;
    return length;
}

} // namespace

const char *parityPrompt() noexcept { return "{\"tool\":\"gen"; }

const char *const *parityGrammarPhrases(core::u32 &outCount) noexcept
{
    outCount = static_cast<core::u32>(sizeof(kGrammarPhrases) / sizeof(kGrammarPhrases[0]));
    return kGrammarPhrases;
}

const char *parityForbiddenPhrase() noexcept { return "set_biome"; }

bool buildParityVocab(TensorArena &arena, Vocab &out)
{
    const core::u32 singles = static_cast<core::u32>(kLastPrintable - kFirstPrintable) + 1u + 1u;
    const core::u32 count = singles + kMergeCount;

    core::u32 blobBytes = singles;
    for (core::u32 i = 0u; i < kMergeCount; ++i)
        blobBytes += textLength(kMerges[i]);

    char *const blob = arena.claim<char>(blobBytes);
    VocabEntry *const entries = arena.claim<VocabEntry>(count);
    if (blob == nullptr || entries == nullptr)
        return false;

    core::u32 cursor = 0u;
    core::u32 token = 0u;
    for (char c = kFirstPrintable; c <= kLastPrintable; ++c)
    {
        blob[cursor] = c;
        entries[token] = VocabEntry{cursor, 1u};
        ++cursor;
        ++token;
    }
    blob[cursor] = '\n';
    entries[token] = VocabEntry{cursor, 1u};
    ++cursor;
    ++token;

    for (core::u32 i = 0u; i < kMergeCount; ++i)
    {
        const core::u32 length = textLength(kMerges[i]);
        for (core::u32 b = 0u; b < length; ++b)
            blob[cursor + b] = kMerges[i][b];
        entries[token] = VocabEntry{cursor, length};
        cursor += length;
        ++token;
    }

    return out.build(arena, blob, blobBytes, entries, count);
}

void foldMindState(MindFoldResult &out, void *memory, core::usize bytes)
{
    out = MindFoldResult{};

    TensorArena arena = memory != nullptr ? TensorArena{memory, bytes} : TensorArena{parityArenaBytes()};

    Vocab vocab;
    if (!buildParityVocab(arena, vocab))
        return;
    out.vocabSize = vocab.size();

    ModelConfig shape = parityModelConfig();
    shape.vocabSize = vocab.size();

    Model model;
    if (!model.synthesise(arena, shape, vocab, parityModelSeed()))
        return;
    out.weightSignature = model.fold(kFnv1aOffsetBasis);

    // The image, and the model read back out of it. Folding the reopened weights
    // against the derived ones is the claim: a loader that dropped a scale or
    // swapped two matrices would still produce a model that runs.
    out.blobBytes = model.blobBytes();
    core::u8 *const image = arena.claim<core::u8>(out.blobBytes);
    if (image != nullptr && model.writeBlob(image, out.blobBytes) == out.blobBytes)
    {
        Model reopened;
        if (reopened.open(arena, image, out.blobBytes) &&
            reopened.fold(kFnv1aOffsetBasis) == out.weightSignature)
            out.blobReopened = 1u;
    }

    Tokenizer tokenizer{vocab};
    const char *const prompt = parityPrompt();
    core::u32 promptTokens[64];
    core::u32 skipped = 0u;
    const core::u32 promptCount =
        tokenizer.encode(prompt, textLength(prompt), promptTokens, 64u, skipped);
    out.promptTokens = promptCount;

    core::u32 promptHash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < promptCount; ++i)
        foldWord(promptHash, promptTokens[i]);
    foldWord(promptHash, skipped);
    out.promptSignature = promptHash;

    Inference inference;
    if (!inference.initialise(arena, model, CachePolicy::Refuse))
        return;

    GenerationParams params{};
    params.sampler.temperature = math::Fixed32::one();
    params.sampler.topK = 4u;
    params.sampler.seed = paritySamplerSeed();
    params.maxTokens = parityGeneratedTokens();

    core::u32 generated[32];
    GenerationReport report{};
    if (!inference.generate(promptTokens, promptCount, params, nullptr, generated, 32u, report))
        return;

    out.generated = report.generated;
    out.draws = report.draws;

    core::u32 tokenHash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < report.generated; ++i)
        foldWord(tokenHash, generated[i]);
    out.tokenSignature = tokenHash;

    core::u32 logitHash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < inference.logits().count; ++i)
        foldWord(logitHash, static_cast<core::u32>(inference.logits().at(i).raw()));
    out.logitSignature = logitHash;

    core::u32 residualHash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < inference.residual().count; ++i)
        foldWord(residualHash, static_cast<core::u32>(inference.residual().at(i).raw()));
    out.residualSignature = residualHash;

    // The constrained run: same weights, same prompt, a language that can only spell
    // two calls.
    core::u32 phraseCount = 0u;
    const char *const *phrases = parityGrammarPhrases(phraseCount);
    Grammar grammar;
    if (!grammar.build(arena, phrases, phraseCount))
        return;

    out.admittedFirst = maskAllowedTokens(grammar, grammar.start(), vocab, arena.claim<bool>(vocab.size()));

    GenerationParams constrained = params;
    constrained.maxTokens = 24u;
    core::u32 constrainedTokens[32];
    GenerationReport constrainedReport{};
    if (!inference.generate(promptTokens, promptCount, constrained, &grammar, constrainedTokens, 32u,
                            constrainedReport))
        return;

    out.constrainedTokens = constrainedReport.generated;
    out.grammarComplete = constrainedReport.grammarComplete ? 1u : 0u;

    core::u32 constrainedHash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < constrainedReport.generated; ++i)
        foldWord(constrainedHash, constrainedTokens[i]);
    out.constrainedSignature = constrainedHash;

    char decoded[128];
    const core::u32 decodedBytes = tokenizer.decode(constrainedTokens, constrainedReport.generated, decoded, 128u);
    core::u32 textHash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < decodedBytes; ++i)
        foldWord(textHash, static_cast<core::u32>(static_cast<core::u8>(decoded[i])));
    foldWord(textHash, decodedBytes);
    out.textSignature = textHash;

    // Can the language reach the forbidden call at all? Walked byte by byte rather
    // than asserted: a mask that happened to be empty on this prompt would look like
    // a guarantee, and it would only be a coincidence.
    const char *const forbidden = parityForbiddenPhrase();
    const core::u32 forbiddenLength = textLength(forbidden);
    GrammarState walk = grammar.start();
    bool reachable = true;
    for (core::u32 i = 0u; i < forbiddenLength && reachable; ++i)
    {
        GrammarState next{};
        reachable = grammar.accepts(walk, forbidden + i, 1u, next);
        walk = next;
    }
    out.forbiddenReachable = (reachable && grammar.complete(walk)) ? 1u : 0u;

    out.arenaBytes = static_cast<core::u32>(arena.used());
}

} // namespace lpl::infer

#endif // LPL_HAS_FOUNDATION
