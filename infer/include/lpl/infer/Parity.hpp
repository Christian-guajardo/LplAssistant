/**
 * @file Parity.hpp
 * @brief The constexpr prompt both sides run.
 *
 * Same weights, same prompt, same seed, same tokens — folded on the host oracle
 * and in ring 0. The gate that proves the demon is deterministic.
 *
 * The weights are DERIVED, not loaded, and that is the load-bearing decision. A gate
 * that shipped a blob to both sides would prove the two targets can read the same
 * bytes; deriving every tensor from a seed makes the gate about the forward pass
 * itself — the quantisation, the normalisation, the rotary angles, the exponential,
 * the sampling draw. The blob is exercised too, but as a separate claim: writing the
 * derived model out and reading it back must reproduce the same weight signature.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_PARITY_HPP
#    define LPL_LPL_INFER_PARITY_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/Model.hpp>

namespace lpl::infer {

/**
 * @brief The transformer both targets build.
 *
 * Sized for a four-mebibyte kernel heap and not for a workstation. Two blocks of
 * width forty-eight is a real transformer — attention, gating, residuals, rotary
 * positions — at about seventy-five kibibytes of weights, which is the largest thing
 * that can sit next to a running engine without the gate becoming a memory test.
 *
 * @return The shape; @c vocabSize is filled from the canonical vocabulary.
 */
[[nodiscard]] constexpr ModelConfig parityModelConfig() noexcept
{
    ModelConfig shape{};
    shape.dim = 48u;
    shape.layers = 2u;
    shape.heads = 4u;
    shape.ffnHidden = 96u;
    shape.contextLength = 32u;
    shape.vocabSize = 0u;
    return shape;
}

/**
 * @brief The seed every tensor is derived from.
 * @return The master seed.
 */
[[nodiscard]] constexpr core::u32 parityModelSeed() noexcept { return 0x1814u; }

/**
 * @brief The seed the sampler draws from.
 * @return The stream seed.
 */
[[nodiscard]] constexpr core::u32 paritySamplerSeed() noexcept { return 0xDEA1u; }

/**
 * @brief Bytes the canonical run claims.
 *
 * Everything the gate touches comes out of one arena: the weights, the image it is
 * written to, the model read back from that image, the cache and the scratch.
 *
 * @warning The bytes CONSUMED are a per-target measurement and NOT a cross-target
 * invariant, which is worth stating because the opposite is the obvious assumption
 * and it is wrong. A bump allocator's occupancy only matches across targets when
 * everything it carves is the same size on both — and @ref LayerWeights is not: it
 * holds nine views, each carrying a pointer, so it is 144 bytes on a 64-bit host and
 * 100 in ring 0. Two layers, in two models (the derived one and the one read back
 * from the image), is 176 bytes of difference before alignment; measured, the gate
 * reports 260877 on the host and 260693 in the kernel.
 *
 * What IS invariant is everything made of fixed-width words: the quantised tensors,
 * the cache, the vocabulary, and the size of the written image. Those are compared.
 * The occupancy is checked against the capacity on each side instead, which is the
 * question it can actually answer.
 *
 * @return The arena capacity.
 */
[[nodiscard]] constexpr core::usize parityArenaBytes() noexcept { return 512u * 1024u; }

/**
 * @brief The text fed to the model.
 * @return A null-terminated prompt.
 */
[[nodiscard]] const char *parityPrompt() noexcept;

/**
 * @brief Tokens the canonical generation may produce.
 * @return The ceiling.
 */
[[nodiscard]] constexpr core::u32 parityGeneratedTokens() noexcept { return 12u; }

/**
 * @brief The phrases the constrained run is allowed to spell.
 *
 * Two calls an engine offers and, deliberately, not a third: the gate asserts that
 * the language admits these and cannot express anything else, which is the claim
 * that makes a small model safe to wire to a command journal.
 *
 * @param outCount Receives the phrase count.
 * @return The phrases.
 */
[[nodiscard]] const char *const *parityGrammarPhrases(core::u32 &outCount) noexcept;

/**
 * @brief A call the grammar must NOT be able to spell.
 * @return The forbidden phrase.
 */
[[nodiscard]] const char *parityForbiddenPhrase() noexcept;

/**
 * @brief Builds the canonical vocabulary.
 *
 * Printable ASCII one byte at a time, a newline, and a short merge list. The merges
 * deliberately do NOT contain any whole tool name: if "generate_world" were one
 * token, the constrained run would be a single admissible choice and the gate would
 * prove nothing about a byte-level acceptor working across a segmentation.
 *
 * @param arena Storage.
 * @param out   Receives the table.
 * @return false when the arena is exhausted.
 */
bool buildParityVocab(TensorArena &arena, Vocab &out);

/**
 * @struct MindFoldResult
 * @brief The signatures the kernel must reproduce.
 *
 * Free of Fixed32 and bool, like every other fold result in the project: every field is a
 * word a test checks or records, and the kernel's records are compared with the host's.
 * @ref arenaBytes differs between targets, so it is checked against its capacity and never
 * recorded.
 */
struct MindFoldResult {
    core::u32 weightSignature{0u};      ///< Fold of every quantised tensor.
    core::u32 promptSignature{0u};      ///< Fold of the tokenised prompt.
    core::u32 logitSignature{0u};       ///< Fold of the scores at the last position.
    core::u32 residualSignature{0u};    ///< Fold of the residual stream at the last position.
    core::u32 tokenSignature{0u};       ///< Fold of the freely sampled tokens.
    core::u32 constrainedSignature{0u}; ///< Fold of the grammar-constrained tokens.
    core::u32 textSignature{0u};        ///< Fold of the bytes those tokens decode to.
    core::u32 vocabSize{0u};            ///< Tokens in the canonical table.
    core::u32 promptTokens{0u};         ///< Tokens the prompt segmented into.
    core::u32 generated{0u};            ///< Tokens the free run produced.
    core::u32 draws{0u};                ///< Random words the free run consumed.
    core::u32 constrainedTokens{0u};    ///< Tokens the constrained run produced.
    core::u32 admittedFirst{0u};        ///< Tokens the grammar allowed at its first step.
    core::u32 grammarComplete{0u};      ///< 1 when the constrained run spelled a whole phrase.
    core::u32 forbiddenReachable{0u};   ///< 1 when the forbidden phrase could be spelled — a failure.
    core::u32 blobBytes{0u};            ///< Size of the written image. Cross-target invariant.
    core::u32 blobReopened{0u};         ///< 1 when the image read back to the same weights.
    core::u32 arenaBytes{0u};           ///< Bytes carved. PER-TARGET — see @ref parityArenaBytes.
};

/**
 * @brief Runs the canonical case and folds every stage of it.
 *
 * @param out    Receives the signatures.
 * @param memory Optional block to run in; nullptr claims one from the allocator.
 * @param bytes  Size of @p memory, ignored when it is null.
 */
void foldMindState(MindFoldResult &out, void *memory = nullptr, core::usize bytes = 0u);

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_PARITY_HPP
