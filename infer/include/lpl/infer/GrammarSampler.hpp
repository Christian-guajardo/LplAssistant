/**
 * @file GrammarSampler.hpp
 * @brief Constrained decoding against a grammar.
 *
 * The sampler masks every token the grammar forbids, so a malformed tool call is
 * not unlikely — it is unrepresentable. This is what makes a small local model
 * reliable enough to drive an engine.
 *
 * The constraint is applied at the BYTE level and not at the token level, and that
 * is the whole difficulty. A vocabulary segments text however its merges happen to
 * fall, so "generate_world" may arrive as three tokens or as seven, and a filter
 * that compared token identifiers against a list of allowed calls would accept a
 * different set of strings depending on how the tokenizer split them. Here a token
 * is admissible exactly when its bytes extend some phrase the grammar still
 * considers possible — which is a statement about the text, so it holds whatever
 * the segmentation.
 *
 * The accepted language is a finite set of literal phrases. That is less than GBNF
 * expresses and is enough for what the engine actually needs: @c agent::emitGbnf
 * exists to name a closed set of tool names and a closed set of enum values, and a
 * closed set is a finite language. Numeric ranges a context-free grammar cannot say
 * anyway, and are checked where they already are, in the call parser.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_INFER_GRAMMARSAMPLER_HPP
#    define LPL_LPL_INFER_GRAMMARSAMPLER_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

#        include <lpl/infer/TensorArena.hpp>
#        include <lpl/infer/Vocab.hpp>

namespace lpl::infer {

/**
 * @brief Phrases one grammar may accept.
 *
 * Thirty-two, so the still-possible set is one word and advancing the state is a
 * mask rather than a list walk. A tool surface wider than this is a sign the
 * grammar should be regenerated per step — which is what @c agent::emitGbnf already
 * does, offering only the calls that are valid right now.
 */
inline constexpr core::u32 kMaxGrammarPhrases = 32u;

/**
 * @struct GrammarState
 * @brief How far into the accepted language the output has got.
 */
struct GrammarState {
    core::u32 candidates{0u}; ///< Bit per phrase still consistent with what was emitted.
    core::u32 depth{0u};      ///< Bytes matched so far.
};

/**
 * @class Grammar
 * @brief A finite set of literal phrases, as a byte-level acceptor.
 */
class Grammar {
public:
    Grammar() = default;

    /**
     * @brief Copies the phrases into the arena.
     * @param arena   Storage.
     * @param phrases Null-terminated strings.
     * @param count   How many; above @ref kMaxGrammarPhrases is refused.
     * @return false when the count is out of range or the arena is exhausted.
     */
    bool build(TensorArena &arena, const char *const *phrases, core::u32 count);

    /**
     * @brief The state before anything has been emitted.
     * @return Every phrase possible, depth zero.
     */
    [[nodiscard]] GrammarState start() const noexcept;

    /**
     * @brief Would @p bytes keep the output inside the language?
     * @param state     Current state.
     * @param bytes     Candidate text.
     * @param length    Its length.
     * @param outNext   Receives the state after accepting, when it returns true.
     * @return true when at least one phrase still matches.
     */
    [[nodiscard]] bool accepts(const GrammarState &state, const char *bytes, core::u32 length,
                               GrammarState &outNext) const noexcept;

    /**
     * @brief Has a whole phrase been emitted?
     * @param state Current state.
     * @return true when some candidate ends exactly here.
     */
    [[nodiscard]] bool complete(const GrammarState &state) const noexcept;

    /**
     * @brief Phrases in the language.
     * @return The count.
     */
    [[nodiscard]] core::u32 size() const noexcept { return _count; }

private:
    const char *_blob{nullptr};
    const core::u32 *_offsets{nullptr};
    const core::u32 *_lengths{nullptr};
    core::u32 _count{0u};
};

/**
 * @brief Marks every token the grammar would accept next.
 *
 * @param grammar The language.
 * @param state   Current state.
 * @param vocab   The token table.
 * @param allowed Receives one flag per token.
 * @return Tokens admitted; zero means the language is exhausted.
 */
core::u32 maskAllowedTokens(const Grammar &grammar, const GrammarState &state, const Vocab &vocab,
                            bool *allowed) noexcept;

} // namespace lpl::infer

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_INFER_GRAMMARSAMPLER_HPP
