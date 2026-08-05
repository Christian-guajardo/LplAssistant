/**
 * @file Intent.hpp
 * @brief What the sovereign asked for, parsed and bounded.
 *
 * The one input that cannot be derived from world state. It is given the same care as
 * an untrusted packet, because from inside the world that is what it is.
 *
 * Concretely that is three decisions rather than three habits. The text is capped at a
 * size chosen HERE and not by the sender. Bytes outside the printable range are dropped
 * and COUNTED, instead of being carried into a prompt where a control character could
 * end a line the grammar was in the middle of. And the kind is derived from a leading
 * keyword, so a caller cannot assert "this is a command" — it has to say something that
 * reads as one.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_MIND_INTENT_HPP
#    define LPL_LPL_MIND_INTENT_HPP

#    include <lpl/Foundation.hpp>

#    if defined(LPL_HAS_FOUNDATION)

namespace lpl::mind {

/// Bytes of an intent that survive parsing.
inline constexpr core::u32 kIntentBytes = 192u;

/**
 * @enum IntentKind
 * @brief What the sovereign is doing by speaking.
 *
 * Separate from the text because the demon treats them differently: a Stop abandons
 * work in flight and has to be recognised before anything else is done with the bytes,
 * and a Correction is addressed to a previous turn rather than to the world.
 */
enum class IntentKind : core::u8 {
    Unknown,    ///< Nothing in it names what is wanted.
    Question,   ///< An answer is expected, and no action.
    Command,    ///< The world is to be changed.
    Correction, ///< The last turn was wrong.
    Stop,       ///< Abandon whatever is running.
};

/**
 * @struct Intent
 * @brief One utterance from the sovereign, after it has been made safe.
 */
struct Intent {
    IntentKind kind{IntentKind::Unknown}; ///< What it is asking for.
    char text[kIntentBytes]{};            ///< The surviving bytes.
    core::u32 bytes{0u};                  ///< How many.
    core::u32 topic{0u};                  ///< Hash of the leading word, for the structured filter.
    core::u32 droppedBytes{0u};           ///< Bytes refused as unacceptable.
    core::u32 truncatedBytes{0u};         ///< Bytes lost to the cap.
};

/**
 * @brief Parses an utterance into a bounded intent.
 *
 * Never fails. An utterance that is entirely unacceptable becomes an empty
 * @ref IntentKind::Unknown with every byte counted in @ref Intent::droppedBytes — a
 * result the caller can see and act on, where an error code would tempt it to drop the
 * question and carry on as though nothing had been said.
 *
 * @param bytes Utterance, as received; may be null when @p count is zero.
 * @param count How many bytes.
 * @return The parsed intent.
 */
[[nodiscard]] Intent parseIntent(const core::u8 *bytes, core::u32 count) noexcept;

/**
 * @brief Hashes a word the way the topic filter does.
 *
 * Exposed because a caller that files a memory under a topic and a caller that looks
 * one up must agree on what a topic IS, and two hashes of one word is the oldest way
 * for a lookup to find nothing while everything looks correct.
 *
 * @param text  Word bytes.
 * @param count How many.
 * @return The topic identifier; 0 for an empty word.
 */
[[nodiscard]] core::u32 topicOf(const char *text, core::u32 count) noexcept;

/**
 * @brief Folds an intent into a signature.
 *
 * @param intent Intent to fold.
 * @param hash   Running value.
 * @return The updated hash.
 */
[[nodiscard]] core::u32 foldIntent(const Intent &intent, core::u32 hash) noexcept;

} // namespace lpl::mind

#    endif // LPL_HAS_FOUNDATION

#endif // LPL_LPL_MIND_INTENT_HPP
