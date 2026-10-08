#include <lpl/mind/Parity.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(reasoning);

/**
 * @brief Gate P17 reasoning: the canonical turn of gate P16 agency, every move chosen by the
 *        transformer. Inside a grammar it never names an action the world did not offer and
 *        finishes the job; the same model, ungoverned, never names a legal one, which is what makes
 *        the first zero a measurement rather than a reassurance.
 *
 * @details The arena's occupancy is checked against its capacity, not recorded: it differs between
 *          a 64-bit host and i686.
 */
LPL_TEST(the_grammar_makes_an_illegal_action_unspellable)
{
    lpl::mind::ReasoningFoldResult folded{};

    lpl::mind::foldReasoning(folded);
    test.check(folded.generations > 0u, "the model really was consulted");
    test.check(folded.illegalActions == 0u, "not one action the world had not offered");
    test.check(folded.freeAttempts > 0u, "the unconstrained control actually ran");
    test.check(folded.freeLegalNames == 0u, "and the same model, ungoverned, never once named a legal action");
    test.check(folded.satisfied == 1u, "constrained, it finishes the job");
    test.check(folded.arenaBytes > 0u && folded.arenaBytes < lpl::mind::parityReasoningArenaBytes(),
               "the run fits the arena it declared");

    lpl::mind::ReasoningFoldResult again{};

    lpl::mind::foldReasoning(again);
    test.check(again.transcriptSignature == folded.transcriptSignature &&
                   again.actionSignature == folded.actionSignature,
               "two runs of the model-driven turn decide the same things");

    test.measureHexadecimal("transcript_signature", folded.transcriptSignature);
    test.measureHexadecimal("action_signature", folded.actionSignature);
    test.measureHexadecimal("utterance_signature", folded.utteranceSignature);
    test.measure("generations", folded.generations);
    test.measure("completions", folded.completions);
    test.measure("grammar_exhausted", folded.grammarExhausted);
    test.measure("tokens_generated", folded.tokensGenerated);
    test.measure("transcript_lines", folded.transcriptLines);
    test.measure("steps_spent", folded.stepsSpent);
    test.measure("free_attempts", folded.freeAttempts);
}
