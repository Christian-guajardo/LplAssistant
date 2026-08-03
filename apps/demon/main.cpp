/**
 * @file main.cpp
 * @brief The demon, hosted: same mind, an operating system underneath.
 *
 * Development target for what runs in ring 0 on the server profile. Identical
 * modules, a different platform beneath them. When the hosted path and the
 * freestanding one disagree, one of them is wrong and the fold says which — which
 * is the only reason to keep both.
 *
 * The body below is the API written from the caller's side, before the callee
 * exists — the cheapest way to find out whether a library is pleasant to use. It
 * is fenced out until the modules it names are implemented, and the entry point
 * fails loudly rather than returning success it has not earned.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/core/Assert.hpp>

#include <lpl/infer/Inference.hpp>
#include <lpl/infer/Model.hpp>
#include <lpl/infer/TensorArena.hpp>
#include <lpl/mind/Budget.hpp>
#include <lpl/mind/Dialogue.hpp>
#include <lpl/mind/Memory.hpp>
#include <lpl/mind/Persona.hpp>
#include <lpl/mind/ReAct.hpp>

int main(int argc, char **argv)
{
    (void) argc;
    (void) argv;

#if 0 // ── intended usage ─────────────────────────────────────────────────────
    // Claimed once, never grown. On the server profile this is the memory the
    // display would have taken; here it simply comes from the host allocator.
    lpl::infer::TensorArena arena{lpl::infer::TensorArena::megabytes(2048)};

    const lpl::infer::Model model = lpl::infer::Model::map(argv[1], arena);
    lpl::infer::Inference inference{model, arena};

    // Not a fine-tune: a personality baked into weights cannot be reviewed, diffed
    // or reverted, and the sovereign must be able to do all three.
    lpl::mind::Persona persona = lpl::mind::Persona::load("persona/laplace.toml");

    // Long memory without retraining. The demon writes its own notes, and because
    // they are text the sovereign can read and correct them.
    lpl::mind::Memory memory{lpl::mind::Memory::store("memory/")};

    lpl::mind::ReAct loop{inference, persona, memory};
    lpl::mind::Dialogue channel = lpl::mind::Dialogue::onStdio();

    while (const auto intent = channel.receive())
    {
        // Bounded in tokens, latency and arena bytes. A demon that thinks past its
        // deadline has already failed, however good the answer.
        const auto reply = loop.run(*intent, lpl::mind::Budget::interactive());
        channel.send(reply);
    }

    return 0;
#endif // ─────────────────────────────────────────────────────────────────────

    LPL_NOT_IMPLEMENTED("lpl-demon (hosted)");
}
