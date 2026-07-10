// laplace-research : CLI autonome du module deep research.
//
//   laplace-research "sujet à creuser" [--guidance "précisions"]
//   laplace-research --resume research_runs/<run>/
//
// LLM : LAPLACE_RESEARCH_LLM_URL (llama-server) prioritaire, sinon le modèle
// local LAPLACE_LLM_MODEL est chargé in-process (grammaire GBNF native).
#include "engine.h"
#include "llm_client.h"
#include "../config.h"
#include "../llm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

int main(int argc, char** argv) {
    using namespace laplace;
    using namespace laplace::research;

    std::string topic, guidance, resume_dir;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--resume") && i + 1 < argc) resume_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--guidance") && i + 1 < argc) guidance = argv[++i];
        else if (topic.empty()) topic = argv[i];
    }
    if (topic.empty() && resume_dir.empty()) {
        std::fprintf(stderr,
                     "usage : laplace-research \"sujet\" [--guidance \"...\"]\n"
                     "        laplace-research --resume <run_dir>\n");
        return 1;
    }

    try {
        ResearchOptions opt = ResearchOptions::from_env();

        // LLM : serveur HTTP si configuré, sinon modèle local in-process.
        std::unique_ptr<Llm> local;
        if (!std::getenv("LAPLACE_RESEARCH_LLM_URL")) {
            Config cfg = Config::from_env();
            std::fprintf(stderr, "[recherche] chargement du modèle local %s\n",
                         cfg.llm_model.c_str());
            local = std::make_unique<Llm>(cfg.llm_model, cfg.n_ctx, cfg.n_threads);
        }
        auto llm = make_llm_client(local.get());

        Engine engine(*llm, opt);
        std::string report = resume_dir.empty() ? engine.run(topic, guidance)
                                                : engine.resume(resume_dir);
        std::printf("%s\n", report.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[recherche] erreur fatale : %s\n", e.what());
        return 1;
    }
}
