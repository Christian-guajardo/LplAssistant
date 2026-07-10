#pragma once
#include <string>

#include "llm_client.h"
#include "state.h"

namespace laplace::research {
struct ResearchOptions;

// Rédige le rapport final dans <run_dir>/report.md :
// plan de sections généré, rédaction progressive section par section
// (star-hengxing : jamais un long rapport en un seul appel), références,
// quality gates URL (HEAD -> GET, offline-aware), diagnostics providers
// et limites. Renvoie le chemin du fichier.
std::string write_report(RunState& st, LlmClient& llm, const ResearchOptions& opt);

} // namespace laplace::research
