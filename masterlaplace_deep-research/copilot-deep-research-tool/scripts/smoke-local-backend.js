/* eslint-disable no-console */
const { runLocalDeepResearch } = require("../out/backends/localBackend");

async function main() {
  const query = process.argv.slice(2).join(" ") || "Vulkan synchronization Intel Arc";
  const scope = process.env.SMOKE_SCOPE || undefined;

  const result = await runLocalDeepResearch(
    query,
    scope,
    {
      maxSources: 10,
      enableReaderProxy: true,
      readerProxyBaseUrl: "https://r.jina.ai/",
      targetDocDomains: ["khronos.org", "nvidia.com", "developer.mozilla.org"],
      maxDocResultsPerDomain: 2,
      requestTimeoutSeconds: 10,
      localLlmEnabled: true,
      localLlmEndpoint: "http://localhost:11434",
      localLlmModel: "mistral:7b",
      localLlmMaxSources: 8,
    },
    {
      isCancellationRequested: false,
      onCancellationRequested: () => ({ dispose() {} }),
    }
  );

  console.log("=== Smoke Test Result ===");
  console.log(`Backend: ${result.backend}`);
  console.log(`Summary: ${result.summary}`);
  console.log(`Sources: ${result.sources.length}`);
  const providerSet = new Set(result.sources.map((s) => s.provider));
  console.log(`Distinct providers in selected sources: ${providerSet.size}`);
  const withExtract = result.sources.filter((s) => (s.extractedText || "").length > 80).length;
  console.log(`Sources with useful extracted text: ${withExtract}`);
  console.log("Top sources:");
  for (const source of result.sources.slice(0, 5)) {
    console.log(`- [${source.provider}] ${source.title} (${source.url})`);
  }

  if (result.providerDiagnostics?.length) {
    console.log("Diagnostics:");
    for (const diag of result.providerDiagnostics) {
      console.log(`- ${diag.provider}: ${diag.status} (${diag.count ?? 0}) ${diag.detail ?? ""}`.trim());
    }
  }
}

main().catch((error) => {
  console.error("Smoke test failed:", error?.message || error);
  process.exit(1);
});
