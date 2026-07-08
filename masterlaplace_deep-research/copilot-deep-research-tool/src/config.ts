import * as vscode from "vscode";
import { DeepResearchConfig } from "./types";

const SECTION = "deepResearch";

export function getDeepResearchConfig(): DeepResearchConfig {
  const cfg = vscode.workspace.getConfiguration(SECTION);

  return {
    backend: cfg.get<"local-first" | "cloud-only">("backend", "local-first"),
    maxSources: cfg.get<number>("maxSources", 8),
    outputFolder: cfg.get<string>("outputFolder", ".github/research_reports"),
    enableCloudFallback: cfg.get<boolean>("enableCloudFallback", true),
    pollIntervalSeconds: cfg.get<number>("pollIntervalSeconds", 15),
    enableReaderProxy: cfg.get<boolean>("enableReaderProxy", true),
    readerProxyBaseUrl: cfg.get<string>("readerProxyBaseUrl", "https://r.jina.ai/"),
    targetDocDomains: cfg
      .get<string[]>("targetDocDomains", [
        "khronos.org",
        "nvidia.com",
        "developer.mozilla.org",
      ])
      .map((d) => d.trim().toLowerCase())
      .filter((d) => d.length > 0),
    maxDocResultsPerDomain: cfg.get<number>("maxDocResultsPerDomain", 2),
    requestTimeoutSeconds: cfg.get<number>("requestTimeoutSeconds", 10),
    enableVerboseLogging: cfg.get<boolean>("enableVerboseLogging", false),
    localLlmEnabled: cfg.get<boolean>("localLlmEnabled", true),
    localLlmEndpoint: cfg.get<string>("localLlmEndpoint", "http://localhost:11434"),
    localLlmModel: cfg.get<string>("localLlmModel", "mistral:7b"),
    localLlmMaxSources: cfg.get<number>("localLlmMaxSources", 8),
  };
}
