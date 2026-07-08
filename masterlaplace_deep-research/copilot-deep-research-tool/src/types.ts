export interface DeepResearchInput {
  query: string;
  scope?: string;
  preferLocal?: boolean;
}

export type ResearchJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface ResearchSource {
  title: string;
  url: string;
  snippet: string;
  provider: string;
  category?: "forum" | "docs" | "repo" | "wiki" | "paper" | "general";
  score?: number;
  publishedAt?: string;
}

export interface ResearchResult {
  summary: string;
  findings: string[];
  sources: ResearchSource[];
  backend: "local" | "cloud";
  limitations?: string[];
  providerDiagnostics?: Array<{
    provider: string;
    status: "ok" | "empty" | "error";
    detail?: string;
    count?: number;
  }>;
}

export interface ResearchJob {
  id: string;
  input: DeepResearchInput;
  createdAt: number;
  updatedAt: number;
  status: ResearchJobStatus;
  reportPath?: string;
  error?: string;
}

export interface DeepResearchConfig {
  backend: "local-first" | "cloud-only";
  maxSources: number;
  outputFolder: string;
  enableCloudFallback: boolean;
  pollIntervalSeconds: number;
  enableReaderProxy: boolean;
  readerProxyBaseUrl: string;
  targetDocDomains: string[];
  maxDocResultsPerDomain: number;
  requestTimeoutSeconds: number;
  enableVerboseLogging: boolean;
  localLlmEnabled: boolean;
  localLlmEndpoint: string;
  localLlmModel: string;
  localLlmMaxSources: number;
}
