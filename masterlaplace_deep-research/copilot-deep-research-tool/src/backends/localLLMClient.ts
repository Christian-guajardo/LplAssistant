import { ResearchSource } from "../types";

interface OllamaTagsResponse {
  models?: Array<{
    name: string;
    size: number;
    modified_at?: string;
  }>;
}

interface OllamaGenerateResponse {
  response: string;
  done: boolean;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  eval_count?: number;
}

export interface OllamaModel {
  name: string;
  size: number;
  sizeGb: string;
}

export async function listOllamaModels(endpoint: string): Promise<OllamaModel[]> {
  const base = endpoint.endsWith("/") ? endpoint.slice(0, -1) : endpoint;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${base}/api/tags`, {
      signal: controller.signal,
      headers: { "User-Agent": "copilot-deep-research-tool" },
    });

    if (!response.ok) return [];

    const data = (await response.json()) as OllamaTagsResponse;
    return (data.models ?? [])
      .map((m) => ({
        name: m.name,
        size: m.size,
        sizeGb: (m.size / 1e9).toFixed(1) + " GB",
      }))
      .sort((a, b) => b.size - a.size);
  } catch {
    return [];
  } finally {
    clearTimeout(timer);
  }
}

export async function detectBestModel(
  endpoint: string,
  configuredModel: string
): Promise<string> {
  const models = await listOllamaModels(endpoint);
  if (models.length === 0) return configuredModel;

  // Exact match on configured model wins
  const exactMatch = models.find(
    (m) => m.name === configuredModel || m.name.startsWith(configuredModel + ":")
  );
  if (exactMatch) return exactMatch.name;

  // Fallback: first available model (sorted by size, largest first)
  console.log(
    `[deep-research] Configured model "${configuredModel}" not found. ` +
    `Available: ${models.map((m) => m.name).join(", ")}. Using "${models[0].name}".`
  );
  return models[0].name;
}

async function callOllama(
  model: string,
  prompt: string,
  endpoint: string,
  timeoutMs: number
): Promise<string> {
  const base = endpoint.endsWith("/") ? endpoint.slice(0, -1) : endpoint;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${base}/api/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "User-Agent": "copilot-deep-research-tool",
      },
      body: JSON.stringify({
        model,
        prompt,
        stream: false,
        options: {
          temperature: 0.3,
          num_predict: 512,
        },
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(
        `Ollama returned HTTP ${response.status}: ${body.slice(0, 200)}`
      );
    }

    const data = (await response.json()) as OllamaGenerateResponse;
    if (!data.response?.trim()) {
      throw new Error("Ollama returned empty response.");
    }

    return data.response.trim();
  } finally {
    clearTimeout(timer);
  }
}

async function tryCallOllama(
  model: string,
  prompt: string,
  endpoint: string,
  timeoutMs: number
): Promise<string | null> {
  try {
    return await callOllama(model, prompt, endpoint, timeoutMs);
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    if (msg.includes("ECONNREFUSED") || msg.includes("fetch failed")) {
      return null; // Ollama not running
    }
    console.warn(`[deep-research] LLM call failed: ${msg}`);
    return null;
  }
}

function buildMapPrompt(source: ResearchSource, query: string): string {
  const truncatedSnippet = source.snippet.slice(0, 800);
  return `You are a precise research analyst. Extract the 2-3 most important factual findings from this content that are directly relevant to the research query below. 
Write each finding as ONE concise sentence. Do NOT add opinions, interpretations, or filler text. 
If the content is not relevant to the query, respond with "NOT_RELEVANT".

QUERY: ${query}
SOURCE: ${source.title} (${source.provider}, ${source.category ?? "general"})
CONTENT: ${truncatedSnippet}

FINDINGS:`;
}

function buildReducePrompt(
  query: string,
  allFindings: string,
  sourceCount: number
): string {
  return `You are an expert research analyst writing an executive summary for a technical audience.

Write a comprehensive research summary (250-350 words) based on the collected findings below. Structure it as:

## Overview
[2-3 sentences contextualizing the topic and why it matters]

## Key Findings
[4-6 bullet points organizing the most important discoveries by theme. Group related findings together. Include specific technical details where available.]

## Gaps & Limitations
[1-2 sentences noting what aspects of the query were NOT covered by the collected sources]

RESEARCH QUERY: ${query}
NUMBER OF SOURCES ANALYZED: ${sourceCount}

COLLECTED FINDINGS:
${allFindings}

Write the summary now (Markdown format, 250-350 words total):`;
}

async function mapPhase(
  sources: ResearchSource[],
  query: string,
  endpoint: string,
  model: string,
  timeoutPerCall: number
): Promise<string[]> {
  const results: string[] = [];
  for (const source of sources) {
    const prompt = buildMapPrompt(source, query);
    const result = await tryCallOllama(model, prompt, endpoint, timeoutPerCall);
    if (result && result.trim() !== "NOT_RELEVANT" && result.length > 10) {
      results.push(`[From "${source.title}"] ${result}`);
    } else if (source.snippet.length > 20) {
      results.push(`[From "${source.title}"] ${source.snippet.slice(0, 300)}`);
    }
  }
  return results;
}

async function reducePhase(
  findings: string[],
  query: string,
  sourceCount: number,
  endpoint: string,
  model: string,
  timeoutMs: number
): Promise<string | null> {
  const combined = findings.join("\n\n---\n\n");
  const prompt = buildReducePrompt(query, combined, sourceCount);
  return tryCallOllama(model, prompt, endpoint, timeoutMs);
}

function buildFallbackRichSummary(
  sources: ResearchSource[],
  findings: string[],
  query: string
): string {
  const providerNames = [...new Set(sources.map((s) => s.provider))];
  const topSources = sources.slice(0, 5).map((s) => `- **${s.title}** (${s.provider}): ${s.snippet.slice(0, 150)}...`).join("\n");

  return `## Research Summary: ${query}

### Overview
This research collected ${sources.length} sources across ${providerNames.length} providers (${providerNames.join(", ")}). A statistical synthesis was performed as the local LLM was unavailable.

### Key Findings
${findings.slice(0, 6).join("\n")}

### Top Sources
${topSources}

### Methodology Note
The findings above are ranked by a composite score incorporating provider authority, recency, and keyword relevance. Enable a local LLM (Ollama) for deeper multi-pass synthesis.`;
}

export async function synthesizeWithLocalLLM(
  sources: ResearchSource[],
  existingFindings: string[],
  query: string,
  options: {
    enabled: boolean;
    endpoint: string;
    model: string;
    maxSources: number;
  }
): Promise<string> {
  if (!options.enabled) {
    return buildFallbackRichSummary(sources, existingFindings, query);
  }

  const sourcesToSummarize = sources.slice(0, options.maxSources).filter(
    (s) => s.snippet.length > 30
  );

  if (sourcesToSummarize.length < 2) {
    return buildFallbackRichSummary(sources, existingFindings, query);
  }

  // Auto-detect best available model, falling back silently if Ollama unreachable
  const resolvedModel = await detectBestModel(options.endpoint, options.model);
  const effectiveModel = resolvedModel !== options.model ? resolvedModel : options.model;
  if (effectiveModel !== options.model) {
    console.log(`[deep-research] Auto-selected model: "${effectiveModel}" (configured "${options.model}" not found)`);
  }

  const MAP_TIMEOUT = 30000; // 30 sec per source (sequential, avoids Ollama concurrency timeouts)
  const REDUCE_TIMEOUT = 60000; // 60 sec for final synthesis

  // Sequential map: Ollama handles one request at a time efficiently
  const findings = await mapPhase(
    sourcesToSummarize,
    query,
    options.endpoint,
    effectiveModel,
    MAP_TIMEOUT
  );

  if (findings.length === 0) {
    return buildFallbackRichSummary(sources, existingFindings, query);
  }

  const synthesis = await reducePhase(
    findings,
    query,
    sourcesToSummarize.length,
    options.endpoint,
    effectiveModel,
    REDUCE_TIMEOUT
  );

  if (synthesis && synthesis.length > 80) {
    return synthesis;
  }

  return buildFallbackRichSummary(sources, existingFindings, query);
}
