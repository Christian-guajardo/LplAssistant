import { ResearchResult } from "../types";

interface GeminiStartResponse {
  name?: string;
  state?: string;
}

interface GeminiInteractionResponse {
  state?: string;
  outputs?: Array<{
    text?: string;
  }>;
}

const BASE_URL = "https://generativelanguage.googleapis.com/v1beta";
const MODEL = "deep-research-pro-preview-12-2025";

function parseInteractionId(name: string | undefined): string | undefined {
  if (!name) {
    return undefined;
  }

  const split = name.split("/");
  return split[split.length - 1];
}

export async function startGeminiDeepResearch(
  query: string,
  apiKey: string
): Promise<string> {
  const url = `${BASE_URL}/interactions?key=${encodeURIComponent(apiKey)}`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      background: true,
      input: {
        text: query,
      },
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Gemini start failed: ${response.status} ${body}`);
  }

  const data = (await response.json()) as GeminiStartResponse;
  const interactionId = parseInteractionId(data.name);

  if (!interactionId) {
    throw new Error("Gemini did not return an interaction ID.");
  }

  return interactionId;
}

export async function checkGeminiInteraction(
  interactionId: string,
  apiKey: string
): Promise<{ done: boolean; failed: boolean; output?: string }> {
  const url = `${BASE_URL}/interactions/${encodeURIComponent(interactionId)}?key=${encodeURIComponent(apiKey)}`;

  const response = await fetch(url, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Gemini polling failed: ${response.status} ${body}`);
  }

  const data = (await response.json()) as GeminiInteractionResponse;
  const state = (data.state ?? "").toLowerCase();

  if (state === "failed") {
    return { done: true, failed: true };
  }

  if (state === "completed") {
    const output = data.outputs?.map((o) => o.text ?? "").filter((v) => v.trim().length > 0).join("\n\n");
    return { done: true, failed: false, output };
  }

  return { done: false, failed: false };
}

export async function runGeminiDeepResearch(
  query: string,
  apiKey: string,
  maxAttempts: number,
  pollIntervalMs: number
): Promise<ResearchResult> {
  const interactionId = await startGeminiDeepResearch(query, apiKey);

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const status = await checkGeminiInteraction(interactionId, apiKey);

    if (status.done && status.failed) {
      throw new Error("Gemini deep research task failed.");
    }

    if (status.done) {
      return {
        summary: "Cloud fallback completed using Gemini Deep Research.",
        findings: [status.output ?? "No output text available."],
        sources: [],
        backend: "cloud",
      };
    }

    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
  }

  throw new Error("Gemini deep research polling timed out.");
}
