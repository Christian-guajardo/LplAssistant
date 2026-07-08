import * as vscode from "vscode";
import { ResearchResult, ResearchSource } from "../types";
import { synthesizeWithLocalLLM } from "./localLLMClient";

interface WikipediaSearchResponse {
  query?: {
    search?: Array<{
      title: string;
      snippet: string;
      pageid: number;
    }>;
  };
}

interface StackExchangeSearchResponse {
  items?: Array<{
    title: string;
    link: string;
    excerpt?: string;
    creation_date?: number;
  }>;
}

interface GithubSearchResponse {
  items?: Array<{
    full_name: string;
    html_url: string;
    description?: string;
    updated_at?: string;
  }>;
}

interface GithubReadmeResponse {
  content?: string;
  encoding?: string;
  html_url?: string;
}

interface GithubIssueSearchResponse {
  items?: Array<{
    title: string;
    html_url: string;
    body?: string;
    updated_at?: string;
  }>;
}

interface GithubReleaseResponse {
  name?: string;
  body?: string;
  html_url?: string;
  published_at?: string;
}

interface RedditSearchResponse {
  data?: {
    children?: Array<{
      data?: {
        title?: string;
        permalink?: string;
        selftext?: string;
        created_utc?: number;
      };
    }>;
  };
}

interface OpenAlexSearchResponse {
  results?: Array<{
    display_name?: string;
    publication_year?: number;
    publication_date?: string;
    id?: string;
    primary_location?: {
      landing_page_url?: string;
      pdf_url?: string;
      source?: {
        display_name?: string;
      };
    };
    abstract_inverted_index?: Record<string, number[]>;
  }>;
}

interface ProviderOutcome {
  provider: string;
  status: "ok" | "empty" | "error";
  detail?: string;
  sources: ResearchSource[];
  count?: number;
}

interface DuckDuckGoLiteResult {
  title: string;
  url: string;
}

export interface LocalBackendOptions {
  maxSources: number;
  enableReaderProxy: boolean;
  readerProxyBaseUrl: string;
  targetDocDomains: string[];
  maxDocResultsPerDomain: number;
  requestTimeoutSeconds: number;
  // Local LLM synthesis
  localLlmEnabled: boolean;
  localLlmEndpoint: string;
  localLlmModel: string;
  localLlmMaxSources: number;
}

const DEFAULT_TIMEOUT_MS = 10000;

const PROVIDER_WEIGHTS: Record<string, number> = {
  openalex: 1.0,
  arxiv: 1.0,
  stackoverflow: 0.95,
  github_docs: 0.95,
  domain_docs: 0.95,
  web_general: 0.92,
  github_issues: 0.9,
  github: 0.9,
  reddit: 0.75,
  wikipedia: 0.7,
  reader_proxy: 0.8,
};

const RETRYABLE_HTTP_CODES = new Set([408, 425, 429, 500, 502, 503, 504]);
const STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "best",
  "for",
  "from",
  "how",
  "in",
  "is",
  "it",
  "of",
  "on",
  "or",
  "that",
  "the",
  "to",
  "with",
]);
const GENERIC_TERMS = new Set([
  "best",
  "practice",
  "practices",
  "guide",
  "tutorial",
  "overview",
  "tips",
  "using",
]);

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  const msg = error.message.toLowerCase();
  if (msg.includes("aborted") || msg.includes("timeout") || msg.includes("network")) {
    return true;
  }

  const match = msg.match(/http\s+(\d{3})/i);
  if (!match) {
    return false;
  }

  const status = Number(match[1]);
  return RETRYABLE_HTTP_CODES.has(status);
}

async function withRetry<T>(run: () => Promise<T>, maxAttempts = 2): Promise<T> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await run();
    } catch (error) {
      lastError = error;
      const shouldRetry = attempt < maxAttempts && isRetryableError(error);
      if (!shouldRetry) {
        throw error;
      }

      await delay(250 * attempt);
    }
  }

  throw lastError instanceof Error ? lastError : new Error("Unknown provider failure.");
}

function clip(value: string, max = 320): string {
  const trimmed = value.trim();
  if (trimmed.length <= max) {
    return trimmed;
  }

  return `${trimmed.slice(0, max - 1)}...`;
}

function stripHtml(value: string): string {
  return value.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
}

function buildScore(source: ResearchSource): number {
  const weight = PROVIDER_WEIGHTS[source.provider] ?? 0.6;
  const lengthScore = Math.min(1, source.snippet.length / 220);
  return Number((weight * 0.7 + lengthScore * 0.3 + computeRecencyBoost(source) * 0.2).toFixed(3));
}

function tokenize(value: string): string[] {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length >= 3 && !STOPWORDS.has(token));
}

function keywordOverlapScore(query: string, source: ResearchSource): number {
  const queryTokens = new Set(tokenize(query));
  if (queryTokens.size === 0) {
    return 0;
  }

  const sourceTokens = new Set(tokenize(`${source.title} ${source.snippet}`));
  let overlap = 0;
  for (const token of queryTokens) {
    if (sourceTokens.has(token)) {
      overlap += 1;
    }
  }

  return overlap / queryTokens.size;
}

function extractCoreTerms(query: string, max = 6): string[] {
  return tokenize(query)
    .filter((token) => !GENERIC_TERMS.has(token))
    .slice(0, max);
}

function buildProviderQuery(query: string): string {
  const core = extractCoreTerms(query, 6);
  if (core.length >= 2) {
    return core.join(" ");
  }

  return query;
}

function hasStrongQueryMatch(query: string, source: ResearchSource): boolean {
  const strongTerms = extractCoreTerms(query, 5);
  if (strongTerms.length === 0) {
    return true;
  }

  const sourceTokens = new Set(tokenize(`${source.title} ${source.snippet}`));
  return strongTerms.some((term) => sourceTokens.has(term));
}

function isPaperHeavyIntent(query: string): boolean {
  const q = query.toLowerCase();
  return /\b(arxiv|paper|research|publication|scientific|academic|study)\b/.test(q);
}

function computeRecencyBoost(source: ResearchSource): number {
  if (!source.publishedAt) {
    return 0;
  }

  const publishedMs = Date.parse(source.publishedAt);
  if (Number.isNaN(publishedMs)) {
    return 0;
  }

  const ageDays = (Date.now() - publishedMs) / (1000 * 60 * 60 * 24);
  if (ageDays <= 7) {
    return 1;
  }
  if (ageDays <= 30) {
    return 0.75;
  }
  if (ageDays <= 180) {
    return 0.4;
  }
  if (ageDays <= 365) {
    return 0.2;
  }

  return 0;
}

function selectWithDiversity(sources: ResearchSource[], maxSources: number): ResearchSource[] {
  const sorted = [...sources].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const providerCount = new Map<string, number>();
  const selected: ResearchSource[] = [];
  const providerCap = Math.max(1, Math.ceil(maxSources / 3));

  for (const source of sorted) {
    if (selected.length >= maxSources) {
      break;
    }

    const count = providerCount.get(source.provider) ?? 0;
    if (count >= providerCap) {
      continue;
    }

    selected.push(source);
    providerCount.set(source.provider, count + 1);
  }

  if (selected.length < maxSources) {
    const selectedUrls = new Set(selected.map((s) => s.url));
    for (const source of sorted) {
      if (selected.length >= maxSources) {
        break;
      }
      if (selectedUrls.has(source.url)) {
        continue;
      }

      selected.push(source);
      selectedUrls.add(source.url);
    }
  }

  return selected;
}

function rebalanceCategories(
  sources: ResearchSource[],
  maxSources: number,
  query: string
): ResearchSource[] {
  const allowPaperHeavy = isPaperHeavyIntent(query);
  const categoryCap = new Map<string, number>();
  categoryCap.set("paper", allowPaperHeavy ? maxSources : Math.max(1, Math.floor(maxSources * 0.3)));

  const sorted = [...sources].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const selected: ResearchSource[] = [];
  const categoryCount = new Map<string, number>();

  for (const source of sorted) {
    if (selected.length >= maxSources) {
      break;
    }

    const category = source.category ?? "general";
    const cap = categoryCap.get(category);
    const current = categoryCount.get(category) ?? 0;
    if (cap !== undefined && current >= cap) {
      continue;
    }

    selected.push(source);
    categoryCount.set(category, current + 1);
  }

  if (selected.length < maxSources) {
    const selectedUrls = new Set(selected.map((s) => s.url));
    for (const source of sorted) {
      if (selected.length >= maxSources) {
        break;
      }
      if (selectedUrls.has(source.url)) {
        continue;
      }

      selected.push(source);
      selectedUrls.add(source.url);
    }
  }

  return selected;
}

function decodeOpenAlexAbstract(index?: Record<string, number[]>): string {
  if (!index) {
    return "";
  }

  const tokens: Array<{ token: string; pos: number }> = [];
  for (const [token, positions] of Object.entries(index)) {
    for (const pos of positions) {
      tokens.push({ token, pos });
    }
  }

  tokens.sort((a, b) => a.pos - b.pos);
  return tokens.map((t) => t.token).join(" ");
}

async function fetchJson<T>(url: string | URL, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      ...init,
      signal: controller.signal,
      headers: {
        "User-Agent": "copilot-deep-research-tool",
        Accept: "application/json",
        ...(init?.headers ?? {}),
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return (await response.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

async function searchWikipedia(query: string, limit: number): Promise<ResearchSource[]> {
  const url = new URL("https://en.wikipedia.org/w/api.php");
  url.searchParams.set("action", "query");
  url.searchParams.set("list", "search");
  url.searchParams.set("format", "json");
  url.searchParams.set("srsearch", query);
  url.searchParams.set("srlimit", String(limit));
  url.searchParams.set("origin", "*");

  const data = await fetchJson<WikipediaSearchResponse>(url);
  const rows = data.query?.search ?? [];

  return rows.map((item) => ({
    title: item.title,
    url: `https://en.wikipedia.org/?curid=${item.pageid}`,
    snippet: clip(stripHtml(item.snippet)),
    provider: "wikipedia",
    category: "wiki",
  }));
}

async function searchStackExchange(query: string, limit: number): Promise<ResearchSource[]> {
  const url = new URL("https://api.stackexchange.com/2.3/search/advanced");
  url.searchParams.set("order", "desc");
  url.searchParams.set("sort", "relevance");
  url.searchParams.set("q", query);
  url.searchParams.set("site", "stackoverflow");
  url.searchParams.set("pagesize", String(limit));
  url.searchParams.set("filter", "!-*jbN-o8P3E5");

  const data = await fetchJson<StackExchangeSearchResponse>(url);
  const rows = data.items ?? [];

  return rows.map((item) => ({
    title: stripHtml(item.title),
    url: item.link,
    snippet: clip(stripHtml(item.excerpt ?? "")),
    provider: "stackoverflow",
    category: "forum",
    publishedAt: item.creation_date
      ? new Date(item.creation_date * 1000).toISOString()
      : undefined,
  }));
}

async function searchGithub(query: string, limit: number): Promise<ResearchSource[]> {
  const url = new URL("https://api.github.com/search/repositories");
  url.searchParams.set("q", query);
  url.searchParams.set("sort", "stars");
  url.searchParams.set("order", "desc");
  url.searchParams.set("per_page", String(limit));

  const data = await fetchJson<GithubSearchResponse>(url, {
    headers: {
      Accept: "application/vnd.github+json",
    },
  });
  const rows = data.items ?? [];

  return rows.map((item) => ({
    title: item.full_name,
    url: item.html_url,
    snippet: clip(item.description?.trim() ?? "No repository description available."),
    provider: "github",
    category: "repo",
    publishedAt: item.updated_at,
  }));
}

async function searchGithubIssues(query: string, limit: number): Promise<ResearchSource[]> {
  const url = new URL("https://api.github.com/search/issues");
  url.searchParams.set("q", `${query} is:issue`);
  url.searchParams.set("sort", "updated");
  url.searchParams.set("order", "desc");
  url.searchParams.set("per_page", String(limit));

  const data = await fetchJson<GithubIssueSearchResponse>(url, {
    headers: {
      Accept: "application/vnd.github+json",
    },
  });

  const rows = data.items ?? [];
  return rows.map((item) => ({
    title: item.title,
    url: item.html_url,
    snippet: clip(item.body?.replace(/\s+/g, " ").trim() ?? "GitHub issue discussion."),
    provider: "github_issues",
    category: "forum",
    publishedAt: item.updated_at,
  }));
}

async function searchGithubDocs(query: string, limit: number): Promise<ResearchSource[]> {
  const url = new URL("https://api.github.com/search/repositories");
  url.searchParams.set("q", query);
  url.searchParams.set("sort", "stars");
  url.searchParams.set("order", "desc");
  url.searchParams.set("per_page", String(Math.max(2, Math.min(5, limit))));

  const data = await fetchJson<GithubSearchResponse>(url, {
    headers: {
      Accept: "application/vnd.github+json",
    },
  });

  const repos = data.items ?? [];
  const docSources: ResearchSource[] = [];

  for (const repo of repos.slice(0, limit)) {
    const fullName = repo.full_name;
    if (!fullName) {
      continue;
    }

    try {
      const readme = await fetchJson<GithubReadmeResponse>(
        `https://api.github.com/repos/${fullName}/readme`,
        {
          headers: {
            Accept: "application/vnd.github+json",
          },
        }
      );

      let snippet = "README fetched but empty.";
      if (readme.content && readme.encoding === "base64") {
        const decoded = Buffer.from(readme.content, "base64").toString("utf8");
        snippet = clip(decoded.replace(/\s+/g, " ").trim(), 360);
      }

      docSources.push({
        title: `${fullName} README`,
        url: readme.html_url ?? `${repo.html_url}#readme`,
        snippet,
        provider: "github_docs",
        category: "docs",
        publishedAt: repo.updated_at,
      });
    } catch {
      // Optional enrichment, ignore readme fetch failure.
    }

    try {
      const latest = await fetchJson<GithubReleaseResponse>(
        `https://api.github.com/repos/${fullName}/releases/latest`,
        {
          headers: {
            Accept: "application/vnd.github+json",
          },
        }
      );

      if (latest.html_url) {
        docSources.push({
          title: `${fullName} release: ${latest.name ?? "latest"}`,
          url: latest.html_url,
          snippet: clip((latest.body ?? "Latest release notes.").replace(/\s+/g, " ").trim(), 360),
          provider: "github_docs",
          category: "docs",
          publishedAt: latest.published_at,
        });
      }
    } catch {
      // Latest release can be missing for many repos.
    }
  }

  return docSources;
}

async function searchReddit(query: string, limit: number): Promise<ResearchSource[]> {
  const url = new URL("https://www.reddit.com/search.json");
  url.searchParams.set("q", query);
  url.searchParams.set("sort", "relevance");
  url.searchParams.set("limit", String(limit));

  const data = await fetchJson<RedditSearchResponse>(url, {
    headers: {
      Accept: "application/json",
    },
  });

  const rows = data.data?.children ?? [];
  return rows
    .map((row) => row.data)
    .filter((row): row is NonNullable<typeof row> => Boolean(row?.title && row?.permalink))
    .map((row) => ({
      title: row.title ?? "Untitled reddit thread",
      url: `https://www.reddit.com${row.permalink}`,
      snippet: clip(stripHtml(row.selftext ?? "Reddit thread discussing this topic.")),
      provider: "reddit",
      category: "forum",
      publishedAt: row.created_utc
        ? new Date(row.created_utc * 1000).toISOString()
        : undefined,
    }));
}

async function searchOpenAlex(query: string, limit: number): Promise<ResearchSource[]> {
  const url = new URL("https://api.openalex.org/works");
  url.searchParams.set("search", query);
  url.searchParams.set("filter", "is_oa:true");
  url.searchParams.set("per-page", String(limit));

  const data = await fetchJson<OpenAlexSearchResponse>(url);
  const rows = data.results ?? [];

  return rows.map((row) => {
    const abstract = clip(decodeOpenAlexAbstract(row.abstract_inverted_index));
    const landing = row.primary_location?.landing_page_url;
    const pdfUrl = row.primary_location?.pdf_url;

    return {
      title: row.display_name ?? "Untitled open-access paper",
      url: pdfUrl ?? landing ?? row.id ?? "https://openalex.org",
      snippet:
        abstract.length > 0
          ? abstract
          : clip(`Open-access work from ${row.primary_location?.source?.display_name ?? "unknown source"}.`),
      provider: "openalex",
      category: "paper",
      publishedAt: row.publication_date ?? (row.publication_year ? `${row.publication_year}-01-01` : undefined),
    };
  });
}

async function searchArxiv(query: string, limit: number): Promise<ResearchSource[]> {
  const url = new URL("https://export.arxiv.org/api/query");
  url.searchParams.set("search_query", `all:${query}`);
  url.searchParams.set("start", "0");
  url.searchParams.set("max_results", String(limit));

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        "User-Agent": "copilot-deep-research-tool",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const xml = await response.text();
    const entries = xml.match(/<entry>[\s\S]*?<\/entry>/g) ?? [];

    return entries.slice(0, limit).map((entry) => {
      const title = (entry.match(/<title>([\s\S]*?)<\/title>/)?.[1] ?? "Untitled arXiv paper").replace(/\s+/g, " ").trim();
      const summary = (entry.match(/<summary>([\s\S]*?)<\/summary>/)?.[1] ?? "").replace(/\s+/g, " ").trim();
      const id = (entry.match(/<id>([\s\S]*?)<\/id>/)?.[1] ?? "https://arxiv.org").trim();
      const published = (entry.match(/<published>([\s\S]*?)<\/published>/)?.[1] ?? "").trim();

      return {
        title,
        url: id,
        snippet: clip(summary),
        provider: "arxiv",
        category: "paper",
        publishedAt: published || undefined,
      };
    });
  } finally {
    clearTimeout(timeout);
  }
}

async function enrichWithReaderProxy(
  source: ResearchSource,
  readerProxyBaseUrl: string
): Promise<ResearchSource> {
  const normalizedBase = readerProxyBaseUrl.endsWith("/")
    ? readerProxyBaseUrl
    : `${readerProxyBaseUrl}/`;
  const proxyUrl = `${normalizedBase}${source.url}`;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

  try {
    const response = await fetch(proxyUrl, {
      signal: controller.signal,
      headers: {
        "User-Agent": "copilot-deep-research-tool",
      },
    });

    if (!response.ok) {
      return source;
    }

    const text = await response.text();
    const cleaned = clip(text.replace(/\s+/g, " ").trim(), 360);
    if (cleaned.length === 0) {
      return source;
    }

    return {
      ...source,
      snippet: cleaned,
      provider: source.provider,
    };
  } catch {
    return source;
  } finally {
    clearTimeout(timeout);
  }
}

function normalizeReaderProxyBase(base: string): string {
  return base.endsWith("/") ? base : `${base}/`;
}

function tryBuildPdfCandidate(source: ResearchSource): string | undefined {
  const url = source.url;
  if (/\.pdf([?#].*)?$/i.test(url)) {
    return url;
  }

  if (/arxiv\.org\/abs\//i.test(url)) {
    return `${url.replace(/\/abs\//i, "/pdf/")}.pdf`;
  }

  return undefined;
}

async function enrichPdfSources(
  sources: ResearchSource[],
  readerProxyBaseUrl: string
): Promise<ResearchSource[]> {
  const normalizedBase = normalizeReaderProxyBase(readerProxyBaseUrl);
  const enriched: ResearchSource[] = [];

  for (const source of sources) {
    const pdfUrl = tryBuildPdfCandidate(source);
    if (!pdfUrl) {
      enriched.push(source);
      continue;
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
    try {
      const response = await fetch(`${normalizedBase}${pdfUrl}`, {
        signal: controller.signal,
        headers: {
          "User-Agent": "copilot-deep-research-tool",
        },
      });

      if (!response.ok) {
        enriched.push(source);
        continue;
      }

      const text = await response.text();
      const cleaned = clip(text.replace(/\s+/g, " ").trim(), 380);
      if (cleaned.length === 0) {
        enriched.push(source);
        continue;
      }

      enriched.push({
        ...source,
        snippet: cleaned,
      });
    } catch {
      enriched.push(source);
    } finally {
      clearTimeout(timeout);
    }
  }

  return enriched;
}

function decodeDuckDuckGoRedirect(url: string): string {
  try {
    const parsed = new URL(url, "https://duckduckgo.com");
    const uddg = parsed.searchParams.get("uddg");
    return uddg ? decodeURIComponent(uddg) : parsed.toString();
  } catch {
    return url;
  }
}

async function searchDuckDuckGoLite(
  query: string,
  maxResults: number
): Promise<DuckDuckGoLiteResult[]> {
  const form = new URLSearchParams();
  form.set("q", query);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

  try {
    const response = await fetch("https://html.duckduckgo.com/html/", {
      method: "POST",
      body: form,
      signal: controller.signal,
      headers: {
        "User-Agent": "copilot-deep-research-tool",
        "Content-Type": "application/x-www-form-urlencoded",
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const html = await response.text();
    const matches = [...html.matchAll(/<a[^>]*class=\"result__a\"[^>]*href=\"([^\"]+)\"[^>]*>([\s\S]*?)<\/a>/gi)];
    const results: DuckDuckGoLiteResult[] = [];

    for (const match of matches) {
      if (results.length >= maxResults) {
        break;
      }

      const href = decodeDuckDuckGoRedirect(match[1]);
      if (!href.startsWith("http")) {
        continue;
      }

      results.push({
        title: clip(stripHtml(match[2]), 120),
        url: href,
      });
    }

    return results;
  } finally {
    clearTimeout(timeout);
  }
}

async function searchDomainDocs(
  query: string,
  domains: string[],
  maxPerDomain: number,
  readerProxyBaseUrl: string
): Promise<ResearchSource[]> {
  if (domains.length === 0) {
    return [];
  }

  const normalizedBase = normalizeReaderProxyBase(readerProxyBaseUrl);
  const sources: ResearchSource[] = [];

  for (const domain of domains) {
    const results = await searchDuckDuckGoLite(
      `site:${domain} ${query}`,
      Math.max(1, maxPerDomain * 2)
    );

    const filtered = results
      .filter((r) => r.url.toLowerCase().includes(domain.toLowerCase()))
      .slice(0, maxPerDomain);

    for (const result of filtered) {
      const fallbackSource: ResearchSource = {
        title: result.title || `${domain} documentation`,
        url: result.url,
        snippet: clip(`Documentation page from ${domain} potentially relevant to: ${query}`, 220),
        provider: "domain_docs",
        category: "docs",
      };

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
      try {
        const response = await fetch(`${normalizedBase}${result.url}`, {
          signal: controller.signal,
          headers: {
            "User-Agent": "copilot-deep-research-tool",
          },
        });

        if (!response.ok) {
          sources.push(fallbackSource);
          continue;
        }

        const text = await response.text();
        const snippet = clip(text.replace(/\s+/g, " ").trim(), 360);
        if (snippet.length === 0) {
          sources.push(fallbackSource);
          continue;
        }

        sources.push({
          ...fallbackSource,
          snippet,
        });
      } catch {
        sources.push(fallbackSource);
      } finally {
        clearTimeout(timeout);
      }
    }
  }

  return sources;
}

async function searchWebGeneral(
  query: string,
  limit: number,
  readerProxyBaseUrl: string
): Promise<ResearchSource[]> {
  const candidates = await searchDuckDuckGoLite(query, Math.max(3, limit * 3));
  const normalizedBase = normalizeReaderProxyBase(readerProxyBaseUrl);
  const sources: ResearchSource[] = [];

  for (const candidate of candidates) {
    if (sources.length >= limit) {
      break;
    }

    const fallbackSource: ResearchSource = {
      title: candidate.title || "Web result",
      url: candidate.url,
      snippet: clip(`Web result relevant to: ${query}`, 220),
      provider: "web_general",
      category: "general",
    };

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
    try {
      const response = await fetch(`${normalizedBase}${candidate.url}`, {
        signal: controller.signal,
        headers: {
          "User-Agent": "copilot-deep-research-tool",
        },
      });

      if (!response.ok) {
        sources.push(fallbackSource);
        continue;
      }

      const text = await response.text();
      const snippet = clip(text.replace(/\s+/g, " ").trim(), 360);
      if (snippet.length === 0) {
        sources.push(fallbackSource);
        continue;
      }

      sources.push({
        ...fallbackSource,
        snippet,
      });
    } catch {
      sources.push(fallbackSource);
    } finally {
      clearTimeout(timeout);
    }
  }

  return sources;
}

async function searchFallbackGeneral(
  query: string,
  limit: number,
  readerProxyBaseUrl: string
): Promise<ResearchSource[]> {
  const candidates = await searchDuckDuckGoLite(query, Math.max(3, limit * 4));
  const normalizedBase = normalizeReaderProxyBase(readerProxyBaseUrl);
  const sources: ResearchSource[] = [];

  for (const candidate of candidates) {
    if (sources.length >= limit) {
      break;
    }

    const fallbackSource: ResearchSource = {
      title: candidate.title || "General web result",
      url: candidate.url,
      snippet: clip(`Web result for: ${query}`, 200),
      provider: "web_fallback",
      category: "general",
    };

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
    try {
      const response = await fetch(`${normalizedBase}${candidate.url}`, {
        signal: controller.signal,
        headers: {
          "User-Agent": "copilot-deep-research-tool",
        },
      });

      if (!response.ok) {
        sources.push(fallbackSource);
        continue;
      }

      const text = await response.text();
      const snippet = clip(text.replace(/\s+/g, " ").trim(), 320);
      if (snippet.length === 0) {
        sources.push(fallbackSource);
        continue;
      }

      sources.push({
        ...fallbackSource,
        snippet,
      });
    } catch {
      sources.push(fallbackSource);
    } finally {
      clearTimeout(timeout);
    }
  }

  return sources;
}

async function collectProvider(
  provider: string,
  run: () => Promise<ResearchSource[]>
): Promise<ProviderOutcome> {
  try {
    const sources = await withRetry(run, 2);
    if (sources.length === 0) {
      return {
        provider,
        status: "empty",
        detail: "No relevant items returned.",
        sources,
      };
    }

    return {
      provider,
      status: "ok",
      count: sources.length,
      detail: "Collected successfully.",
      sources,
    };
  } catch (error) {
    return {
      provider,
      status: "error",
      detail: error instanceof Error ? error.message : "Unknown provider error.",
      sources: [],
    };
  }
}

function dedupeSources(sources: ResearchSource[]): ResearchSource[] {
  const seen = new Set<string>();
  const deduped: ResearchSource[] = [];

  for (const source of sources) {
    if (seen.has(source.url)) {
      continue;
    }

    seen.add(source.url);
    deduped.push(source);
  }

  return deduped;
}

export async function runLocalDeepResearch(
  query: string,
  scope: string | undefined,
  options: LocalBackendOptions,
  token: vscode.CancellationToken
): Promise<ResearchResult> {
  if (token.isCancellationRequested) {
    throw new Error("Research cancelled before start.");
  }

  const effectiveQuery = scope ? `${query} ${scope}` : query;
  const providerQuery = buildProviderQuery(effectiveQuery);
  const perProvider = Math.max(2, Math.ceil(options.maxSources / 6));

  const outcomes = await Promise.all([
    collectProvider("wikipedia", () => searchWikipedia(providerQuery, perProvider)),
    collectProvider("stackoverflow", () => searchStackExchange(providerQuery, perProvider)),
    collectProvider("github", () => searchGithub(providerQuery, perProvider)),
    collectProvider("github_issues", () => searchGithubIssues(providerQuery, perProvider)),
    collectProvider("github_docs", () => searchGithubDocs(providerQuery, perProvider)),
    collectProvider("domain_docs", () =>
      searchDomainDocs(
        providerQuery,
        options.targetDocDomains,
        options.maxDocResultsPerDomain,
        options.readerProxyBaseUrl
      )
    ),
    collectProvider("web_general", () =>
      searchWebGeneral(providerQuery, perProvider, options.readerProxyBaseUrl)
    ),
    collectProvider("reddit", () => searchReddit(providerQuery, perProvider)),
    collectProvider("openalex", () => searchOpenAlex(providerQuery, perProvider)),
    collectProvider("arxiv", () => searchArxiv(providerQuery, perProvider)),
  ]);

  if (token.isCancellationRequested) {
    throw new Error("Research cancelled during collection.");
  }

  const primarySourceCount = dedupeSources(
    outcomes.flatMap((outcome) => outcome.sources)
  ).length;

  if (primarySourceCount < Math.ceil(options.maxSources / 2)) {
    const fallbackOutcome = await collectProvider("web_fallback", () =>
      searchFallbackGeneral(
        providerQuery,
        Math.max(2, options.maxSources - primarySourceCount),
        options.readerProxyBaseUrl
      )
    );
    outcomes.push(fallbackOutcome);
  }

  const providerDiagnostics = outcomes.map((o) => ({
    provider: o.provider,
    status: o.status,
    detail: o.detail,
    count: o.sources.length,
  }));

  let allSources: ResearchSource[] = dedupeSources(
    outcomes.flatMap((outcome) => outcome.sources)
  ).map((source) => ({
    ...source,
    score: buildScore(source),
  }));

  allSources = allSources
    .map((source) => {
      const overlap = keywordOverlapScore(effectiveQuery, source);
      return {
        ...source,
        score: Number(((source.score ?? 0) + overlap * 0.6).toFixed(3)),
      };
    })
    .filter((source) => {
      const overlap = keywordOverlapScore(effectiveQuery, source);
      const strongMatch = hasStrongQueryMatch(effectiveQuery, source);
      if (source.category === "paper" && !isPaperHeavyIntent(effectiveQuery) && !strongMatch) {
        return false;
      }

      return overlap >= 0.1 || (source.score ?? 0) >= 0.85;
    });

  allSources = selectWithDiversity(allSources, options.maxSources);
  allSources = rebalanceCategories(allSources, options.maxSources, effectiveQuery);

  if (options.enableReaderProxy) {
    const enrichedCount = Math.min(3, allSources.length);
    const enriched = await Promise.all(
      allSources.slice(0, enrichedCount).map((source) =>
        enrichWithReaderProxy(source, options.readerProxyBaseUrl)
      )
    );
    allSources = [...enriched, ...allSources.slice(enrichedCount)].map((source) => ({
      ...source,
      score: buildScore(source),
    }));
  }

  allSources = await enrichPdfSources(allSources, options.readerProxyBaseUrl);
  allSources = allSources.map((source) => ({
    ...source,
    score: buildScore(source),
  }));

  if (allSources.length === 0) {
    throw new Error("No local sources could be collected.");
  }

  const findings = allSources.slice(0, 5).map((source, index) => {
    const fact =
      source.snippet.length > 0
        ? source.snippet
        : "No summary snippet available.";
    return `F${index + 1}: [${source.provider}] ${source.title} -> ${fact}`;
  });

  const limitations = providerDiagnostics
    .filter((d) => d.status !== "ok")
    .map(
      (d) =>
        `${d.provider}: ${
          d.status === "empty"
            ? "no relevant items"
            : d.detail ?? "collection error"
        }`
    );

  const statisticalSummary = buildDynamicSummary(allSources, findings, providerDiagnostics, effectiveQuery);

  const summary = await synthesizeWithLocalLLM(allSources, findings, effectiveQuery, {
    enabled: options.localLlmEnabled,
    endpoint: options.localLlmEndpoint,
    model: options.localLlmModel,
    maxSources: options.localLlmMaxSources,
  });

  return {
    summary: summary.length > 50 ? summary : statisticalSummary,
    findings,
    sources: allSources,
    backend: "local",
    limitations,
    providerDiagnostics,
  };
}

function buildDynamicSummary(
  sources: ResearchSource[],
  findings: string[],
  diagnostics: Array<{ provider: string; status: string; detail?: string; count?: number }>,
  query: string
): string {
  const providerCounts = new Map<string, { count: number; ok: boolean }>();
  for (const d of diagnostics) {
    providerCounts.set(d.provider, {
      count: d.count ?? 0,
      ok: d.status === "ok",
    });
  }

  const categoryCounts = new Map<string, string[]>();
  for (const source of sources) {
    const cat = source.category || "general";
    const existing = categoryCounts.get(cat) ?? [];
    existing.push(source.title);
    categoryCounts.set(cat, existing);
  }

  const activeProviders = [...providerCounts.entries()]
    .filter(([, info]) => info.ok && info.count > 0)
    .map(([name, info]) => `${name} (${info.count})`);

  const providerSummary =
    activeProviders.length > 0
      ? `Data was successfully collected from ${activeProviders.length} provider${
          activeProviders.length > 1 ? "s" : ""
        }: ${activeProviders.join(", ")}.`
      : "Data collection was attempted across multiple providers.";

  const categorySummaries: string[] = [];
  if (categoryCounts.has("paper")) {
    const papers = categoryCounts.get("paper")!;
    categorySummaries.push(
      `${papers.length} research paper${papers.length > 1 ? "s" : ""} were found covering this topic.`
    );
  }
  if (categoryCounts.has("docs")) {
    const docs = categoryCounts.get("docs")!;
    categorySummaries.push(
      `${docs.length} official documentation source${docs.length > 1 ? "s" : ""} were identified.`
    );
  }
  if (categoryCounts.has("forum")) {
    const forums = categoryCounts.get("forum")!;
    categorySummaries.push(
      `${forums.length} community discussion${forums.length > 1 ? "s" : ""} from forums and issue trackers were included.`
    );
  }
  if (categoryCounts.has("general")) {
    const web = categoryCounts.get("general")!;
    categorySummaries.push(
      `${web.length} general web source${web.length > 1 ? "s" : ""} with relevant information were retrieved.`
    );
  }

  const topTitles = sources.slice(0, 3).map((s) => `"${s.title}"`);
  const keySourcesLine =
    topTitles.length > 0
      ? `Key sources include ${topTitles.join(", ")}.`
      : "";

  const categoryBreakdown =
    categorySummaries.length > 0
      ? `Source breakdown: ${categorySummaries.join(" ")}`
      : "";

  const failedProviders = [...providerCounts.entries()]
    .filter(([, info]) => !info.ok)
    .map(([name]) => name);

  const limitationNote =
    failedProviders.length > 0
      ? `Note: ${failedProviders.join(
          ", "
        )} could not be reached or returned no relevant results (see Limitations section).`
      : "";

  const queryContext =
    query.length > 0
      ? `This research was conducted in response to the query: "${query.slice(
          0,
          120
        )}${query.length > 120 ? "..." : ""}".`
      : "";

  return [
    queryContext,
    providerSummary,
    categoryBreakdown,
    keySourcesLine,
    limitationNote,
    "",
    "The findings below present the most relevant and highly-ranked results, ordered by a composite score that accounts for provider authority, recency, and keyword relevance to the query.",
  ]
    .filter((line) => line.length > 0)
    .join(" ");
}
