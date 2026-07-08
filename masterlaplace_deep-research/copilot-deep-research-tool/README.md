# Deep Research Tool for VS Code Copilot

A **local-first**, **asynchronous** deep research tool integrated with GitHub Copilot Chat. Searches multiple authoritative sources (forums, documentation, scientific papers, code repositories) and writes structured markdown reports into your workspace.

## Features

- **Multi-provider research pipeline** (10 distinct sources):
  - Forums: StackOverflow, Reddit, GitHub Issues
  - Documentation: GitHub READMEs/Releases, targeted domain docs (Khronos, NVIDIA, MDN)
  - Scientific: OpenAlex (open-access), arXiv
  - Code: GitHub repositories
  - Reference: Wikipedia
  - General web: DuckDuckGo Lite + reader proxy (+ fallback for edge cases)

- **🧠 Local LLM synthesis (Map-Reduce via Ollama)**:
  - **Map phase**: Each source summarized independently by a local model (parallel, ~3 sec each)
  - **Reduce phase**: All findings synthesized into a structured report (~6 sec)
  - **Auto-detection**: Scans installed Ollama models, picks the best one automatically
  - **Graceful fallback**: Falls back to statistical summary if Ollama is unavailable
  - **Total latency**: ~30 sec added for Gemini-quality reports, fully local & private

- **Quality-focused ranking**:
  - Provider weight system (papers & academic sources ranked high, but reduced for non-academic queries)
  - Lexical relevance matching (query terms vs source content)
  - Recency boost (newer sources score higher)
  - Category balancing (prevents "all papers" bias)

- **Intelligent enrichment**:
  - Reader proxy (Jina) to extract clean text from dynamic pages (React, Vue, JavaScript)
  - PDF candidate detection (direct .pdf URLs + arXiv abs→pdf conversion)
  - Markdown extraction from noisy web pages
  - Fallback general search when primary sources are scarce

- **Operational transparency**:
  - Per-provider diagnostics (ok / empty / error with counts)
  - Limitations report (which providers failed and why)
  - Fully traceable source metadata

- **Configurable via VS Code settings**:
  - `deepResearch.backend`: local-first or cloud-only
  - `deepResearch.maxSources`: target number of results (default: 8)
  - `deepResearch.enableReaderProxy`: toggle cleaner text extraction (default: true)
  - `deepResearch.readerProxyBaseUrl`: custom reader proxy endpoint (default: https://r.jina.ai/)
  - `deepResearch.targetDocDomains`: list of doc domains to search (default: khronos.org, nvidia.com, developer.mozilla.org)
  - `deepResearch.maxDocResultsPerDomain`: per-domain result cap (default: 2)
  - `deepResearch.localLlmEnabled`: enable local LLM map-reduce synthesis (default: true)
  - `deepResearch.localLlmEndpoint`: Ollama API endpoint (default: http://localhost:11434)
  - `deepResearch.localLlmModel`: Ollama model for synthesis (default: mistral:7b, auto-detected if unavailable)
  - `deepResearch.localLlmMaxSources`: max sources fed to LLM synthesis (default: 8)
  - `deepResearch.requestTimeoutSeconds`: HTTP timeout per provider (default: 10)
  - `deepResearch.enableVerboseLogging`: detailed debug logging (default: false)

## Usage

In Copilot Chat, reference the tool with `#deep_research`:

```
#deep_research Find recent Vulkan sync best practices and Intel Arc driver issues
```

Or use the command palette:

```
Deep Research: Show Jobs
Deep Research: Cancel Job
Deep Research: Re-run Job
Deep Research: Configure Local LLM Model
Deep Research: Configure Gemini API Key
```

## Reports

Research outputs are written to `.github/research_reports/` as timestamped markdown files with:

- Executive summary
- Top 5 findings with provider attribution
- Full source list with URLs and extracted text
- Provider diagnostics
- Known limitations

## Architecture

- **Extension** (`src/extension.ts`): VS Code activation, command registration
- **Tool Interface** (`src/researchTool.ts`): Copilot Chat integration
- **Async Orchestration** (`src/taskManager.ts`): Job queue, persistence, report generation
- **Local Backend** (`src/backends/localBackend.ts`): Multi-provider fetch pipeline (10 providers)
- **LLM Synthesis** (`src/backends/localLLMClient.ts`): Map-reduce synthesis via Ollama with auto-detection
- **Cloud Fallback** (`src/backends/geminiClient.ts`): Gemini Interactions API (optional)
- **Config/Types** (`src/config.ts`, `src/types.ts`): Settings management, type contracts

## Testing

Run smoke tests to validate local backend quality:

```bash
npm run smoke:local
npm run smoke:local:query -- "your custom query here"
```

Expected outcomes: 4–10 sources from 2–4 distinct providers, diagnostics show which sources succeeded/failed/were empty.

## Providers & Coverage

| Provider | Type | Status | Notes |
|----------|------|--------|-------|
| Wikipedia | Reference | Active | API-based, reliable |
| StackOverflow | Forum | Active | Tag-based filtering |
| GitHub Repos | Code | Active | Stars-sorted |
| GitHub Issues | Forum | Active | Updated-sorted, great for troubleshooting |
| GitHub Docs | Docs | Active | README + releases |
| Domain Docs | Docs | Active | Site-targeted + reader proxy |
| Web General | Web | Active | DDG Lite + reader proxy |
| Reddit | Forum | Active | May return 403 per rate limits |
| OpenAlex | Science | Active | Open-access papers only (filter: is_oa:true) |
| arXiv | Science | Active | All research papers |
| Web Fallback | Web | Active | Triggers only if primary sources < 50% target |

## Known Limitations

- **Web dynamic content**: Reader proxy helps but isn't guaranteed on all sites (Cloudflare, advanced anti-bot). Fallback strategy activates when coverage is low.
- **Reddit endpoint**: Public API sometimes returns 403 per rate limits or IP restrictions.
- **PDF extraction**: Fallback to plain text approximation via reader proxy; no advanced layout parsing or OCR.
- **Large documents**: Context windows limit full ingestion of massive PDFs (workaround: reader proxy clips to 360 chars per source).
- **GitHub API rate limits**: Unauthenticated requests limited to 60/hour; OAuth with token can reach 5000/hour.

## 🧠 Local LLM Synthesis (Map-Reduce)

The extension can use a local LLM via **Ollama** to produce Gemini-quality research reports through a two-phase process:

### How it works

```
┌─────────────────────────────────────────┐
│  MAP PHASE (parallel)                    │
│  Source 1 → LLM → 2-3 key findings      │
│  Source 2 → LLM → 2-3 key findings      │
│  ... (up to 8 sources × ~3 sec each)    │
├─────────────────────────────────────────┤
│  REDUCE PHASE                            │
│  All findings → LLM → structured report │
│  (~6 sec, produces ~300 words)          │
└─────────────────────────────────────────┘
```

### Setup

```bash
# Install Ollama and pull a model
ollama pull mistral:7b       # Great balance (4.1 GB)
ollama pull llama3.1:8b      # Best quality (5.1 GB)
ollama pull phi3:mini         # Fastest (2.3 GB)
```

### Auto-detection

The extension automatically scans installed Ollama models via `GET /api/tags`. If the configured model is not found, the largest available model is used instead. If Ollama is not running, it falls back gracefully to a statistical summary.

### Manual configuration

`Ctrl+Shift+P` → `Deep Research: Configure Local LLM Model`

This opens a QuickPick listing all installed models with their sizes. Pick one and it's saved in VS Code settings.

### Settings

```json
{
  "deepResearch.localLlmEnabled": true,
  "deepResearch.localLlmEndpoint": "http://localhost:11434",
  "deepResearch.localLlmModel": "mistral:7b",
  "deepResearch.localLlmMaxSources": 8
}
```

Set `localLlmEnabled: false` to always use the statistical fallback (faster, less detailed).

## Performance Notes

- Typical research run: 3–8 seconds (parallel providers)
- Reader proxy enrichment: +500ms per source (configurable)
- PDF extraction: +300ms per PDF candidate (cached per run)
- Local LLM synthesis (map-reduce): +25-40 sec (8 sources × ~3 sec map + ~6 sec reduce)
- Network timeouts: 10 seconds per provider (configurable)

## Future Improvements

- Hardened provider retry logic (2-attempt backoff already in place)
- OAuth-based GitHub API to increase rate limits
- Local vector indexing for large knowledge bases
- Integration with MCP (Model Context Protocol) for broader tool ecosystem
- Advanced PDF parsing (layout-aware extraction, figure/table detection)
- Custom domain allowlists and blocklists

## License

MIT

## Contributing

Found a bug or have a feature request? Open an issue or check the project structure for potential improvements.
