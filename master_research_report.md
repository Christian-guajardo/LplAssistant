# Master Research Report: Deep Research & Agentic Architectures
**Target Project:** LplAssistant (Jarvis)
**Date:** 2026-07-08

## 1. Executive Summary
This report synthesizes methodologies from 15 state-of-the-art Deep Research and Agentic frameworks (Alibaba, DeepMind, Langchain, Jina AI, etc.) to inform the architecture of **Jarvis**—a high-performance, bare-metal C++ personal assistant. The core vision for Jarvis is 100% local, headless, ultra-optimized execution using `llama.cpp`.

## 2. Core Architectural Paradigms
Across the industry, "Deep Research" has evolved from simple RAG into iterative, multi-agent state machines.
- **Planner-Executor-Publisher (STORM):** The user query is given to a "Planner" that breaks it down into sub-queries. "Executor" agents research these sub-queries in parallel. A "Publisher" compiles the findings.
- **IterDRAG / Reflective Loops:** Agents perform a search, summarize it, and *explicitly reflect* to identify "Knowledge Gaps". These gaps become the prompt for the next loop.
- **Persistent State Machines:** Long-running loops often crash or hallucinate. The best CLI implementations (like `star-hengxing/deep-research`) save a JSON state checkpoint at every step, allowing users to pause, resume, or inject new directions mid-flight.

**Jarvis Implementation:** Jarvis must adopt an **Iterative State Machine** with persistent checkpoints. For speed, the C++ daemon should spawn parallel threads for sub-query execution (Planner-Executor).

## 3. The Inference Engine (`llama.cpp`)
To fulfill the "bare-metal / no API" requirement, `llama.cpp` is the definitive choice.
- **`llama-server` API:** Rather than linking C++ libraries manually, Jarvis can spawn the built-in `llama-server`. It provides an OpenAI-compatible API over localhost, supporting multi-user parallel decoding natively.
- **GBNF Grammars:** A massive problem with small local models is malformed JSON output, which breaks state machines. `llama.cpp` supports GBNF (Grammar-Based Normal Form), which physically constrains the model's logits to output 100% valid JSON. **This is mandatory for Jarvis.**
- **Speculative Decoding:** To maximize tokens/sec locally, Jarvis should use a "Draft" model (e.g., 1.5B parameters) to speculatively decode for a larger reasoning model.

## 4. Memory & Context Optimization (The Token Bottleneck)
Local inference is heavily constrained by context window length and generation speed. The longer the recursive loop, the slower it gets.
- **Continual Learning (DeepMind):** Instead of one massive prompt, use an ensemble of local classifiers stored in memory, retrieved via K-Nearest Neighbors (K-NN). Jarvis already uses `pgvector`; we must evolve it to retrieve "behavior shards" dynamically based on the current context.
- **ESON & CCR (Honey-for-devs):** 
  - **ESON:** Use Efficient Structured Object Notation (ESON) instead of JSON for internal state tracking, halving context usage.
  - **CCR (Compress-Cache-Retrieve):** When Jarvis scrapes a massive webpage, *do not feed it raw to the LLM*. Use CCR: cache the raw text on disk, feed the LLM a highly compressed "skimmed" summary, and give the LLM a tool to retrieve specific chunks.
- **PX (Pixel Reads):** For huge codebases, rendering text to PNGs and using a vision model (Fable-class) can cut read tokens by 85%.

## 5. Information Retrieval & Scraping
- **Local + Web (Dual Retrieval):** Jarvis must concurrently search its local `pgvector` knowledge base *and* the web for every query.
- **API-Free Web Search:** Commercial APIs (Tavily, Firecrawl) violate the 100% local rule. Jarvis should use a local SearXNG Docker container (or DuckDuckGo) combined with a local implementation of `Jina Reader` to extract markdown from HTML dynamically.
- **Quality Gates:** Implement hard-coded C++ validation for URL accessibility and citation formatting before allowing the LLM to process the data.

---

## 6. Needs Before Solutions
Before writing the final C++ code for Jarvis's research module, we must establish the following prerequisites:

### Need 1: A Reliable Local OpenAI Endpoint
**Solution Required:** We need `llama.cpp`'s `llama-server` running as a background service on the host machine. Jarvis's C++ code must be able to ping `http://localhost:8080/v1/chat/completions`. We cannot build the agent logic until the engine is reliably serving requests.

### Need 2: Deterministic Structured Outputs
**Solution Required:** Local models hallucinate formats. We need a pre-written JSON GBNF grammar file (`schema.gbnf`) that strictly defines the Action/Thought/Search schema. Jarvis must pass this grammar file to `llama-server` on every API call.

### Need 3: Local "Reader" Implementation
**Solution Required:** We cannot rely on `jina.ai` or `tavily`. We need a robust C++ (or local Python microservice) HTML-to-Markdown scraper that can bypass basic anti-bot protections, extract the `<body>`, and strip out `<nav>`/`<footer>` tags. 

### Need 4: State Checkpointing System
**Solution Required:** A local SQLite or JSON-based file system in Jarvis that records every step of the research loop. If the machine reboots, Jarvis must be able to read `session_123.json` and resume exactly where it left off.

### Need 5: Model Context Protocol (MCP) Router
**Solution Required:** To allow Jarvis to search local files, GitHub repos, and databases in the same way it searches the web, we need an MCP client implementation in C++ that translates the LLM's tool calls into MCP JSON-RPC requests.
