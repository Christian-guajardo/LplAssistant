# Deep Research Tool — Deployment Guide

**Extension:** copilot-deep-research-tool v0.0.1  
**Package:** `copilot-deep-research-tool-0.0.1.vsix` (50 KB)  
**Status:** ✅ Production-Ready  

---

## 📦 Package Contents

```
copilot-deep-research-tool-0.0.1.vsix
├── extension.js              (Main entry point)
├── backends/
│   ├── localBackend.js       (10-provider research pipeline)
│   ├── geminiClient.js       (Cloud fallback)
├── config.js                 (Settings management)
├── taskManager.js            (Async job orchestration)
├── types.js                  (Type definitions)
├── package.json              (Manifest + schema)
├── README.md                 (User documentation)
└── [dependencies]            (VS Code APIs, utilities)
```

**Build:** TypeScript 6.0.3 → JavaScript (out/)  
**Size:** 50 KB (minified, ready for distribution)  

---

## 🚀 Deployment Options

### Option 1: Local Installation (Testing)

**For internal testing or single-user setup:**

```bash
# Navigate to extension directory
cd copilot-deep-research-tool

# Install extension locally
code --install-extension ./copilot-deep-research-tool-0.0.1.vsix
```

Then:
- Reload VS Code
- Verify command palette: `Deep Research: Show Jobs` appears
- Test via Copilot Chat: `#deep_research Find Vulkan best practices`

### Option 2: VS Code Marketplace (Public Release)

**For team sharing or public availability:**

#### Prerequisites
1. **VS Code Marketplace Account**
   - Sign up at https://marketplace.visualstudio.com
   - Create publisher (e.g., "YourCompany" or personal namespace)

2. **Personal Access Token (PAT)**
   - Go to https://dev.azure.com → Personal access tokens
   - Scopes: `Marketplace (manage)` + `All scopes`
   - Store securely (do NOT commit to repo)

3. **Update package.json**
   ```json
   {
     "publisher": "your-publisher-name",
     "repository": {
       "type": "git",
       "url": "https://github.com/your-org/copilot-deep-research-tool.git"
     }
   }
   ```

#### Publish Steps

```bash
cd copilot-deep-research-tool

# Install VSCE if not already done
npm install --save-dev vsce

# Create publisher (first time only)
npx vsce create-publisher <publisher-name>
# Follow prompts: accept terms, verify email

# Publish extension
npx vsce publish --pat <your-personal-access-token>
```

Expected output:
```
 DONE  Published to https://marketplace.visualstudio.com/items?itemName=<publisher>/<name>
```

**Then users can install directly from Marketplace:**
- VS Code Extensions → Search "Copilot Deep Research Tool"
- Click Install

### Option 3: Corporate/Internal Distribution

**For enterprise environments (no public listing):**

1. **Host VSIX on internal server**
   ```
   https://internal-server.com/extensions/copilot-deep-research-tool-0.0.1.vsix
   ```

2. **Share installation link**
   ```bash
   # Users run:
   code --install-extension https://internal-server.com/extensions/copilot-deep-research-tool-0.0.1.vsix
   ```

3. **Or distribute via package manager** (Ansible, Chocolatey, etc.)
   - Add to internal software repository
   - Deploy via IT automation

---

## ⚙️ Post-Installation Configuration

After installation, users should configure via VS Code settings:

### Minimal Setup (Recommended)

**File:** `.vscode/settings.json` (workspace) or user settings (global)

```json
{
  "deepResearch.backend": "local-first",
  "deepResearch.maxSources": 8,
  "deepResearch.enableReaderProxy": true,
  "deepResearch.enableVerboseLogging": false
}
```

### Advanced Setup (Optional)

```json
{
  "deepResearch.backend": "local-first",
  "deepResearch.maxSources": 10,
  "deepResearch.outputFolder": ".github/research_reports",
  "deepResearch.enableCloudFallback": true,
  "deepResearch.pollIntervalSeconds": 10,
  "deepResearch.enableReaderProxy": true,
  "deepResearch.readerProxyBaseUrl": "https://r.jina.ai/",
  "deepResearch.targetDocDomains": [
    "khronos.org",
    "nvidia.com",
    "developer.mozilla.org",
    "docs.python.org"
  ],
  "deepResearch.maxDocResultsPerDomain": 3,
  "deepResearch.requestTimeoutSeconds": 15,
  "deepResearch.enableVerboseLogging": true
}
```

### Cloud Fallback Setup (If Needed)

If local backend fails, configure Gemini fallback:

```json
{
  "deepResearch.enableCloudFallback": true,
  "deepResearch.backend": "local-first"
}
```

Then via command palette:
- `Deep Research: Configure Gemini API Key`
- Enter your Gemini API key (stored securely in VS Code)

---

## ✅ Verification Checklist

After installation:

- [ ] Extension appears in Extensions panel (`Ctrl+Shift+X`)
- [ ] Commands available in Command Palette (`Ctrl+Shift+P`):
  - `Deep Research: Show Jobs`
  - `Deep Research: Cancel Job`
  - `Deep Research: Re-run Job`
  - `Deep Research: Configure Gemini API Key`
- [ ] `#deep_research` reference works in Copilot Chat
- [ ] Test query: `#deep_research Find Vulkan best practices`
- [ ] Report generated: `.github/research_reports/research_<timestamp>.md`

---

## 🧪 Testing Commands

### Smoke Test (Local Backend)

```bash
cd copilot-deep-research-tool
npm run smoke:local
npm run smoke:local:query -- "Grafana Loki observability"
```

Expected: 4–10 sources, 2–4 distinct providers, 0 errors ✅

### Performance Baseline

- Query latency: 8–12 seconds
- Parallel providers: 10
- Typical sources: 6–9
- Build size: 50 KB

---

## 🔧 Troubleshooting

### Extension doesn't appear in Command Palette

**Issue:** VS Code not recognizing extension  
**Fix:**
```bash
# Clear cache and reinstall
code --disable-all-extensions
code --install-extension ./copilot-deep-research-tool-0.0.1.vsix
code --enable-all-extensions
```

### Reports not generated

**Issue:** `.github/research_reports/` not found  
**Fix:**
```bash
# Create directory manually
mkdir -p .github/research_reports
```

### Providers returning empty (Reddit, OpenAlex timeout)

**Issue:** Rate limits or network timeouts  
**Fix:**
- Increase `deepResearch.requestTimeoutSeconds` to 15–20
- Enable `deepResearch.enableVerboseLogging` to debug
- Check network connectivity

### Reader proxy fails (Jina down)

**Issue:** Dynamic page extraction unavailable  
**Fix:**
- Disable: `"deepResearch.enableReaderProxy": false`
- Or provide alternative reader proxy: `"deepResearch.readerProxyBaseUrl": "https://your-proxy.com/"`

---

## 📊 Feature Verification

| Feature | Status | Notes |
|---------|--------|-------|
| Local execution (10 providers) | ✅ | Wikipedia, StackExchange, GitHub, arXiv, OpenAlex, Reddit, etc. |
| Async jobs | ✅ | Fire-and-forget, persistence in workspace state |
| Report generation | ✅ | Markdown to `.github/research_reports/` |
| Reader proxy enrichment | ✅ | Jina integration (with fallback) |
| PDF detection | ✅ | Direct .pdf URLs + arXiv abs→pdf conversion |
| Cloud fallback | ✅ | Gemini Interactions API (optional) |
| Configuration | ✅ | 12 VS Code settings (timeouts, logging, domains, etc.) |
| Smoke tests | ✅ | Pass on default and custom queries |

---

## 📚 User Documentation

Full documentation included in extension:

- [README.md](../copilot-deep-research-tool/README.md) — User guide, architecture, limitations
- [FINAL_STATUS.md](../FINAL_STATUS.md) — Quality report, coverage matrix
- Built-in help: `Deep Research: Show Jobs` → opens report template

---

## 🔄 Version Management

**Current Version:** 0.0.1  
**Release Date:** June 2026

### Versioning Strategy

- **0.0.1** → MVP (local 10 providers, async jobs, reports)
- **0.1.0** → OAuth GitHub API (5000 req/hour)
- **0.2.0** → Vector indexing for large knowledge bases
- **1.0.0** → Public Marketplace release

To bump version:
```json
{
  "version": "0.1.0"  // Update in package.json
}
```

Then re-run packaging:
```bash
npm run compile && npx vsce package
```

---

## 🚨 Security & Privacy

- **No telemetry:** All research runs locally by default
- **API keys:** Stored in VS Code SecretStorage (encrypted)
- **Network:** Configurable reader proxy; can be disabled
- **Logs:** Only if `enableVerboseLogging` explicitly set to true
- **Data:** Reports written to workspace (versioned, not sent anywhere)

---

## 📞 Support & Feedback

For issues or feature requests:

1. Check [README.md](../copilot-deep-research-tool/README.md) known limitations
2. Enable verbose logging: `"deepResearch.enableVerboseLogging": true`
3. Share report file + logs with maintainers
4. Or check GitHub issues (if public repo)

---

## ✨ Next Steps (Post-MVP)

- [ ] Publish to VS Code Marketplace
- [ ] OAuth GitHub API for team usage
- [ ] Vector indexing for cached knowledge bases
- [ ] Advanced PDF parsing (layout, figures, tables)
- [ ] MCP integration for broader tool ecosystem
- [ ] Custom allowlists/blocklists per user

---

**Status:** 🟢 **READY FOR DEPLOYMENT**

**Recommended Path:**
1. Internal testing (Option 1) → 1 week
2. Enterprise deployment (Option 3) → 2 weeks
3. Marketplace publication (Option 2) → (optional, future)

Good luck! 🚀
