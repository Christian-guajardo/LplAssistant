import * as vscode from "vscode";
import * as path from "path";
import { getDeepResearchConfig } from "./config";
import { runGeminiDeepResearch } from "./backends/geminiClient";
import { runLocalDeepResearch } from "./backends/localBackend";
import { getGeminiApiKey } from "./secrets";
import { DeepResearchInput, ResearchJob, ResearchResult } from "./types";

const JOB_STATE_KEY = "deepResearch.jobs";

function buildJobId(): string {
  return `dr_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function buildReportMarkdown(input: DeepResearchInput, result: ResearchResult): string {
  const findings = result.findings.map((line) => `- ${line}`).join("\n");
  const sources = result.sources
    .map((source) => {
      const meta: string[] = [source.provider];
      if (source.category) {
        meta.push(source.category);
      }
      if (source.score !== undefined) {
        meta.push(`score=${source.score}`);
      }
      if (source.publishedAt) {
        meta.push(`published=${source.publishedAt}`);
      }

      return `- [${source.title}](${source.url}) (${meta.join(", ")})\n  - ${source.snippet}`;
    })
    .join("\n");
  const diagnostics = (result.providerDiagnostics ?? [])
    .map(
      (d) =>
        `- ${d.provider}: ${d.status}${
          d.count !== undefined ? ` (${d.count})` : ""
        }${d.detail ? ` - ${d.detail}` : ""}`
    )
    .join("\n");
  const limitations = (result.limitations ?? [])
    .map((line) => `- ${line}`)
    .join("\n");

  return [
    "# Deep Research Report",
    "",
    `- Query: ${input.query}`,
    `- Scope: ${input.scope ?? "none"}`,
    `- Backend: ${result.backend}`,
    `- Generated: ${new Date().toISOString()}`,
    "",
    "## Summary",
    "",
    result.summary,
    "",
    "## Findings",
    "",
    findings.length > 0 ? findings : "- No findings.",
    "",
    "## Sources",
    "",
    sources.length > 0 ? sources : "- No explicit sources captured.",
    "",
    "## Provider Diagnostics",
    "",
    diagnostics.length > 0 ? diagnostics : "- No provider diagnostics available.",
    "",
    "## Limitations",
    "",
    limitations.length > 0 ? limitations : "- No major limitations reported.",
    "",
  ].join("\n");
}

async function writeReport(
  input: DeepResearchInput,
  result: ResearchResult,
  outputFolder: string
): Promise<vscode.Uri> {
  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  if (!workspaceFolder) {
    throw new Error("No workspace folder is open.");
  }

  const outputDir = vscode.Uri.file(
    path.join(workspaceFolder.uri.fsPath, outputFolder)
  );
  await vscode.workspace.fs.createDirectory(outputDir);

  const fileName = `deep-research-${Date.now()}.md`;
  const reportUri = vscode.Uri.file(path.join(outputDir.fsPath, fileName));
  const markdown = buildReportMarkdown(input, result);

  await vscode.workspace.fs.writeFile(reportUri, Buffer.from(markdown, "utf8"));
  return reportUri;
}

export class DeepResearchTaskManager {
  private jobs = new Map<string, ResearchJob>();
  private runningJobTokens = new Map<string, vscode.CancellationTokenSource>();

  constructor(private readonly context: vscode.ExtensionContext) {
    this.restoreJobs();
  }

  public getJob(jobId: string): ResearchJob | undefined {
    return this.jobs.get(jobId);
  }

  public listJobs(): ResearchJob[] {
    return [...this.jobs.values()].sort((a, b) => b.createdAt - a.createdAt);
  }

  public async startJob(input: DeepResearchInput): Promise<ResearchJob> {
    const now = Date.now();
    const job: ResearchJob = {
      id: buildJobId(),
      input,
      createdAt: now,
      updatedAt: now,
      status: "queued",
    };

    this.jobs.set(job.id, job);
    this.persistJobs();

    void this.runJob(job.id);
    return job;
  }

  public async rerunJob(jobId: string): Promise<ResearchJob | undefined> {
    const existing = this.jobs.get(jobId);
    if (!existing) {
      return undefined;
    }

    return this.startJob({
      query: existing.input.query,
      scope: existing.input.scope,
      preferLocal: existing.input.preferLocal,
    });
  }

  public cancelJob(jobId: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job) {
      return false;
    }

    const tokenSource = this.runningJobTokens.get(jobId);
    if (tokenSource) {
      tokenSource.cancel();
      tokenSource.dispose();
      this.runningJobTokens.delete(jobId);
    }

    job.status = "cancelled";
    job.updatedAt = Date.now();
    job.error = "Cancelled by user.";
    this.persistJobs();
    return true;
  }

  private async runJob(jobId: string): Promise<void> {
    const job = this.jobs.get(jobId);
    if (!job) {
      return;
    }

    if (job.status === "cancelled") {
      return;
    }

    const cfg = getDeepResearchConfig();
    job.status = "running";
    job.updatedAt = Date.now();
    this.persistJobs();

    const tokenSource = new vscode.CancellationTokenSource();
    this.runningJobTokens.set(jobId, tokenSource);

    try {
      const result = await this.executeResearch(
        job.input,
        cfg,
        this.context,
        tokenSource.token
      );

      if (tokenSource.token.isCancellationRequested) {
        job.status = "cancelled";
        job.updatedAt = Date.now();
        job.error = "Cancelled by user.";
        this.persistJobs();
        return;
      }

      const reportUri = await writeReport(job.input, result, cfg.outputFolder);
      job.reportPath = reportUri.fsPath;
      job.status = "completed";
      job.updatedAt = Date.now();
      this.persistJobs();

      await vscode.window.showTextDocument(reportUri, {
        preview: false,
      });

      void vscode.window.showInformationMessage(
        `Deep Research finished (${job.id}). Report generated.`
      );
    } catch (error) {
      if (tokenSource.token.isCancellationRequested) {
        job.status = "cancelled";
        job.updatedAt = Date.now();
        job.error = "Cancelled by user.";
        this.persistJobs();
        return;
      }

      job.status = "failed";
      job.updatedAt = Date.now();
      job.error = error instanceof Error ? error.message : "Unknown error";
      this.persistJobs();

      void vscode.window.showErrorMessage(
        `Deep Research failed (${job.id}): ${job.error}`
      );
    } finally {
      this.runningJobTokens.delete(jobId);
      tokenSource.dispose();
    }
  }

  private async executeResearch(
    input: DeepResearchInput,
    cfg: ReturnType<typeof getDeepResearchConfig>,
    context: vscode.ExtensionContext,
    token: vscode.CancellationToken
  ): Promise<ResearchResult> {
    const preferLocal = input.preferLocal ?? true;

    if (cfg.backend === "cloud-only" || !preferLocal) {
      return this.executeCloud(input, cfg, context);
    }

    try {
      return await runLocalDeepResearch(
        input.query,
        input.scope,
        {
          maxSources: cfg.maxSources,
          enableReaderProxy: cfg.enableReaderProxy,
          readerProxyBaseUrl: cfg.readerProxyBaseUrl,
          targetDocDomains: cfg.targetDocDomains,
          maxDocResultsPerDomain: cfg.maxDocResultsPerDomain,
          requestTimeoutSeconds: cfg.requestTimeoutSeconds,
          localLlmEnabled: cfg.localLlmEnabled,
          localLlmEndpoint: cfg.localLlmEndpoint,
          localLlmModel: cfg.localLlmModel,
          localLlmMaxSources: cfg.localLlmMaxSources,
        },
        token
      );
    } catch (localError) {
      if (!cfg.enableCloudFallback) {
        throw localError;
      }

      return this.executeCloud(input, cfg, context);
    }
  }

  private async executeCloud(
    input: DeepResearchInput,
    cfg: ReturnType<typeof getDeepResearchConfig>,
    context: vscode.ExtensionContext
  ): Promise<ResearchResult> {
    const apiKey = await getGeminiApiKey(context);
    if (!apiKey) {
      throw new Error(
        "Gemini API key is not configured. Run 'Deep Research: Configure Gemini API Key'."
      );
    }

    const pollIntervalMs = Math.max(5, cfg.pollIntervalSeconds) * 1000;
    return runGeminiDeepResearch(input.query, apiKey, 240, pollIntervalMs);
  }

  private persistJobs(): void {
    const payload = JSON.stringify([...this.jobs.values()]);
    void this.context.workspaceState.update(JOB_STATE_KEY, payload);
  }

  private restoreJobs(): void {
    const payload = this.context.workspaceState.get<string>(JOB_STATE_KEY);
    if (!payload) {
      return;
    }

    try {
      const list = JSON.parse(payload) as ResearchJob[];
      for (const job of list) {
        if (job.status === "running" || job.status === "queued") {
          // Running tasks cannot be resumed deterministically after reload.
          job.status = "failed";
          job.error = "VS Code restarted while the task was running.";
          job.updatedAt = Date.now();
        }

        this.jobs.set(job.id, job);
      }
    } catch {
      // Ignore invalid persisted state.
    }
  }
}
