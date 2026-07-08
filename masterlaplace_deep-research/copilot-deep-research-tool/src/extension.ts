import * as vscode from "vscode";
import { promptAndStoreGeminiApiKey } from "./secrets";
import { DeepResearchTaskManager } from "./taskManager";
import { DeepResearchTool } from "./researchTool";
import { listOllamaModels } from "./backends/localLLMClient";

const CONFIGURE_API_KEY_COMMAND = "deepResearch.configureGeminiApiKey";
const CONFIGURE_LOCAL_LLM_COMMAND = "deepResearch.configureLocalLLM";
const SHOW_JOBS_COMMAND = "deepResearch.showJobs";
const CANCEL_JOB_COMMAND = "deepResearch.cancelJob";
const RERUN_JOB_COMMAND = "deepResearch.rerunJob";
const TOOL_NAME = "run_deep_research";

export function activate(context: vscode.ExtensionContext): void {
  const manager = new DeepResearchTaskManager(context);

  const configureCommand = vscode.commands.registerCommand(
    CONFIGURE_API_KEY_COMMAND,
    async () => {
      const stored = await promptAndStoreGeminiApiKey(context);
      if (stored) {
        void vscode.window.showInformationMessage(
          "Gemini API key stored securely for deep research fallback."
        );
      }
    }
  );

  const configureLocalLLMCommand = vscode.commands.registerCommand(
    CONFIGURE_LOCAL_LLM_COMMAND,
    async () => {
      const cfg = vscode.workspace.getConfiguration("deepResearch");
      const endpoint = cfg.get<string>("localLlmEndpoint", "http://localhost:11434");

      const models = await listOllamaModels(endpoint);
      if (models.length === 0) {
        const action = await vscode.window.showErrorMessage(
          "Ollama is not running or has no models installed. Start Ollama and pull a model first.",
          "Open Terminal"
        );
        if (action === "Open Terminal") {
          const terminal = vscode.window.createTerminal("Ollama Setup");
          terminal.show();
          terminal.sendText("ollama pull mistral:7b");
        }
        return;
      }

      const items = models.map((m) => ({
        label: m.name,
        description: m.sizeGb,
        detail: "Larger models produce better summaries but are slower",
      }));

      const picked = await vscode.window.showQuickPick(items, {
        title: "Select Ollama model for Deep Research synthesis",
        placeHolder: "Choose a local LLM model...",
        matchOnDescription: true,
      });

      if (!picked) return;

      await cfg.update("localLlmModel", picked.label, vscode.ConfigurationTarget.Global);
      void vscode.window.showInformationMessage(
        `Deep Research will now use "${picked.label}" for local synthesis.`
      );
    }
  );

  const toolDisposable = vscode.lm.registerTool(
    TOOL_NAME,
    new DeepResearchTool(manager)
  );

  const showJobsCommand = vscode.commands.registerCommand(
    SHOW_JOBS_COMMAND,
    async () => {
      const jobs = manager.listJobs();
      if (jobs.length === 0) {
        void vscode.window.showInformationMessage("No deep research jobs found.");
        return;
      }

      const pick = await vscode.window.showQuickPick(
        jobs.map((job) => ({
          label: `${job.id} [${job.status}]`,
          description: job.input.query,
          detail: job.reportPath ?? job.error ?? "No report yet",
          jobId: job.id,
        })),
        {
          title: "Deep Research Jobs",
          placeHolder: "Select a job to inspect",
        }
      );

      if (!pick) {
        return;
      }

      const selected = manager.getJob(pick.jobId);
      if (!selected) {
        return;
      }

      if (selected.reportPath) {
        await vscode.window.showTextDocument(vscode.Uri.file(selected.reportPath), {
          preview: false,
        });
      } else {
        void vscode.window.showInformationMessage(
          `Job ${selected.id} is ${selected.status}.`
        );
      }
    }
  );

  const cancelJobCommand = vscode.commands.registerCommand(
    CANCEL_JOB_COMMAND,
    async () => {
      const cancellable = manager
        .listJobs()
        .filter((job) => job.status === "queued" || job.status === "running");

      if (cancellable.length === 0) {
        void vscode.window.showInformationMessage(
          "No running or queued deep research jobs to cancel."
        );
        return;
      }

      const pick = await vscode.window.showQuickPick(
        cancellable.map((job) => ({
          label: `${job.id} [${job.status}]`,
          description: job.input.query,
          jobId: job.id,
        })),
        {
          title: "Cancel Deep Research Job",
          placeHolder: "Select a job",
        }
      );

      if (!pick) {
        return;
      }

      const cancelled = manager.cancelJob(pick.jobId);
      if (cancelled) {
        void vscode.window.showInformationMessage(`Job ${pick.jobId} cancelled.`);
      }
    }
  );

  const rerunJobCommand = vscode.commands.registerCommand(
    RERUN_JOB_COMMAND,
    async () => {
      const jobs = manager.listJobs();
      if (jobs.length === 0) {
        void vscode.window.showInformationMessage("No deep research jobs to re-run.");
        return;
      }

      const pick = await vscode.window.showQuickPick(
        jobs.map((job) => ({
          label: `${job.id} [${job.status}]`,
          description: job.input.query,
          detail: job.input.scope ?? "No scope",
          jobId: job.id,
        })),
        {
          title: "Re-run Deep Research Job",
          placeHolder: "Select a previous job to re-run",
        }
      );

      if (!pick) {
        return;
      }

      const rerun = await manager.rerunJob(pick.jobId);
      if (!rerun) {
        return;
      }

      void vscode.window.showInformationMessage(
        `Deep research re-run started as ${rerun.id}.`
      );
    }
  );

  context.subscriptions.push(
    configureCommand,
    configureLocalLLMCommand,
    showJobsCommand,
    cancelJobCommand,
    rerunJobCommand,
    toolDisposable
  );
}

export function deactivate(): void {
  // No-op.
}
