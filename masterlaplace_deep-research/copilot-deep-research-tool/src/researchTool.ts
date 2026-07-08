import * as vscode from "vscode";
import { DeepResearchTaskManager } from "./taskManager";
import { DeepResearchInput } from "./types";

export class DeepResearchTool
  implements vscode.LanguageModelTool<DeepResearchInput>
{
  constructor(private readonly manager: DeepResearchTaskManager) {}

  async prepareInvocation(
    options: vscode.LanguageModelToolInvocationPrepareOptions<DeepResearchInput>
  ): Promise<vscode.PreparedToolInvocation> {
    const prompt = new vscode.MarkdownString(
      [
        "Start deep research job?",
        "",
        `Query: ${options.input.query}`,
        options.input.scope ? `Scope: ${options.input.scope}` : "",
        "",
        "This may collect data from public web sources and write a report in your workspace.",
      ]
        .filter((line) => line.length > 0)
        .join("\n")
    );

    return {
      invocationMessage: "Launching deep research in background",
      confirmationMessages: {
        title: "Confirm deep research run",
        message: prompt,
      },
    };
  }

  async invoke(
    options: vscode.LanguageModelToolInvocationOptions<DeepResearchInput>
  ): Promise<vscode.LanguageModelToolResult> {
    if (!options.input.query || options.input.query.trim().length === 0) {
      throw new Error("query is required and must not be empty.");
    }

    const job = await this.manager.startJob({
      query: options.input.query.trim(),
      scope: options.input.scope?.trim(),
      preferLocal: options.input.preferLocal,
    });

    const text = [
      `Deep research started in background. Job ID: ${job.id}.`,
      "The report will be written to your workspace when completed.",
      "You can continue coding while this runs.",
    ].join(" ");

    return new vscode.LanguageModelToolResult([
      new vscode.LanguageModelTextPart(text),
    ]);
  }
}
