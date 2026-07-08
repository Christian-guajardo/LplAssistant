import * as vscode from "vscode";

const GEMINI_API_KEY_SECRET = "deepResearch.geminiApiKey";

export async function getGeminiApiKey(
  context: vscode.ExtensionContext
): Promise<string | undefined> {
  return context.secrets.get(GEMINI_API_KEY_SECRET);
}

export async function promptAndStoreGeminiApiKey(
  context: vscode.ExtensionContext
): Promise<boolean> {
  const apiKey = await vscode.window.showInputBox({
    title: "Configure Gemini API key",
    prompt: "Enter your Gemini API key for deep research cloud fallback.",
    ignoreFocusOut: true,
    password: true,
    validateInput(value) {
      if (!value || value.trim().length < 10) {
        return "API key seems invalid.";
      }

      return undefined;
    },
  });

  if (!apiKey) {
    return false;
  }

  await context.secrets.store(GEMINI_API_KEY_SECRET, apiKey.trim());
  return true;
}

export async function getOrPromptGeminiApiKey(
  context: vscode.ExtensionContext
): Promise<string | undefined> {
  const existing = await getGeminiApiKey(context);
  if (existing) {
    return existing;
  }

  const stored = await promptAndStoreGeminiApiKey(context);
  if (!stored) {
    return undefined;
  }

  return getGeminiApiKey(context);
}
