import { useState } from "react";
import { X, Zap } from "lucide-react";
import { saveSetup } from "../api";

interface SetupWizardProps {
  onComplete: () => void;
  onClose: () => void;
}

const PRESETS: Record<string, { base: string; model: string }> = {
  deepseek: { base: "https://api.deepseek.com/", model: "deepseek-chat" },
  dashscope: {
    base: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: "qwen-plus",
  },
  openai: { base: "https://api.openai.com/v1", model: "gpt-4o" },
  ollama: { base: "http://localhost:11434/v1", model: "llama3" },
};

export default function SetupWizard({ onComplete, onClose }: SetupWizardProps) {
  const [provider, setProvider] = useState("deepseek");
  const [apiKey, setApiKey] = useState("");
  const [apiBase, setApiBase] = useState(PRESETS.deepseek.base);
  const [model, setModel] = useState(PRESETS.deepseek.model);
  const [ncbiEmail, setNcbiEmail] = useState("");
  const [ncbiKey, setNcbiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const selectPreset = (key: string) => {
    setProvider(key);
    if (key in PRESETS) {
      setApiBase(PRESETS[key].base);
      setModel(PRESETS[key].model);
    } else {
      setApiBase("");
      setModel("");
    }
  };

  const handleSave = async () => {
    if (!apiKey || !apiBase || !model) {
      setError("API key, base URL, and model are required.");
      return;
    }
    setSaving(true);
    setError("");
    try {
      await saveSetup({
        api_key: apiKey,
        api_base: apiBase,
        model,
        ncbi_email: ncbiEmail,
        ncbi_api_key: ncbiKey,
      });
      onComplete();
    } catch (e: any) {
      setError(e.message || "Failed to save configuration.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="setup-overlay" onClick={onClose}>
      <div className="setup-modal" onClick={(e) => e.stopPropagation()}>
        <button className="setup-close" onClick={onClose}>
          <X size={16} />
        </button>

        <div className="setup-header">
          <Zap size={20} className="setup-icon" />
          <h2>Configure LitScribe</h2>
          <p>Connect an OpenAI-compatible LLM to get started.</p>
        </div>

        <div className="setup-providers">
          {Object.entries({
            deepseek: "DeepSeek",
            dashscope: "DashScope",
            openai: "OpenAI",
            ollama: "Ollama",
            custom: "Custom",
          }).map(([key, label]) => (
            <button
              key={key}
              className={`setup-provider ${provider === key ? "active" : ""}`}
              onClick={() => selectPreset(key)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="setup-fields">
          <label>
            <span>API Key</span>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
            />
          </label>

          <label>
            <span>Base URL</span>
            <input
              type="text"
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
              placeholder="https://api.example.com/v1"
            />
          </label>

          <label>
            <span>Model</span>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="model-name"
            />
          </label>

          <div className="setup-optional">
            <span className="setup-optional-label">Optional: Academic APIs</span>
            <div className="setup-row">
              <label>
                <span>NCBI Email</span>
                <input
                  type="email"
                  value={ncbiEmail}
                  onChange={(e) => setNcbiEmail(e.target.value)}
                  placeholder="For PubMed access"
                />
              </label>
              <label>
                <span>NCBI API Key</span>
                <input
                  type="text"
                  value={ncbiKey}
                  onChange={(e) => setNcbiKey(e.target.value)}
                  placeholder="Optional"
                />
              </label>
            </div>
          </div>
        </div>

        {error && <div className="setup-error">{error}</div>}

        <button
          className="setup-save"
          onClick={handleSave}
          disabled={saving || !apiKey || !apiBase || !model}
        >
          {saving ? "Saving..." : "Save & Start"}
        </button>
      </div>
    </div>
  );
}
