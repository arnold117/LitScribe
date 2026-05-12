import { useState, useRef, useEffect } from "react";
import type { ChatMessage, PlanSection } from "../types";

interface ChatProps {
  messages: ChatMessage[];
  onSend: (message: string, attachment?: File) => void;
  onPlanUpdate: (sections: PlanSection[], constraints: string) => void;
  onPlanExecute: () => void;
  selectionContext: string;
  onClearSelection: () => void;
  loading: boolean;
}

export default function Chat({
  messages,
  onSend,
  onPlanUpdate,
  onPlanExecute,
  selectionContext,
  onClearSelection,
  loading,
}: ChatProps) {
  const [input, setInput] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (selectionContext) inputRef.current?.focus();
  }, [selectionContext]);

  const handleSubmit = () => {
    const text = input.trim();
    if (!text && !selectionContext) return;

    let fullMessage = "";
    if (selectionContext) {
      const preview = selectionContext.slice(0, 200) + (selectionContext.length > 200 ? "..." : "");
      fullMessage = `[Selected: "${preview}"]\n\n${text}`;
      onClearSelection();
    } else {
      fullMessage = text;
    }
    onSend(fullMessage);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFile = (file: File) => {
    if (!file) return;
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (ext === "md" || ext === "txt") {
      const reader = new FileReader();
      reader.onload = (ev) => {
        const text = ev.target?.result as string;
        onSend(`[Uploaded: ${file.name}]\n\n${input.trim() || "Parse this outline and show me a plan."}`, file);
        setInput("");
      };
      reader.readAsText(file);
    } else if (ext === "docx") {
      onSend(`[Uploaded: ${file.name}]\n\n${input.trim() || "Parse this outline and show me a plan."}`, file);
      setInput("");
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div
      className={`chat-container ${dragOver ? "chat-dragover" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <div className="chat-header">
        <span>Chat</span>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">&#128172;</div>
            <div>Drop a file or describe what you want</div>
            <div className="chat-empty-hint">
              "Upload an outline and generate a review focusing on Chinese Ferula species"
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg chat-msg-${msg.role}`}>
            {msg.role !== "system" && (
              <div className="chat-msg-role">
                {msg.role === "user" ? "You" : "LitScribe"}
              </div>
            )}

            {msg.type === "plan" ? (
              <PlanCard
                data={msg.data}
                onUpdate={onPlanUpdate}
                onExecute={onPlanExecute}
              />
            ) : msg.type === "progress" ? (
              <ProgressCard data={msg.data} />
            ) : msg.type === "coverage" ? (
              <CoverageCard data={msg.data} />
            ) : msg.type === "actions" ? (
              <ActionsCard data={msg.data} onAction={(a) => onSend(a)} />
            ) : (
              <div className="chat-msg-content">
                {msg.attachment && (
                  <div className="chat-attachment">
                    &#128206; {msg.attachment.name}
                  </div>
                )}
                {msg.content}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-msg chat-msg-assistant">
            <div className="chat-msg-role">LitScribe</div>
            <div className="chat-msg-content chat-loading">
              <span className="dot-pulse" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        {selectionContext && (
          <div className="chat-selection-badge">
            <span>
              "{selectionContext.slice(0, 80)}
              {selectionContext.length > 80 ? "..." : ""}"
            </span>
            <button onClick={onClearSelection}>&times;</button>
          </div>
        )}
        <div className="chat-input-row">
          <button
            className="chat-attach-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Upload file"
          >
            &#128206;
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".docx,.md,.txt"
            style={{ display: "none" }}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              selectionContext
                ? "What to do with this selection?"
                : "Describe what you want, or drop a file..."
            }
            rows={2}
            disabled={loading}
          />
          <button
            className="chat-send-btn"
            onClick={handleSubmit}
            disabled={loading || (!input.trim() && !selectionContext)}
          >
            &#10148;
          </button>
        </div>
      </div>
    </div>
  );
}


function PlanCard({
  data,
  onUpdate,
  onExecute,
}: {
  data: any;
  onUpdate: (sections: PlanSection[], constraints: string) => void;
  onExecute: () => void;
}) {
  const [sections, setSections] = useState<PlanSection[]>(data.sections || []);
  const [constraints, setConstraints] = useState(data.constraints || "");
  const [expanded, setExpanded] = useState(true);

  const toggleSection = (i: number) => {
    const updated = [...sections];
    updated[i] = { ...updated[i], enabled: !updated[i].enabled };
    setSections(updated);
    onUpdate(updated, constraints);
  };

  const enabledCount = sections.filter((s) => s.enabled).length;

  return (
    <div className="chat-plan-card">
      <div className="plan-header" onClick={() => setExpanded(!expanded)}>
        <span className="plan-title">
          Outline Plan ({enabledCount}/{sections.length} sections)
        </span>
        <span className="plan-toggle">{expanded ? "▾" : "▸"}</span>
      </div>

      {expanded && (
        <>
          <div className="plan-sections">
            {sections.map((s, i) => (
              <label key={i} className={`plan-section ${s.enabled ? "" : "plan-section-disabled"}`}>
                <input
                  type="checkbox"
                  checked={s.enabled}
                  onChange={() => toggleSection(i)}
                />
                <span className="plan-section-num">{s.number}</span>
                <span>{s.title}</span>
              </label>
            ))}
          </div>

          <div className="plan-constraints">
            <label>Constraints (species, scope, etc.)</label>
            <textarea
              value={constraints}
              onChange={(e) => {
                setConstraints(e.target.value);
                onUpdate(sections, e.target.value);
              }}
              placeholder="e.g., Focus on F. sinkiangensis, F. bungeana, F. songarica..."
              rows={2}
            />
          </div>

          <div className="plan-actions">
            <button className="plan-run-btn" onClick={onExecute}>
              Generate {enabledCount} sections
            </button>
          </div>
        </>
      )}
    </div>
  );
}


function ProgressCard({ data }: { data: any }) {
  const { current, total, title, papers, words, done } = data;
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;

  return (
    <div className="chat-progress-card">
      <div className="progress-bar-outer">
        <div className="progress-bar-inner" style={{ width: `${pct}%` }} />
      </div>
      <div className="progress-info">
        {done ? (
          <span>All {total} sections complete</span>
        ) : (
          <span>
            [{current}/{total}] {title}
            {papers !== undefined && <span className="progress-detail"> — {papers} papers, {words} words</span>}
          </span>
        )}
      </div>
    </div>
  );
}


function CoverageCard({ data }: { data: any }) {
  return (
    <div className="chat-coverage-card">
      <div className="coverage-header">
        Coverage: {data.covered}/{data.total} ({data.coverage_pct}%)
      </div>
      {data.missing_entities?.length > 0 && (
        <div className="coverage-missing">
          <span className="coverage-label">Missing:</span>
          {data.missing_entities.map((e: string, i: number) => (
            <span key={i} className="coverage-tag coverage-tag-missing">{e}</span>
          ))}
        </div>
      )}
      {data.covered_entities?.length > 0 && (
        <div className="coverage-found">
          <span className="coverage-label">Covered:</span>
          {data.covered_entities.slice(0, 10).map((e: string, i: number) => (
            <span key={i} className="coverage-tag coverage-tag-found">{e}</span>
          ))}
          {data.covered_entities.length > 10 && (
            <span className="coverage-tag">+{data.covered_entities.length - 10}</span>
          )}
        </div>
      )}
    </div>
  );
}


function ActionsCard({ data, onAction }: { data: any; onAction: (action: string) => void }) {
  return (
    <div className="chat-actions-card">
      {data.text && <div className="actions-text">{data.text}</div>}
      <div className="actions-btns">
        {data.actions?.map((a: { label: string; value: string }, i: number) => (
          <button key={i} className="action-btn" onClick={() => onAction(a.value)}>
            {a.label}
          </button>
        ))}
      </div>
    </div>
  );
}
