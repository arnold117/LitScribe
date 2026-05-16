import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Paperclip,
  ArrowUp,
  X,
  ChevronDown,
  ChevronRight,
  Loader2,
  MessageSquare,
  Upload,
  GripVertical,
} from "lucide-react";
import type { ChatMessage, PlanSection, PipelineStep, SearchPaper } from "../types";

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
      const preview =
        selectionContext.slice(0, 200) +
        (selectionContext.length > 200 ? "..." : "");
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
    if (!["md", "txt", "docx"].includes(ext || "")) return;
    const userText = input.trim();
    onSend(
      userText ? `[Uploaded: ${file.name}]\n\n${userText}` : `[Uploaded: ${file.name}]`,
      file,
    );
    setInput("");
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
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <div className="chat-header">
        <MessageSquare size={13} />
        <span>Chat</span>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <Upload size={32} strokeWidth={1.2} />
            <p>Drop an outline or ask a research question</p>
            <div className="chat-empty-examples">
              <button className="chat-example-btn" onClick={() => { setInput("Review recent advances in CRISPR for CHO cell engineering"); inputRef.current?.focus(); }}>
                CRISPR for CHO cell engineering
              </button>
              <button className="chat-example-btn" onClick={() => { setInput("综述大语言模型在医学影像诊断中的应用"); inputRef.current?.focus(); }}>
                LLM在医学影像中的应用
              </button>
              <button className="chat-example-btn" onClick={() => { setInput("Review transformer architectures for time series forecasting"); inputRef.current?.focus(); }}>
                Transformers for time series
              </button>
            </div>
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-msg chat-msg-${msg.role}`}>
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
            ) : msg.type === "steps" ? (
              <StepsCard data={msg.data} />
            ) : msg.type === "progress" ? (
              <ProgressCard data={msg.data} />
            ) : msg.type === "coverage" ? (
              <CoverageCard data={msg.data} />
            ) : msg.type === "actions" ? (
              <ActionsCard data={msg.data} onAction={(a) => onSend(a)} />
            ) : msg.type === "grounding" ? (
              <GroundingCard data={msg.data} />
            ) : (
              <div className="chat-msg-content">
                {msg.attachment && (
                  <div className="chat-attachment">
                    <Paperclip size={12} />
                    {msg.attachment.name}
                  </div>
                )}
                {msg.role === "assistant" ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content}
                  </ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-msg chat-msg-assistant">
            <div className="chat-msg-role">LitScribe</div>
            <div className="chat-msg-content chat-loading">
              <Loader2 size={14} className="spin" />
              <span>Thinking...</span>
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
            <button onClick={onClearSelection}>
              <X size={12} />
            </button>
          </div>
        )}
        <div className="chat-input-row">
          <button
            className="chat-attach-btn"
            onClick={() => fileInputRef.current?.click()}
            title="Upload .docx / .md / .txt"
          >
            <Paperclip size={15} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".docx,.md,.txt"
            style={{ display: "none" }}
            onChange={(e) =>
              e.target.files?.[0] && handleFile(e.target.files[0])
            }
          />
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              selectionContext
                ? "What to do with this selection?"
                : "Ask anything or drop a file..."
            }
            rows={2}
            disabled={loading}
          />
          <button
            className="chat-send-btn"
            onClick={handleSubmit}
            disabled={loading || (!input.trim() && !selectionContext)}
          >
            <ArrowUp size={16} />
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
  const [sections, setSections] = useState<PlanSection[]>(
    data.sections || [],
  );
  const [constraints, setConstraints] = useState(data.constraints || "");
  const [expanded, setExpanded] = useState(true);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [dropTarget, setDropTarget] = useState<number | null>(null);

  const toggleSection = (i: number) => {
    const updated = [...sections];
    updated[i] = { ...updated[i], enabled: !updated[i].enabled };
    setSections(updated);
    onUpdate(updated, constraints);
  };

  const handleDragStart = (i: number) => {
    setDragIndex(i);
  };

  const handleDragOver = (e: React.DragEvent, i: number) => {
    e.preventDefault();
    setDropTarget(i);
  };

  const handleDrop = (i: number) => {
    if (dragIndex === null || dragIndex === i) {
      setDragIndex(null);
      setDropTarget(null);
      return;
    }
    const updated = [...sections];
    const [moved] = updated.splice(dragIndex, 1);
    updated.splice(i, 0, moved);
    setSections(updated);
    onUpdate(updated, constraints);
    setDragIndex(null);
    setDropTarget(null);
  };

  const handleDragEnd = () => {
    setDragIndex(null);
    setDropTarget(null);
  };

  const enabledCount = sections.filter((s) => s.enabled).length;

  return (
    <div className="card card-plan">
      <div className="card-header" onClick={() => setExpanded(!expanded)}>
        <span>
          Outline Plan ({enabledCount}/{sections.length} sections)
        </span>
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </div>

      {expanded && (
        <>
          <div className="plan-sections">
            {sections.map((s, i) => (
              <div
                key={`${s.number}-${i}`}
                className={`plan-section ${s.enabled ? "" : "disabled"} ${dragIndex === i ? "dragging" : ""} ${dropTarget === i ? "drop-target" : ""}`}
                draggable
                onDragStart={() => handleDragStart(i)}
                onDragOver={(e) => handleDragOver(e, i)}
                onDrop={() => handleDrop(i)}
                onDragEnd={handleDragEnd}
              >
                <span className="plan-grip">
                  <GripVertical size={12} />
                </span>
                <input
                  type="checkbox"
                  checked={s.enabled}
                  onChange={() => toggleSection(i)}
                />
                <span className="plan-num">{s.number}</span>
                <span className="plan-title">{s.title}</span>
              </div>
            ))}
          </div>

          <div className="plan-constraints">
            <label>Constraints</label>
            <textarea
              value={constraints}
              onChange={(e) => {
                setConstraints(e.target.value);
                onUpdate(sections, e.target.value);
              }}
              placeholder="e.g., Focus on specific species, time range, methods..."
              rows={2}
            />
          </div>

          <div className="plan-footer">
            <button className="plan-run" onClick={onExecute}>
              Generate {enabledCount} sections
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function ProgressCard({ data }: { data: any }) {
  const { current, total, title, papers, words } = data || {};
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;

  return (
    <div className="card card-progress">
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="progress-label">
        [{current}/{total}] {title}
        {papers !== undefined && (
          <span className="progress-detail">
            {" "}
            — {papers} papers, {words} words
          </span>
        )}
      </div>
    </div>
  );
}

function CoverageCard({ data }: { data: any }) {
  return (
    <div className="card card-coverage">
      <div className="coverage-header">
        Coverage: {data.covered}/{data.total} ({data.coverage_pct}%)
      </div>
      {data.missing_entities?.length > 0 && (
        <div className="coverage-row">
          <span className="coverage-label">Missing:</span>
          <div className="coverage-tags">
            {data.missing_entities.map((e: string, i: number) => (
              <span key={i} className="tag tag-red">
                {e}
              </span>
            ))}
          </div>
        </div>
      )}
      {data.covered_entities?.length > 0 && (
        <div className="coverage-row">
          <span className="coverage-label">Covered:</span>
          <div className="coverage-tags">
            {data.covered_entities.slice(0, 10).map((e: string, i: number) => (
              <span key={i} className="tag tag-green">
                {e}
              </span>
            ))}
            {data.covered_entities.length > 10 && (
              <span className="tag">+{data.covered_entities.length - 10}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function GroundingCard({ data }: { data: any }) {
  return (
    <div className="card card-grounding">
      <div className="grounding-header">Citation Grounding</div>
      <div className="grounding-stats">
        <div className="grounding-stat">
          <span className="grounding-num">{data.verified}</span>
          <span className="grounding-label">Verified</span>
        </div>
        <div className="grounding-stat">
          <span className="grounding-num">{data.total}</span>
          <span className="grounding-label">Total</span>
        </div>
        <div className="grounding-stat">
          <span className="grounding-num">{data.accuracy}%</span>
          <span className="grounding-label">Accuracy</span>
        </div>
        {data.unsupported > 0 && (
          <div className="grounding-stat warn">
            <span className="grounding-num">{data.unsupported}</span>
            <span className="grounding-label">Auto-fixed</span>
          </div>
        )}
      </div>
    </div>
  );
}

function StepsCard({ data }: { data: { steps: PipelineStep[] } }) {
  const [expanded, setExpanded] = useState(true);
  const steps = data?.steps || [];
  const activeStep = steps.find((s) => s.status === "active");
  const doneCount = steps.filter((s) => s.status === "done").length;

  return (
    <div className="card card-steps">
      <div className="card-header" onClick={() => setExpanded(!expanded)}>
        <span>
          {activeStep
            ? `${activeStep.label}...`
            : doneCount === steps.length
              ? "Pipeline complete"
              : "Pipeline"}
        </span>
        <span className="steps-count">{doneCount}/{steps.length}</span>
      </div>
      {expanded && (
        <div className="steps-list">
          {steps.map((step, i) => (
            <div key={i} className={`step-item step-${step.status}`}>
              <span className="step-indicator">
                {step.status === "done" ? "✓" : step.status === "active" ? <Loader2 size={12} className="spin" /> : "○"}
              </span>
              <div className="step-body">
                <span className="step-label">{step.label}</span>
                {step.summary && <span className="step-summary">{step.summary}</span>}
                {step.status === "done" && step.details?.papers && (
                  <div className="step-papers">
                    {(step.details.papers as SearchPaper[]).slice(0, 5).map((p, j) => (
                      <div key={j} className="step-paper">
                        {p.url ? (
                          <a href={p.url} target="_blank" rel="noopener noreferrer">{p.title}</a>
                        ) : (
                          <span>{p.title}</span>
                        )}
                        <span className="step-paper-meta">
                          {p.authors?.[0]?.split(",")[0]}, {p.year}
                        </span>
                      </div>
                    ))}
                    {step.details.papers.length > 5 && (
                      <div className="step-paper-more">+{step.details.papers.length - 5} more</div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ActionsCard({
  data,
  onAction,
}: {
  data: any;
  onAction: (action: string) => void;
}) {
  return (
    <div className="card card-actions">
      {data.text && <div className="actions-text">{data.text}</div>}
      <div className="actions-btns">
        {data.actions?.map(
          (a: { label: string; value: string }, i: number) => (
            <button
              key={i}
              className="action-btn"
              onClick={() => onAction(a.value)}
            >
              {a.label}
            </button>
          ),
        )}
      </div>
    </div>
  );
}
