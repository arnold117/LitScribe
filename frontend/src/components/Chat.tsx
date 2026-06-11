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
  Copy,
  Check,
  Square,
} from "lucide-react";
import type { ChatMessage, PlanSection, PipelineStep, SearchPaper } from "../types";

interface ChatProps {
  messages: ChatMessage[];
  onSend: (message: string, attachment?: File) => void;
  onStop: () => void;
  onPlanUpdate: (sections: PlanSection[], constraints: string) => void;
  onPlanExecute: () => void;
  selectionContext: string;
  onClearSelection: () => void;
  seedInput?: string;
  onSeedConsumed?: () => void;
  loading: boolean;
}

const SLASH_COMMANDS = [
  {
    cmd: "/grill-me",
    label: "质询我的研究思路 · adversarial questioning",
    prompt:
      "扮演一位严格但建设性的审稿人。针对当前综述（或我正在研究的问题），连续提出尖锐的问题：方法选择的依据、与已有工作的真正差异、潜在的薄弱环节、被忽略的反例。每轮提 3 个问题，等我回答后基于回答继续追问。",
  },
  {
    cmd: "/diagnose",
    label: "写作诊断 · writing diagnosis",
    prompt:
      "对当前综述做一次写作诊断：结构完整性、各节论证密度、引用分布是否均衡、过度 hedging 的表述、重复或冗余的内容。按优先级列出具体修改建议，标注所在章节。",
  },
  {
    cmd: "/abstract",
    label: "生成摘要 · abstract (EN + 中文)",
    prompt: "为当前综述生成约 200 词的学术摘要：先英文版，再中文版。突出研究空白、综述范围与核心结论。",
  },
  {
    cmd: "/consistency",
    label: "跨节一致性检查 · consistency check",
    prompt: "Check cross-section consistency",
  },
  {
    cmd: "/search",
    label: "文献搜索 · literature search",
    prompt: "帮我搜索以下主题的最新文献，按相关性总结每篇的核心贡献：",
  },
];

export default function Chat({
  messages,
  onSend,
  onStop,
  onPlanUpdate,
  onPlanExecute,
  selectionContext,
  onClearSelection,
  seedInput,
  onSeedConsumed,
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

  // Prefill from onboarding example (fills box, user reviews & hits enter)
  useEffect(() => {
    if (seedInput) {
      setInput(seedInput);
      inputRef.current?.focus();
      onSeedConsumed?.();
    }
  }, [seedInput]);

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

  const slashMatches =
    input.startsWith("/") && !input.includes("\n")
      ? SLASH_COMMANDS.filter((c) => c.cmd.startsWith(input.trim()))
      : [];

  const pickSlash = (c: (typeof SLASH_COMMANDS)[number]) => {
    setInput(c.prompt);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (slashMatches.length > 0) pickSlash(slashMatches[0]);
      else handleSubmit();
    }
    if (e.key === "Escape" && slashMatches.length > 0) setInput("");
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
            ) : msg.type === "analysis" ? (
              <AnalysisCard data={msg.data} />
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
                {msg.content && msg.role === "assistant" && (
                  <CopyButton text={msg.content} />
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
        {slashMatches.length > 0 && (
          <div className="slash-menu">
            {slashMatches.map((c) => (
              <button key={c.cmd} className="slash-item" onClick={() => pickSlash(c)}>
                <span className="slash-cmd">{c.cmd}</span>
                <span className="slash-label">{c.label}</span>
              </button>
            ))}
          </div>
        )}
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
                : 'Ask anything, type "/" for commands, or drop a file...'
            }
            rows={2}
            disabled={loading}
          />
          {loading ? (
            <button
              className="chat-send-btn chat-stop-btn"
              onClick={onStop}
              title="Stop generation"
            >
              <Square size={13} fill="currentColor" />
            </button>
          ) : (
            <button
              className="chat-send-btn"
              onClick={handleSubmit}
              disabled={!input.trim() && !selectionContext}
            >
              <ArrowUp size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <button className="chat-copy-btn" onClick={handleCopy} title="Copy">
      {copied ? <Check size={12} /> : <Copy size={12} />}
    </button>
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

  const toggleAll = (enabled: boolean) => {
    const updated = sections.map((s) => ({ ...s, enabled }));
    setSections(updated);
    onUpdate(updated, constraints);
  };

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
          <div className="plan-bulk-actions">
            <button onClick={() => toggleAll(true)} disabled={enabledCount === sections.length}>Select all</button>
            <button onClick={() => toggleAll(false)} disabled={enabledCount === 0}>Deselect all</button>
          </div>
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
  const { current, total, title, papers, words, stage } = data || {};
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
      {stage && (
        <div className="progress-stage">
          <Loader2 size={11} className="spin" />
          <span>{stage}</span>
        </div>
      )}
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

function AnalysisCard({ data }: { data: any }) {
  if (!data) return null;
  const maxCites = Math.max(1, ...(data.per_section || []).map((s: any) => s.citations));
  const flags: { label: string; tone: string }[] = data.flags || [];

  const metrics = [
    { label: "字数", value: data.word_count?.toLocaleString() },
    { label: "句数", value: data.sentence_count },
    { label: "平均句长", value: data.avg_sentence_length },
    { label: "词汇多样性", value: data.lexical_diversity },
    { label: "引用数", value: data.citation_count },
    { label: "引用密度/千字", value: data.citation_density },
    { label: "带引用句占比", value: `${Math.round((data.cited_sentence_ratio || 0) * 100)}%` },
    { label: "模糊措辞", value: data.hedge_count },
  ];

  return (
    <div className="card card-analysis">
      <div className="analysis-header">写作分析 · Writing Analysis</div>
      <div className="analysis-grid">
        {metrics.map((m) => (
          <div className="analysis-metric" key={m.label}>
            <span className="analysis-num">{m.value}</span>
            <span className="analysis-label">{m.label}</span>
          </div>
        ))}
      </div>
      {data.per_section?.length > 0 && (
        <div className="analysis-dist">
          <div className="analysis-dist-title">引用分布</div>
          {data.per_section.map((s: any, i: number) => (
            <div className="analysis-bar-row" key={i}>
              <span className="analysis-bar-label" title={s.title}>{s.title}</span>
              <div className="analysis-bar-track">
                <div className="analysis-bar-fill" style={{ width: `${(s.citations / maxCites) * 100}%` }} />
              </div>
              <span className="analysis-bar-num">{s.citations}</span>
            </div>
          ))}
        </div>
      )}
      <div className="analysis-flags">
        {flags.map((f, i) => (
          <span key={i} className={`analysis-flag ${f.tone}`}>{f.label}</span>
        ))}
      </div>
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
