import { useState, useEffect } from "react";
import { X, FileText, Plus, Trash2, Loader2, ChevronLeft } from "lucide-react";
import { fetchTemplates, createTemplate, deleteTemplate } from "../api";

interface Template {
  id: string;
  label: string;
  builtin: boolean;
  needs_papers: boolean;
}

interface TemplatesModalProps {
  hasReview: boolean;
  onClose: () => void;
  onApply: (id: string, instructions: string, wordCount: number) => void;
}

export default function TemplatesModal({ hasReview, onClose, onApply }: TemplatesModalProps) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [selected, setSelected] = useState<Template | null>(null);
  const [instructions, setInstructions] = useState("");
  const [wordCount, setWordCount] = useState(800);
  const [creating, setCreating] = useState(false);

  const load = () => fetchTemplates().then((d) => setTemplates(d.templates || [])).catch(() => {});
  useEffect(() => { load(); }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const handleDelete = async (id: string) => {
    await deleteTemplate(id).catch(() => {});
    load();
  };

  if (creating) {
    return <CreateForm onCancel={() => setCreating(false)} onCreated={() => { setCreating(false); load(); }} />;
  }

  return (
    <div className="setup-overlay" onClick={onClose}>
      <div className="tpl-modal" onClick={(e) => e.stopPropagation()}>
        <button className="setup-close" onClick={onClose}><X size={16} /></button>

        {!selected ? (
          <>
            <div className="tpl-title">写作模板</div>
            <div className="tpl-sub">基于当前综述的文献，生成相关工作、摘要、翻译、审稿回复等</div>
            <div className="tpl-list">
              {templates.map((t) => (
                <div
                  key={t.id}
                  className={`tpl-item ${t.needs_papers && !hasReview ? "disabled" : ""}`}
                  onClick={() => { if (!(t.needs_papers && !hasReview)) { setSelected(t); } }}
                  title={t.needs_papers && !hasReview ? "需要先生成综述" : ""}
                >
                  <FileText size={14} className="tpl-item-icon" />
                  <span className="tpl-item-label">{t.label}</span>
                  {!t.builtin && <span className="tpl-tag">自定义</span>}
                  {t.needs_papers && <span className="tpl-need">需论文</span>}
                  {!t.builtin && (
                    <button className="tpl-del" onClick={(e) => { e.stopPropagation(); handleDelete(t.id); }} title="删除">
                      <Trash2 size={11} />
                    </button>
                  )}
                </div>
              ))}
            </div>
            <button className="tpl-new" onClick={() => setCreating(true)}>
              <Plus size={13} /><span>新建自定义模板</span>
            </button>
          </>
        ) : (
          <>
            <button className="tpl-back" onClick={() => setSelected(null)}>
              <ChevronLeft size={14} /><span>返回</span>
            </button>
            <div className="tpl-title">{selected.label}</div>
            <label className="tpl-field">
              <span>你的说明 / 上下文{!selected.needs_papers ? "（翻译/润色：把要处理的文本贴这里）" : ""}</span>
              <textarea
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                rows={6}
                placeholder={selected.needs_papers
                  ? "例如：本文提出一种基于扩散模型的时序预测方法，强调长程依赖建模…"
                  : "把要翻译/润色的文本粘贴到这里…"}
                autoFocus
              />
            </label>
            {selected.needs_papers && (
              <label className="tpl-field-inline">
                <span>目标字数</span>
                <input
                  type="number"
                  value={wordCount}
                  min={100}
                  step={100}
                  onChange={(e) => setWordCount(parseInt(e.target.value) || 800)}
                />
              </label>
            )}
            <button
              className="tpl-run"
              disabled={!instructions.trim()}
              onClick={() => { onApply(selected.id, instructions.trim(), wordCount); onClose(); }}
            >
              生成
            </button>
          </>
        )}
      </div>
    </div>
  );
}

function CreateForm({ onCancel, onCreated }: { onCancel: () => void; onCreated: () => void }) {
  const [id, setId] = useState("");
  const [label, setLabel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [saving, setSaving] = useState(false);

  const valid = id.trim() && label.trim() && prompt.trim();

  const save = async () => {
    setSaving(true);
    await createTemplate(id.trim(), label.trim(), prompt).catch(() => {});
    setSaving(false);
    onCreated();
  };

  return (
    <div className="setup-overlay" onClick={onCancel}>
      <div className="tpl-modal" onClick={(e) => e.stopPropagation()}>
        <button className="tpl-back" onClick={onCancel}><ChevronLeft size={14} /><span>返回</span></button>
        <div className="tpl-title">新建自定义模板</div>
        <div className="tpl-sub">
          Prompt 里可用占位符：<code>{"{user_instructions}"}</code> <code>{"{papers}"}</code> <code>{"{num_papers}"}</code> <code>{"{word_count}"}</code>
        </div>
        <label className="tpl-field"><span>ID（英文，唯一）</span>
          <input value={id} onChange={(e) => setId(e.target.value)} placeholder="my-template" /></label>
        <label className="tpl-field"><span>名称（列表显示）</span>
          <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="我的模板" /></label>
        <label className="tpl-field"><span>Prompt 模板</span>
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={8}
            placeholder={"基于以下论文撰写……\n\n{user_instructions}\n\n论文（{num_papers} 篇）：\n{papers}\n\n约 {word_count} 字。"} /></label>
        <button className="tpl-run" disabled={!valid || saving} onClick={save}>
          {saving ? <Loader2 size={14} className="spin" /> : "保存模板"}
        </button>
      </div>
    </div>
  );
}
