import { useState, useRef, useEffect, useMemo } from "react";
import {
  Plus,
  FileText,
  Settings,
  PanelLeftClose,
  PanelLeft,
  BookOpen,
  Pencil,
  MessagesSquare,
  History,
  FolderOpen,
  Network,
  RotateCcw,
  Download,
  Paperclip,
  ExternalLink,
  Trash2,
} from "lucide-react";
import type { Conversation, ContentVersion, ReferenceEntry } from "../types";

export interface SidebarFile {
  name: string;
  kind: "upload" | "generated";
  detail: string;
  onDownload?: () => void;
}

type SidebarTab = "sessions" | "versions" | "files" | "graph";

const TABS: { id: SidebarTab; label: string; icon: React.ReactNode }[] = [
  { id: "sessions", label: "Sessions", icon: <MessagesSquare size={14} /> },
  { id: "versions", label: "Versions", icon: <History size={14} /> },
  { id: "files", label: "Files", icon: <FolderOpen size={14} /> },
  { id: "graph", label: "Graph", icon: <Network size={14} /> },
];

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  collapsed: boolean;
  versions: ContentVersion[];
  files: SidebarFile[];
  references: ReferenceEntry[];
  onToggle: () => void;
  onNewReview: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onOpenSettings: () => void;
  onRestoreVersion: (v: ContentVersion) => void;
}

export default function Sidebar({
  conversations,
  activeId,
  collapsed,
  versions,
  files,
  references,
  onToggle,
  onNewReview,
  onSelect,
  onRename,
  onDelete,
  onOpenSettings,
  onRestoreVersion,
}: SidebarProps) {
  const [tab, setTab] = useState<SidebarTab>("sessions");
  const [deleteTarget, setDeleteTarget] = useState<Conversation | null>(null);

  if (collapsed) {
    return (
      <div className="sidebar sidebar-collapsed">
        <button className="sidebar-toggle" onClick={onToggle} title="Expand sidebar">
          <PanelLeft size={16} />
        </button>
        <button className="sidebar-icon-btn" onClick={onNewReview} title="New review">
          <Plus size={16} />
        </button>
        <div className="sidebar-collapsed-sessions">
          {conversations.slice(0, 8).map((c) => (
            <button
              key={c.id}
              className={`sidebar-icon-btn ${activeId === c.id ? "active" : ""}`}
              onClick={() => onSelect(c.id)}
              title={c.title}
            >
              <FileText size={14} />
            </button>
          ))}
        </div>
        <div className="sidebar-bottom">
          <button className="sidebar-icon-btn" onClick={onOpenSettings} title="Settings">
            <Settings size={16} />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <BookOpen size={18} className="brand-icon" />
          <span className="brand-text">LitScribe</span>
        </div>
        <button className="sidebar-toggle" onClick={onToggle} title="Collapse sidebar">
          <PanelLeftClose size={16} />
        </button>
      </div>

      <div className="sidebar-tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`sidebar-tab ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
            title={t.label}
          >
            {t.icon}
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      {tab === "sessions" && (
        <SessionsTab
          conversations={conversations}
          activeId={activeId}
          onNewReview={onNewReview}
          onSelect={onSelect}
          onRename={onRename}
          onDelete={(id) => setDeleteTarget(conversations.find((c) => c.id === id) || null)}
        />
      )}
      {tab === "versions" && (
        <VersionsTab versions={versions} hasActive={!!activeId} onRestore={onRestoreVersion} />
      )}
      {tab === "files" && <FilesTab files={files} hasActive={!!activeId} />}
      {tab === "graph" && <GraphTab references={references} />}

      <div className="sidebar-bottom">
        <button className="sidebar-settings-btn" onClick={onOpenSettings}>
          <Settings size={15} />
          <span>Settings</span>
        </button>
      </div>

      {deleteTarget && (
        <DeleteConfirmModal
          conv={deleteTarget}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={() => {
            onDelete(deleteTarget.id);
            setDeleteTarget(null);
          }}
        />
      )}
    </div>
  );
}

function DeleteConfirmModal({
  conv,
  onCancel,
  onConfirm,
}: {
  conv: Conversation;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
      if (e.key === "Enter") onConfirm();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel, onConfirm]);

  return (
    <div className="setup-overlay" onClick={onCancel}>
      <div className="delete-modal" onClick={(e) => e.stopPropagation()}>
        <div className="delete-modal-icon">
          <Trash2 size={18} />
        </div>
        <div className="delete-modal-title">Delete conversation?</div>
        <div className="delete-modal-name">"{conv.title || "Untitled"}"</div>
        <div className="delete-modal-hint">
          {conv.reviewMeta
            ? `Including its review (${conv.reviewMeta.papers} papers, ${(conv.reviewMeta.words || 0).toLocaleString()} words). `
            : ""}
          This cannot be undone.
        </div>
        <div className="delete-modal-actions">
          <button className="delete-modal-cancel" onClick={onCancel}>Cancel</button>
          <button className="delete-modal-confirm" onClick={onConfirm}>Delete</button>
        </div>
      </div>
    </div>
  );
}

/* ── Sessions ─────────────────────────────── */

function SessionsTab({
  conversations,
  activeId,
  onNewReview,
  onSelect,
  onRename,
  onDelete,
}: {
  conversations: Conversation[];
  activeId: string | null;
  onNewReview: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <>
      <button className="sidebar-new-btn" onClick={onNewReview}>
        <Plus size={15} />
        <span>New Review</span>
      </button>

      <div className="sidebar-sessions">
        {conversations.length === 0 && (
          <div className="sidebar-empty">No conversations yet</div>
        )}
        {conversations.map((c) => (
          <SessionItem
            key={c.id}
            conv={c}
            active={activeId === c.id}
            onSelect={() => onSelect(c.id)}
            onRename={(title) => onRename(c.id, title)}
            onDelete={() => onDelete(c.id)}
          />
        ))}
      </div>
    </>
  );
}

function SessionItem({
  conv,
  active,
  onSelect,
  onRename,
  onDelete,
}: {
  conv: Conversation;
  active: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(conv.title);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) inputRef.current?.focus();
  }, [editing]);

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== conv.title) onRename(trimmed);
    setEditing(false);
  };

  return (
    <div
      className={`sidebar-session ${active ? "active" : ""}`}
      onClick={() => !editing && onSelect()}
    >
      <button
        className="session-delete-btn"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        title="Delete conversation"
      >
        <Trash2 size={12} />
      </button>
      {editing ? (
        <input
          ref={inputRef}
          className="session-rename-input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") { setDraft(conv.title); setEditing(false); }
          }}
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <div className="session-question">
          {conv.title || "Untitled"}
          {active && (
            <button
              className="session-edit-btn"
              onClick={(e) => { e.stopPropagation(); setDraft(conv.title); setEditing(true); }}
              title="Rename"
            >
              <Pencil size={10} />
            </button>
          )}
        </div>
      )}
      <div className="session-meta">
        {conv.reviewMeta ? (
          <>
            <span>{conv.reviewMeta.papers} papers</span>
            <span className="session-dot" />
            <span>{(conv.reviewMeta.words || 0).toLocaleString()} words</span>
            <span className="session-dot" />
            <span>{conv.reviewMeta.score.toFixed(2)}</span>
            <span className="session-dot" />
          </>
        ) : (
          <>
            <span>{conv.messages.length} msgs</span>
            <span className="session-dot" />
          </>
        )}
        <span>{formatDate(conv.createdAt)}</span>
      </div>
    </div>
  );
}

/* ── Versions ─────────────────────────────── */

function VersionsTab({
  versions,
  hasActive,
  onRestore,
}: {
  versions: ContentVersion[];
  hasActive: boolean;
  onRestore: (v: ContentVersion) => void;
}) {
  if (!hasActive || versions.length === 0) {
    return (
      <div className="sidebar-tab-body">
        <div className="sidebar-empty">
          {hasActive ? "No versions yet — versions are saved on review and refine" : "Open a conversation to see its versions"}
        </div>
      </div>
    );
  }

  return (
    <div className="sidebar-tab-body">
      {[...versions].reverse().map((v, i) => {
        const idx = versions.length - 1 - i;
        const isCurrent = idx === versions.length - 1;
        const time = new Date(v.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        return (
          <div
            key={v.timestamp}
            className={`sidebar-version ${isCurrent ? "current" : ""}`}
            onClick={() => !isCurrent && onRestore(v)}
            title={isCurrent ? "Current version" : "Restore this version"}
          >
            <div className="sidebar-version-main">
              <span className="sidebar-version-label">{v.label}</span>
              {isCurrent ? (
                <span className="version-badge">current</span>
              ) : (
                <RotateCcw size={11} className="sidebar-version-restore" />
              )}
            </div>
            <div className="session-meta">
              <span>{time}</span>
              <span className="session-dot" />
              <span>{countWords(v.content).toLocaleString()} words</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Files ────────────────────────────────── */

function FilesTab({ files, hasActive }: { files: SidebarFile[]; hasActive: boolean }) {
  if (files.length === 0) {
    return (
      <div className="sidebar-tab-body">
        <div className="sidebar-empty">
          {hasActive ? "No files yet — uploads and exports will appear here" : "Open a conversation to see its files"}
        </div>
      </div>
    );
  }

  const uploads = files.filter((f) => f.kind === "upload");
  const generated = files.filter((f) => f.kind === "generated");

  return (
    <div className="sidebar-tab-body">
      {uploads.length > 0 && (
        <>
          <div className="sidebar-section-label sidebar-files-label">Uploaded</div>
          {uploads.map((f, i) => <FileRow key={`u-${i}`} file={f} />)}
        </>
      )}
      {generated.length > 0 && (
        <>
          <div className="sidebar-section-label sidebar-files-label">Generated</div>
          {generated.map((f, i) => <FileRow key={`g-${i}`} file={f} />)}
        </>
      )}
    </div>
  );
}

function FileRow({ file }: { file: SidebarFile }) {
  return (
    <div
      className={`sidebar-file ${file.onDownload ? "clickable" : ""}`}
      onClick={() => file.onDownload?.()}
      title={file.onDownload ? "Download" : file.name}
    >
      {file.kind === "upload" ? (
        <Paperclip size={13} className="sidebar-file-icon" />
      ) : (
        <FileText size={13} className="sidebar-file-icon" />
      )}
      <div className="sidebar-file-info">
        <div className="sidebar-file-name">{file.name}</div>
        <div className="sidebar-file-detail">{file.detail}</div>
      </div>
      {file.onDownload && <Download size={12} className="sidebar-file-download" />}
    </div>
  );
}

/* ── Graph ────────────────────────────────── */

function GraphTab({ references }: { references: ReferenceEntry[] }) {
  const [selected, setSelected] = useState<ReferenceEntry | null>(null);

  const layout = useMemo(() => {
    const W = 244;
    const H = 248;
    const cx = W / 2;
    const cy = H / 2;
    const years = references.map((r) => parseInt(r.year)).filter((y) => !isNaN(y));
    const minYear = years.length ? Math.min(...years) : 0;
    const maxYear = years.length ? Math.max(...years) : 0;
    const twoRings = references.length > 12;
    const nodes = references.map((r, i) => {
      const radius = twoRings && i % 2 === 1 ? 68 : 96;
      const angle = (2 * Math.PI * i) / references.length - Math.PI / 2;
      const year = parseInt(r.year);
      // 0 = oldest, 1 = newest; single-year sets render as newest
      const t = !isNaN(year) && maxYear > minYear ? (year - minYear) / (maxYear - minYear) : 1;
      return {
        ref: r,
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
        t,
      };
    });
    return { W, H, cx, cy, nodes, minYear, maxYear };
  }, [references]);

  if (references.length === 0) {
    return (
      <div className="sidebar-tab-body">
        <div className="sidebar-empty">No citation graph yet — generate a review first</div>
      </div>
    );
  }

  const { W, H, cx, cy, nodes, minYear, maxYear } = layout;

  return (
    <div className="sidebar-tab-body sidebar-graph">
      <svg width={W} height={H} className="citation-graph">
        {nodes.map((n, i) => (
          <line key={`l-${i}`} x1={cx} y1={cy} x2={n.x} y2={n.y} className="graph-edge" />
        ))}
        {nodes.map((n, i) => (
          <circle
            key={`n-${i}`}
            cx={n.x}
            cy={n.y}
            r={selected === n.ref ? 7 : 5}
            className={`graph-node ${selected === n.ref ? "selected" : ""}`}
            style={{ fill: yearColor(n.t) }}
            onClick={() => setSelected(selected === n.ref ? null : n.ref)}
          >
            <title>{`${n.ref.authors} (${n.ref.year}) — ${n.ref.title}`}</title>
          </circle>
        ))}
        <circle cx={cx} cy={cy} r={9} className="graph-center" />
        <text x={cx} y={cy + 1} className="graph-center-label" textAnchor="middle" dominantBaseline="middle">
          {references.length}
        </text>
      </svg>

      <div className="graph-legend">
        {minYear > 0 && maxYear > minYear ? (
          <>
            <span className="graph-legend-dot" style={{ background: yearColor(0) }} />
            <span>{minYear}</span>
            <span className="graph-legend-bar" />
            <span>{maxYear}</span>
            <span className="graph-legend-dot" style={{ background: yearColor(1) }} />
          </>
        ) : (
          <span>{references.length} cited papers</span>
        )}
      </div>

      {selected && (
        <div className="graph-detail">
          <div className="graph-detail-title">{selected.title}</div>
          <div className="graph-detail-meta">{selected.authors} ({selected.year})</div>
          {selected.venue && <div className="graph-detail-venue">{selected.venue}</div>}
          {selected.doi && (
            <a
              className="graph-detail-link"
              href={`https://doi.org/${selected.doi}`}
              target="_blank"
              rel="noreferrer"
            >
              <ExternalLink size={11} />
              <span>doi:{selected.doi}</span>
            </a>
          )}
        </div>
      )}
    </div>
  );
}

// Interpolate from muted gray (oldest) to accent indigo (newest)
function yearColor(t: number): string {
  const from = [82, 82, 91];
  const to = [129, 140, 248];
  const c = from.map((f, i) => Math.round(f + (to[i] - f) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

function countWords(text: string): number {
  const cjk = (text.match(/[一-鿿㐀-䶿]/g) || []).length;
  const latin = text.replace(/[一-鿿㐀-䶿]/g, "").split(/\s+/).filter(Boolean).length;
  return cjk + latin;
}

function formatDate(iso: string) {
  if (!iso) return "";
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 86400000) return "Today";
  if (diff < 172800000) return "Yesterday";
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}
