import { useState, useRef, useEffect } from "react";
import {
  Plus,
  FileText,
  Settings,
  PanelLeftClose,
  PanelLeft,
  BookOpen,
  Pencil,
} from "lucide-react";
import type { Conversation } from "../types";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  collapsed: boolean;
  onToggle: () => void;
  onNewReview: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onOpenSettings: () => void;
}

export default function Sidebar({
  conversations,
  activeId,
  collapsed,
  onToggle,
  onNewReview,
  onSelect,
  onRename,
  onOpenSettings,
}: SidebarProps) {
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

      <button className="sidebar-new-btn" onClick={onNewReview}>
        <Plus size={15} />
        <span>New Review</span>
      </button>

      <div className="sidebar-section-label">Conversations</div>

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
          />
        ))}
      </div>

      <div className="sidebar-bottom">
        <button className="sidebar-settings-btn" onClick={onOpenSettings}>
          <Settings size={15} />
          <span>Settings</span>
        </button>
      </div>
    </div>
  );
}

function SessionItem({
  conv,
  active,
  onSelect,
  onRename,
}: {
  conv: Conversation;
  active: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
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

function formatDate(iso: string) {
  if (!iso) return "";
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 86400000) return "Today";
  if (diff < 172800000) return "Yesterday";
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}
