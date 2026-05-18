import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Placeholder from "@tiptap/extension-placeholder";
import Highlight from "@tiptap/extension-highlight";
import Underline from "@tiptap/extension-underline";
import { Table } from "@tiptap/extension-table";
import { TableRow } from "@tiptap/extension-table-row";
import { TableCell } from "@tiptap/extension-table-cell";
import { TableHeader } from "@tiptap/extension-table-header";
import { Markdown } from "tiptap-markdown";
import { useEffect, useCallback, useMemo, useState, useRef } from "react";
import {
  Bold,
  Italic,
  Underline as UnderlineIcon,
  Heading1,
  Heading2,
  Heading3,
  List,
  ListOrdered,
  Quote,
  Code,
  Highlighter,
  Send,
  Download,
  GitCompareArrows,
  ChevronDown,
  ChevronRight,
  History,
  RotateCcw,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import DiffView from "./DiffView";
import type { ReferenceEntry, ContentVersion, CitationFormat } from "../types";

interface EditorProps {
  content: string;
  previousContent: string;
  appendixContent: string;
  references: ReferenceEntry[];
  versions: ContentVersion[];
  onSelectionSend: (text: string) => void;
  onExport: (format: string, includeAppendix?: boolean, citeFormat?: CitationFormat) => void;
  onAcceptChanges: () => void;
  onRevertChanges: () => void;
  onRestoreVersion: (v: ContentVersion) => void;
}

export default function Editor({
  content,
  previousContent,
  appendixContent,
  references,
  versions,
  onSelectionSend,
  onExport,
  onAcceptChanges,
  onRevertChanges,
  onRestoreVersion,
}: EditorProps) {
  const [showDiff, setShowDiff] = useState(false);
  const [appendixExpanded, setAppendixExpanded] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [showVersions, setShowVersions] = useState(false);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; ref: ReferenceEntry } | null>(null);
  const editorWrapRef = useRef<HTMLDivElement>(null);
  const hasChanges = !!previousContent && previousContent !== content;

  useEffect(() => {
    if (hasChanges) setShowDiff(true);
  }, [previousContent]);

  const refMap = useMemo(() => {
    const map = new Map<string, ReferenceEntry>();
    for (const r of references) {
      const surname = r.authors.split(",")[0].trim();
      map.set(`${surname}, ${r.year}`, r);
      map.set(`${surname} et al., ${r.year}`, r);
      const parts = r.authors.split(",");
      if (parts.length > 1) {
        const firstName = parts[0].trim();
        map.set(`${firstName}, ${r.year}`, r);
        map.set(`${firstName} et al., ${r.year}`, r);
      }
    }
    return map;
  }, [references]);

  useEffect(() => {
    const el = editorWrapRef.current;
    if (!el || references.length === 0) return;

    const handleMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target.classList.contains("citation-ref")) {
        const key = target.getAttribute("data-cite");
        if (key) {
          const ref = refMap.get(key);
          if (ref) {
            const rect = target.getBoundingClientRect();
            const containerRect = el.getBoundingClientRect();
            setTooltip({
              x: rect.left - containerRect.left + rect.width / 2,
              y: rect.top - containerRect.top - 4,
              ref,
            });
          }
        }
      }
    };
    const handleMouseOut = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target.classList.contains("citation-ref")) setTooltip(null);
    };
    el.addEventListener("mouseover", handleMouseOver);
    el.addEventListener("mouseout", handleMouseOut);
    return () => {
      el.removeEventListener("mouseover", handleMouseOver);
      el.removeEventListener("mouseout", handleMouseOut);
    };
  }, [refMap, references]);

  useEffect(() => {
    const el = editorWrapRef.current;
    if (!el || references.length === 0) return;
    const timer = setTimeout(() => annotateCitations(el, refMap), 200);
    return () => clearTimeout(timer);
  }, [content, references]);

  const editor = useEditor({
    extensions: [
      StarterKit,
      Placeholder.configure({
        placeholder: "Start a review from the chat panel, or drop an outline file →",
      }),
      Highlight,
      Underline,
      Table.configure({ resizable: false }),
      TableRow,
      TableCell,
      TableHeader,
      Markdown,
    ],
    content: "",
    editable: true,
  });

  useEffect(() => {
    if (editor && content) {
      editor.commands.setContent(content);
    }
  }, [editor, content]);

  const handleSendSelection = useCallback(() => {
    if (!editor) return;
    const { from, to } = editor.state.selection;
    if (from === to) return;
    const text = editor.state.doc.textBetween(from, to, " ");
    if (text.trim()) onSelectionSend(text.trim());
  }, [editor, onSelectionSend]);

  const wordCount = useMemo(() => {
    if (!editor) return 0;
    const text = editor.state.doc.textContent;
    const cjk = (text.match(/[一-鿿㐀-䶿]/g) || []).length;
    const latin = text
      .replace(/[一-鿿㐀-䶿]/g, "")
      .split(/\s+/)
      .filter(Boolean).length;
    return cjk + latin;
  }, [editor?.state.doc.content]);

  const hasSelection = editor
    ? editor.state.selection.from !== editor.state.selection.to
    : false;

  return (
    <div className="editor-container">
      <div className="editor-toolbar">
        <div className="toolbar-group">
          <ToolbarButton
            icon={<Bold size={14} />}
            active={editor?.isActive("bold")}
            onClick={() => editor?.chain().focus().toggleBold().run()}
            title="Bold"
          />
          <ToolbarButton
            icon={<Italic size={14} />}
            active={editor?.isActive("italic")}
            onClick={() => editor?.chain().focus().toggleItalic().run()}
            title="Italic"
          />
          <ToolbarButton
            icon={<UnderlineIcon size={14} />}
            active={editor?.isActive("underline")}
            onClick={() => editor?.chain().focus().toggleUnderline().run()}
            title="Underline"
          />
          <ToolbarButton
            icon={<Highlighter size={14} />}
            active={editor?.isActive("highlight")}
            onClick={() => editor?.chain().focus().toggleHighlight().run()}
            title="Highlight"
          />
        </div>

        <div className="toolbar-divider" />

        <div className="toolbar-group">
          <ToolbarButton
            icon={<Heading1 size={14} />}
            active={editor?.isActive("heading", { level: 1 })}
            onClick={() =>
              editor?.chain().focus().toggleHeading({ level: 1 }).run()
            }
            title="Heading 1"
          />
          <ToolbarButton
            icon={<Heading2 size={14} />}
            active={editor?.isActive("heading", { level: 2 })}
            onClick={() =>
              editor?.chain().focus().toggleHeading({ level: 2 }).run()
            }
            title="Heading 2"
          />
          <ToolbarButton
            icon={<Heading3 size={14} />}
            active={editor?.isActive("heading", { level: 3 })}
            onClick={() =>
              editor?.chain().focus().toggleHeading({ level: 3 }).run()
            }
            title="Heading 3"
          />
        </div>

        <div className="toolbar-divider" />

        <div className="toolbar-group">
          <ToolbarButton
            icon={<List size={14} />}
            active={editor?.isActive("bulletList")}
            onClick={() => editor?.chain().focus().toggleBulletList().run()}
            title="Bullet list"
          />
          <ToolbarButton
            icon={<ListOrdered size={14} />}
            active={editor?.isActive("orderedList")}
            onClick={() => editor?.chain().focus().toggleOrderedList().run()}
            title="Numbered list"
          />
          <ToolbarButton
            icon={<Quote size={14} />}
            active={editor?.isActive("blockquote")}
            onClick={() => editor?.chain().focus().toggleBlockquote().run()}
            title="Block quote"
          />
          <ToolbarButton
            icon={<Code size={14} />}
            active={editor?.isActive("codeBlock")}
            onClick={() => editor?.chain().focus().toggleCodeBlock().run()}
            title="Code block"
          />
        </div>

        <div className="toolbar-spacer" />

        {hasChanges && (
          <button
            className={`toolbar-action ${showDiff ? "toolbar-action-active" : ""}`}
            onClick={() => setShowDiff(!showDiff)}
            title="Toggle track changes"
          >
            <GitCompareArrows size={13} />
            <span>Changes</span>
          </button>
        )}

        {hasSelection && (
          <button
            className="toolbar-action"
            onClick={handleSendSelection}
            title="Send selection to chat"
          >
            <Send size={13} />
            <span>Chat</span>
          </button>
        )}

        {versions.length > 1 && (
          <div className="export-menu-container">
            <button
              className={`toolbar-action ${showVersions ? "toolbar-action-active" : ""}`}
              onClick={() => { setShowVersions(!showVersions); setShowExportMenu(false); }}
              title="Version history"
            >
              <History size={13} />
              <span>v{versions.length}</span>
            </button>
            {showVersions && (
              <div className="export-dropdown version-dropdown" onMouseLeave={() => setShowVersions(false)}>
                {[...versions].reverse().map((v, i) => {
                  const idx = versions.length - 1 - i;
                  const isCurrent = idx === versions.length - 1;
                  const time = new Date(v.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                  return (
                    <button
                      key={v.timestamp}
                      className={isCurrent ? "version-current" : ""}
                      onClick={() => {
                        if (!isCurrent) onRestoreVersion(v);
                        setShowVersions(false);
                      }}
                    >
                      <span className="version-label">{v.label}</span>
                      <span className="version-time">{time}</span>
                      {!isCurrent && <RotateCcw size={11} />}
                      {isCurrent && <span className="version-badge">current</span>}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {content && (
          <div className="export-menu-container">
            <button
              className="toolbar-action"
              onClick={() => { setShowExportMenu(!showExportMenu); setShowVersions(false); }}
              title="Export"
            >
              <Download size={13} />
              <span>Export</span>
            </button>
            {showExportMenu && (
              <div className="export-dropdown" onMouseLeave={() => setShowExportMenu(false)}>
                <div className="export-section-label">Citation format</div>
                <button onClick={() => { onExport("markdown", !appendixContent, "bracket"); setShowExportMenu(false); }}>
                  [Author, Year] — default
                </button>
                <button onClick={() => { onExport("markdown", !appendixContent, "apa"); setShowExportMenu(false); }}>
                  (Author, Year) — APA
                </button>
                <button onClick={() => { onExport("markdown", !appendixContent, "vancouver"); setShowExportMenu(false); }}>
                  [1], [2] — Vancouver
                </button>
                {appendixContent && (
                  <>
                    <div className="export-divider" />
                    <button onClick={() => { onExport("markdown", false); setShowExportMenu(false); }}>
                      Core only (no appendix)
                    </button>
                  </>
                )}
                <div className="export-divider" />
                <button onClick={() => { onExport("bibtex"); setShowExportMenu(false); }}>
                  BibTeX (.bib)
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="editor-body" ref={editorWrapRef}>
        {showDiff && hasChanges ? (
          <DiffView
            oldText={previousContent}
            newText={content}
            onAccept={() => {
              onAcceptChanges();
              setShowDiff(false);
            }}
            onRevert={() => {
              onRevertChanges();
              setShowDiff(false);
            }}
          />
        ) : (
          <div className="editor-scroll">
            <EditorContent editor={editor} className="editor-content" />

            {appendixContent && (
              <div className="appendix-panel">
                <button
                  className="appendix-toggle"
                  onClick={() => setAppendixExpanded(!appendixExpanded)}
                >
                  {appendixExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  <span>Supplementary Materials</span>
                  <span className="appendix-badge">Appendix</span>
                </button>
                {appendixExpanded && (
                  <div className="appendix-content">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {appendixContent}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {tooltip && (
          <div
            className="citation-tooltip"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            <div className="citation-tooltip-title">{tooltip.ref.title}</div>
            <div className="citation-tooltip-authors">{tooltip.ref.authors} ({tooltip.ref.year})</div>
            {tooltip.ref.venue && <div className="citation-tooltip-venue">{tooltip.ref.venue}</div>}
          </div>
        )}
      </div>

      <div className="editor-footer">
        <span className="editor-word-count">
          {wordCount > 0 ? `${wordCount.toLocaleString()} words` : ""}
        </span>
      </div>
    </div>
  );
}

function ToolbarButton({
  icon,
  active,
  onClick,
  title,
}: {
  icon: React.ReactNode;
  active?: boolean;
  onClick?: () => void;
  title: string;
}) {
  return (
    <button
      className={`toolbar-btn ${active ? "active" : ""}`}
      onClick={onClick}
      title={title}
      type="button"
    >
      {icon}
    </button>
  );
}

const CITE_RE = /\[([A-Z一-鿿][^\[\]]{2,40}?,\s*\d{4}[a-z]?)\]/g;

function annotateCitations(container: HTMLElement, refMap: Map<string, ReferenceEntry>) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const replacements: { node: Text; spans: { start: number; end: number; key: string }[] }[] = [];

  let node: Text | null;
  while ((node = walker.nextNode() as Text | null)) {
    if (node.parentElement?.closest(".appendix-content, .citation-ref, .citation-tooltip")) continue;
    const text = node.textContent || "";
    const spans: { start: number; end: number; key: string }[] = [];
    let match;
    CITE_RE.lastIndex = 0;
    while ((match = CITE_RE.exec(text))) {
      const key = match[1];
      if (refMap.has(key)) {
        spans.push({ start: match.index, end: match.index + match[0].length, key });
      }
    }
    if (spans.length > 0) replacements.push({ node, spans });
  }

  for (const { node, spans } of replacements) {
    const parent = node.parentNode;
    if (!parent) continue;
    const text = node.textContent || "";
    const frag = document.createDocumentFragment();
    let lastEnd = 0;
    for (const { start, end, key } of spans) {
      if (start > lastEnd) frag.appendChild(document.createTextNode(text.slice(lastEnd, start)));
      const span = document.createElement("span");
      span.className = "citation-ref";
      span.setAttribute("data-cite", key);
      span.textContent = text.slice(start, end);
      frag.appendChild(span);
      lastEnd = end;
    }
    if (lastEnd < text.length) frag.appendChild(document.createTextNode(text.slice(lastEnd)));
    parent.replaceChild(frag, node);
  }
}
