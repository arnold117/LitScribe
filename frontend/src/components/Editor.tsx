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
import { useEffect, useCallback, useMemo, useState } from "react";
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
} from "lucide-react";
import DiffView from "./DiffView";

interface EditorProps {
  content: string;
  previousContent: string;
  onSelectionSend: (text: string) => void;
  onExport: (format: string) => void;
  onAcceptChanges: () => void;
  onRevertChanges: () => void;
}

export default function Editor({
  content,
  previousContent,
  onSelectionSend,
  onExport,
  onAcceptChanges,
  onRevertChanges,
}: EditorProps) {
  const [showDiff, setShowDiff] = useState(false);
  const hasChanges = !!previousContent && previousContent !== content;

  useEffect(() => {
    if (hasChanges) setShowDiff(true);
  }, [previousContent]);

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

        {content && (
          <button
            className="toolbar-action"
            onClick={() => onExport("markdown")}
            title="Export Markdown"
          >
            <Download size={13} />
            <span>MD</span>
          </button>
        )}
      </div>

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
        <EditorContent editor={editor} className="editor-content" />
      )}

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
