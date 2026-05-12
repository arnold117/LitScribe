import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Placeholder from "@tiptap/extension-placeholder";
import Highlight from "@tiptap/extension-highlight";
import Underline from "@tiptap/extension-underline";
import { Markdown } from "tiptap-markdown";
import { useEffect, useCallback } from "react";

interface EditorProps {
  content: string;
  onSelectionSend: (text: string) => void;
}

export default function Editor({ content, onSelectionSend }: EditorProps) {
  const editor = useEditor({
    extensions: [
      StarterKit,
      Placeholder.configure({ placeholder: "Your review will appear here..." }),
      Highlight,
      Underline,
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
    if (text.trim()) {
      onSelectionSend(text.trim());
    }
  }, [editor, onSelectionSend]);

  return (
    <div className="editor-container">
      <div className="editor-toolbar">
        <span className="toolbar-title">Editor</span>
        <button
          className="toolbar-btn"
          onClick={handleSendSelection}
          title="Send selection to chat"
        >
          Send to Chat
        </button>
      </div>
      <EditorContent editor={editor} className="editor-content" />
    </div>
  );
}
