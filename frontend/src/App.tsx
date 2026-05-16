import { useState, useCallback, useEffect, useRef } from "react";
import Sidebar from "./components/Sidebar";
import Editor from "./components/Editor";
import Chat from "./components/Chat";
import SetupWizard from "./components/SetupWizard";
import type { ChatMessage, PlanSection, PlanState, Conversation, PipelineStep } from "./types";
import {
  checkHealth,
  startReview,
  refineReview,
  sendChat,
  uploadOutline,
  startOutlineReview,
  exportReview,
  readSSE,
} from "./api";
import "./App.css";

let msgId = 0;
const uid = () => `msg-${++msgId}`;

const STORAGE_KEY = "litscribe-conversations";

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveConversations(convs: Conversation[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(convs));
  } catch {}
}

function extractTitle(message: string): string {
  let text = message
    .replace(/\[Uploaded:.*?\]/g, "")
    .replace(/\[Selected:.*?\]/gs, "")
    .trim();
  if (!text) text = message;
  return text.slice(0, 60) || "New conversation";
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(null);

  const [editorContent, setEditorContent] = useState("");
  const [previousContent, setPreviousContent] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [selectionContext, setSelectionContext] = useState("");
  const [loading, setLoading] = useState(false);
  const [plan, setPlan] = useState<PlanState | null>(null);
  const [pendingOutline, setPendingOutline] = useState<{
    sections: PlanSection[];
    text: string;
    language: string;
    constraints: string;
  } | null>(null);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showSetup, setShowSetup] = useState(false);

  // Ref to always have latest state in async callbacks
  const stateRef = useRef({ messages, editorContent, previousContent, plan, pendingOutline });
  stateRef.current = { messages, editorContent, previousContent, plan, pendingOutline };

  useEffect(() => {
    checkHealth()
      .then((h) => { if (!h.configured) setShowSetup(true); })
      .catch(() => {});
  }, []);

  // --- Persist conversations ---

  const persistConversation = useCallback((convId: string | null = activeId) => {
    if (!convId) return;
    const s = stateRef.current;
    setConversations((prev) => {
      const idx = prev.findIndex((c) => c.id === convId);
      if (idx < 0) return prev;
      const updated = [...prev];
      updated[idx] = {
        ...updated[idx],
        messages: s.messages,
        editorContent: s.editorContent,
        previousContent: s.previousContent,
        plan: s.plan,
        pendingOutline: s.pendingOutline,
      };
      saveConversations(updated);
      return updated;
    });
  }, [activeId]);

  // Auto-save current conversation when messages or editor change
  useEffect(() => {
    if (activeId && messages.length > 0) {
      const timer = setTimeout(() => persistConversation(), 500);
      return () => clearTimeout(timer);
    }
  }, [messages, editorContent, activeId]);

  // --- Create conversation on first message ---

  function ensureConversation(firstMessage: string): string {
    if (activeId) return activeId;

    const id = `conv-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const conv: Conversation = {
      id,
      title: extractTitle(firstMessage),
      createdAt: new Date().toISOString(),
      messages: [],
      editorContent: "",
      previousContent: "",
      plan: null,
      pendingOutline: null,
    };
    setConversations((prev) => {
      const updated = [conv, ...prev];
      saveConversations(updated);
      return updated;
    });
    setActiveId(id);
    return id;
  }

  // --- Switch conversation ---

  const handleSelectConversation = (id: string) => {
    if (id === activeId) return;

    // Save current
    persistConversation();

    // Load target
    const target = conversations.find((c) => c.id === id);
    if (!target) return;

    setActiveId(id);
    setMessages(target.messages);
    setEditorContent(target.editorContent);
    setPreviousContent(target.previousContent);
    setPlan(target.plan);
    setPendingOutline(target.pendingOutline);
    setSelectionContext("");
  };

  const handleNewReview = () => {
    persistConversation();
    setActiveId(null);
    setEditorContent("");
    setPreviousContent("");
    setMessages([]);
    setPlan(null);
    setPendingOutline(null);
    setSelectionContext("");
  };

  // --- Helpers ---

  const addMsg = (msg: Omit<ChatMessage, "id" | "timestamp">) =>
    setMessages((prev) => [...prev, { ...msg, id: uid(), timestamp: Date.now() }]);

  const updateLastProgress = (content: string, data?: any) =>
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.type === "progress") {
        return [
          ...prev.slice(0, -1),
          { ...last, content, data, timestamp: Date.now() },
        ];
      }
      return [
        ...prev,
        { id: uid(), role: "assistant" as const, content, type: "progress" as const, data, timestamp: Date.now() },
      ];
    });

  // --- Plan handling ---

  const handlePlanUpdate = useCallback(
    (sections: PlanSection[], constraints: string) => {
      setPlan((prev) => (prev ? { ...prev, sections, constraints } : null));
    },
    [],
  );

  const handlePlanExecute = useCallback(async () => {
    if (!plan) return;
    const enabled = plan.sections.filter((s) => s.enabled);
    if (!enabled.length) return;

    setLoading(true);
    addMsg({
      role: "user",
      content: `Generate ${enabled.length} sections${plan.constraints ? ` (constraints: ${plan.constraints.slice(0, 80)}...)` : ""}`,
    });

    try {
      const sectionFilter = enabled.map((s) => s.number).join(",");
      const res = await startOutlineReview(
        plan.outlineText,
        plan.language,
        plan.maxPapers,
        plan.constraints,
        sectionFilter,
      );
      await readSSE(res, handleOutlineEvent);
    } catch (err: any) {
      addMsg({ role: "assistant", content: `Error: ${err.message}` });
    } finally {
      setLoading(false);
    }
  }, [plan]);

  function handleOutlineEvent(event: string, data: any) {
    if (event === "section_start") {
      updateLastProgress(`[${data.index + 1}/${data.total}] ${data.title}`, {
        current: data.index + 1, total: data.total, title: data.title,
      });
    } else if (event === "section_done") {
      updateLastProgress(`[${data.index + 1}/${data.total}] ${data.title}`, {
        current: data.index + 1, total: data.total, title: data.title,
        papers: data.papers, words: data.words,
      });
    } else if (event === "complete") {
      setEditorContent(data.text);
      setMessages((prev) => [
        ...prev.filter((m) => m.type !== "progress"),
        { id: uid(), role: "assistant", content: `Done: ${data.total_words} words, ${data.total_papers} papers (${data.time}s)`, timestamp: Date.now() },
      ]);

      if (data.coverage) {
        addMsg({ role: "assistant", content: "", type: "coverage", data: data.coverage });
      }

      addMsg({
        role: "assistant",
        content: "",
        type: "actions",
        data: {
          text: "What's next?",
          actions: [
            { label: "Check consistency", value: "Check cross-section consistency" },
            { label: "Export Markdown", value: "__export_md__" },
            { label: "Coverage report", value: "Show coverage report" },
          ],
        },
      });

      // Update conversation with review metadata
      setConversations((prev) => {
        const updated = prev.map((c) =>
          c.id === activeId
            ? { ...c, reviewMeta: { papers: data.total_papers, words: data.total_words, score: 0 } }
            : c,
        );
        saveConversations(updated);
        return updated;
      });
    }
  }

  // --- Chat message handling ---

  const handleSend = useCallback(
    async (message: string, attachment?: File) => {
      if (message === "__show_plan__") { showPlanFromOutline(); return; }
      if (message === "__generate_all__") { await generateAllFromOutline(); return; }
      if (message === "__export_md__") { await handleExport("markdown"); return; }
      if (message === "__export_bib__") { await handleExport("bibtex"); return; }

      // Create conversation on first real message
      ensureConversation(message);

      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        content: message,
        timestamp: Date.now(),
        attachment: attachment ? { name: attachment.name, type: attachment.type } : undefined,
      };
      setMessages((prev) => [...prev, userMsg]);

      if (attachment) {
        setLoading(true);
        try {
          await handleFileUpload(attachment, message);
        } catch (err: any) {
          addMsg({ role: "assistant", content: `Error: ${err.message}` });
        } finally {
          setLoading(false);
        }
        return;
      }

      const isReview = /review|综述|generate|生成/i.test(message) && !message.startsWith("[Selected:");
      const isRefine = message.startsWith("[Selected:") || /改|修改|refine|rewrite|expand|缩减|展开/i.test(message);

      setLoading(true);
      try {
        if (isRefine && editorContent) await doRefine(message);
        else if (isReview && !editorContent) await doStreamReview(message);
        else await doChat(message);
      } catch (err: any) {
        addMsg({ role: "assistant", content: `Error: ${err.message}` });
      } finally {
        setLoading(false);
      }
    },
    [editorContent, activeId],
  );

  // --- File upload ---

  async function handleFileUpload(file: File, userMessage: string) {
    let outlineText = "";

    if (file.name.endsWith(".md") || file.name.endsWith(".txt")) {
      outlineText = await file.text();
    } else if (file.name.endsWith(".docx")) {
      const result = await uploadOutline(file);
      if (result.error) { addMsg({ role: "assistant", content: `Error: ${result.error}` }); return; }
      outlineText = result.text;
    }

    if (!outlineText) return;

    const lines = outlineText.split("\n").filter((l) => l.trim());
    const sections: PlanSection[] = [];
    const numPattern = /^(\d+(?:\.\d+)*)\s+(.+)/;

    for (const line of lines) {
      const m = line.trim().match(numPattern);
      if (m) {
        sections.push({ title: m[2].trim(), number: m[1], level: m[1].split(".").length, enabled: true });
      }
    }

    const leafSections = sections.filter((s, i) => {
      const next = sections[i + 1];
      return !next || next.level <= s.level;
    });

    let constraints = "";
    const cm = userMessage.match(/(?:只|关注|focus|constraint|约束|品种|species)[：:]\s*(.+)/i)
      || userMessage.match(/(?:只关注|只写|focus on)\s+(.+)/i);
    if (cm) constraints = cm[1].trim();

    const language = /中文|zh|chinese/i.test(userMessage) ? "zh" : "en";
    setPendingOutline({ sections: leafSections, text: outlineText, language, constraints });

    addMsg({
      role: "assistant",
      content: `Parsed **${file.name}**: ${leafSections.length} sections found.\n\n${leafSections.slice(0, 8).map((s) => `- ${s.number} ${s.title}`).join("\n")}${leafSections.length > 8 ? `\n- ... and ${leafSections.length - 8} more` : ""}`,
    });

    addMsg({
      role: "assistant",
      content: "",
      type: "actions",
      data: {
        text: "What would you like to do?",
        actions: [
          { label: "Select sections & generate", value: "__show_plan__" },
          { label: "Generate all sections", value: "__generate_all__" },
        ],
      },
    });
  }

  function showPlanFromOutline() {
    if (!pendingOutline) return;
    const planState: PlanState = {
      sections: pendingOutline.sections,
      constraints: pendingOutline.constraints,
      language: pendingOutline.language,
      maxPapers: 10,
      outlineText: pendingOutline.text,
    };
    setPlan(planState);
    addMsg({ role: "assistant", content: "", type: "plan", data: planState });
  }

  async function generateAllFromOutline() {
    if (!pendingOutline) return;
    setPlan({
      sections: pendingOutline.sections,
      constraints: pendingOutline.constraints,
      language: pendingOutline.language,
      maxPapers: 10,
      outlineText: pendingOutline.text,
    });

    setLoading(true);
    addMsg({ role: "user", content: `Generate all ${pendingOutline.sections.length} sections` });

    try {
      const res = await startOutlineReview(pendingOutline.text, pendingOutline.language, 10, pendingOutline.constraints);
      await readSSE(res, handleOutlineEvent);
    } catch (err: any) {
      addMsg({ role: "assistant", content: `Error: ${err.message}` });
    } finally {
      setLoading(false);
    }
  }

  // --- Review / Refine / Chat ---

  async function doStreamReview(question: string) {
    const PIPELINE: PipelineStep[] = [
      { name: "plan", label: "Planning research", status: "pending" },
      { name: "search", label: "Searching papers", status: "pending" },
      { name: "read", label: "Analyzing papers", status: "pending" },
      { name: "contradictions", label: "Detecting contradictions", status: "pending" },
      { name: "synthesize", label: "Writing review", status: "pending" },
      { name: "ground", label: "Verifying citations", status: "pending" },
      { name: "review", label: "Evaluating quality", status: "pending" },
    ];

    const stepsId = uid();
    setMessages((prev) => [
      ...prev,
      { id: stepsId, role: "assistant", content: "", type: "steps", data: { steps: [...PIPELINE] }, timestamp: Date.now() },
    ]);

    const updateStep = (name: string, status: PipelineStep["status"], summary?: string, details?: any) => {
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== stepsId) return m;
          const steps = (m.data?.steps || []).map((s: PipelineStep) => {
            if (s.name === name) return { ...s, status, summary, details: details ?? s.details };
            if (status === "active" && s.status === "active") return { ...s, status: "done" as const };
            return s;
          });
          return { ...m, data: { steps }, timestamp: Date.now() };
        }),
      );
    };

    // Update conversation title with the question
    updateConversationTitle(question);

    const isChinese = /[一-鿿]/.test(question);
    const language = isChinese ? "zh" : "en";
    const res = await startReview(question, 15, language, "");
    await readSSE(res, (event, data) => {
      if (event === "status") {
        updateStep(data.step, "active");
      } else if (event === "plan") {
        updateStep("plan", "done", `${data.domain} — ${data.sub_topics?.length} topics`);
      } else if (event === "search") {
        updateStep("search", "done", `Found ${data.papers_found} papers`, { papers: data.papers });
      } else if (event === "read") {
        updateStep("read", "done", `${data.analyzed} papers analyzed`);
      } else if (event === "contradictions") {
        updateStep("contradictions", "done", data.count > 0 ? `${data.count} contradictions found` : "No contradictions");
      } else if (event === "synthesis") {
        updateStep("synthesize", "done", `${data.word_count} words, ${data.themes?.length} themes`);
      } else if (event === "grounding") {
        updateStep("ground", "done", `${data.verified}/${data.total} verified (${data.accuracy}%)`);
      } else if (event === "review") {
        updateStep("review", "done", `Score: ${data.score?.toFixed(2)}`);
      } else if (event === "complete") {
        let reviewText = data.text || "";
        if (!reviewText.trimStart().startsWith("# ")) {
          reviewText = `# ${question.trim()}\n\n${reviewText}`;
        }
        setEditorContent(reviewText);

        addMsg({
          role: "assistant",
          content: `Review complete: **${data.papers} papers**, **${data.word_count} words**, score **${data.score?.toFixed(2)}** (${data.time}s)`,
        });

        addMsg({
          role: "assistant",
          content: "",
          type: "actions",
          data: {
            text: "What would you like to do next?",
            actions: [
              { label: "Refine review", value: "Refine this review to improve clarity and flow" },
              { label: "Export Markdown", value: "__export_md__" },
              { label: "Export BibTeX", value: "__export_bib__" },
            ],
          },
        });

        setConversations((prev) => {
          const updated = prev.map((c) =>
            c.id === activeId
              ? { ...c, title: question.slice(0, 60), reviewMeta: { papers: data.papers, words: data.word_count, score: data.score } }
              : c,
          );
          saveConversations(updated);
          return updated;
        });
      }
    });
  }

  function updateConversationTitle(title: string) {
    setConversations((prev) => {
      const updated = prev.map((c) =>
        c.id === activeId ? { ...c, title: title.slice(0, 60) } : c,
      );
      saveConversations(updated);
      return updated;
    });
  }

  async function doRefine(message: string) {
    const instruction = message.replace(/\[Selected:.*?\]\s*/s, "").trim();
    const data = await refineReview(instruction || message);
    if (data.text) {
      setPreviousContent(editorContent);
      setEditorContent(data.text);
      addMsg({
        role: "assistant",
        content: `Refined: ${data.word_count} words (${data.stats?.added || 0} added, ${data.stats?.removed || 0} removed). Review changes in the editor.`,
      });
    }
  }

  async function doChat(message: string) {
    const data = await sendChat(message);
    addMsg({ role: "assistant", content: data.response || data.error || "No response" });
  }

  // --- Export ---

  const handleExport = async (format: string) => {
    try {
      const data = await exportReview(format);
      const blob = new Blob([data.content], { type: "text/plain" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `review.${format === "bibtex" ? "bib" : "md"}`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (err: any) {
      addMsg({ role: "assistant", content: `Export failed: ${err.message}` });
    }
  };

  return (
    <div className="app">
      {showSetup && (
        <SetupWizard
          onComplete={() => setShowSetup(false)}
          onClose={() => setShowSetup(false)}
        />
      )}

      <Sidebar
        conversations={conversations}
        activeId={activeId}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        onNewReview={handleNewReview}
        onSelect={handleSelectConversation}
        onRename={(id, title) => {
          setConversations((prev) => {
            const updated = prev.map((c) => c.id === id ? { ...c, title } : c);
            saveConversations(updated);
            return updated;
          });
        }}
        onOpenSettings={() => setShowSetup(true)}
      />

      <main className="app-main">
        <Editor
          content={editorContent}
          previousContent={previousContent}
          onSelectionSend={setSelectionContext}
          onExport={handleExport}
          onAcceptChanges={() => setPreviousContent("")}
          onRevertChanges={() => {
            setEditorContent(previousContent);
            setPreviousContent("");
          }}
        />
        <Chat
          messages={messages}
          onSend={handleSend}
          onPlanUpdate={handlePlanUpdate}
          onPlanExecute={handlePlanExecute}
          selectionContext={selectionContext}
          onClearSelection={() => setSelectionContext("")}
          loading={loading}
        />
      </main>
    </div>
  );
}
