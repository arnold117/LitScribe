import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import Sidebar from "./components/Sidebar";
import type { SidebarFile } from "./components/Sidebar";
import Editor from "./components/Editor";
import Chat from "./components/Chat";
import SetupWizard from "./components/SetupWizard";
import Onboarding from "./components/Onboarding";
import TemplatesModal from "./components/TemplatesModal";
import type { ChatMessage, PlanSection, PlanState, Conversation, ReferenceEntry, ContentVersion, CitationFormat } from "./types";
import {
  checkHealth,
  planReview,
  refineReview,
  sendChat,
  uploadOutline,
  startOutlineReview,
  exportReview,
  fetchWritingAnalysis,
  applyTemplate,
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

const APPENDIX_HEADINGS = [
  "Methodology Comparison",
  "Research Timeline",
  "Statistical Summary",
  "Suggested Figures",
  "Comparative Analysis",
];

function splitReviewContent(text: string): { core: string; appendix: string } {
  const lines = text.split("\n");
  let splitIdx = lines.length;
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^##\s+(.+)/);
    if (m && APPENDIX_HEADINGS.some((h) => m[1].trim().startsWith(h))) {
      splitIdx = i;
      break;
    }
  }
  const core = lines.slice(0, splitIdx).join("\n").trimEnd();
  const appendix = lines.slice(splitIdx).join("\n").trim();
  return { core, appendix };
}

function parseReferences(text: string): ReferenceEntry[] {
  const entries: ReferenceEntry[] = [];
  const refSection = text.match(/## References\n([\s\S]*?)(?=\n##\s|\n*$)/);
  if (!refSection) return entries;
  const refLines = refSection[1].split("\n").filter((l) => l.trim());
  for (const line of refLines) {
    // Tolerate both canonical "[key]: Authors (Year). Title" and list-style
    // "- [key] Authors (Year). Title" (optional leading dash, optional colon).
    const m = line.match(/^\s*-?\s*\[([^\]]+)\]:?\s*(.+?)\s*\((\d{4})\)\.\s*(.+)/);
    if (m) {
      const rest = m[4];
      const venueMatch = rest.match(/\*(.+?)\*\./);
      const doiMatch = rest.match(/doi:(\S+)/);
      const title = rest.replace(/\*.*?\*\.?/, "").replace(/doi:\S+/, "").replace(/\.\s*$/, "").trim();
      entries.push({
        key: m[1],
        authors: m[2].trim(),
        year: m[3],
        title,
        venue: venueMatch?.[1] || "",
        doi: doiMatch?.[1] || "",
      });
    }
  }
  return entries;
}

const CITE_PATTERN = /\[([A-Z一-鿿][^\[\]]{2,40}?,\s*\d{4}[a-z]?)\]/g;

function convertCitations(text: string, format: CitationFormat, refs: ReferenceEntry[]): string {
  if (format === "bracket") return text;

  if (format === "apa") {
    return text.replace(CITE_PATTERN, "($1)");
  }

  if (format === "vancouver") {
    const seen = new Map<string, number>();
    let counter = 0;

    const refMap = new Map<string, ReferenceEntry>();
    for (const r of refs) {
      const surname = r.authors.split(",")[0].trim();
      refMap.set(`${surname}, ${r.year}`, r);
      refMap.set(`${surname} et al., ${r.year}`, r);
    }

    const body = text.replace(/\n## References\n[\s\S]*$/, "");
    const numbered = body.replace(CITE_PATTERN, (_match, key: string) => {
      if (!seen.has(key)) {
        counter++;
        seen.set(key, counter);
      }
      return `[${seen.get(key)}]`;
    });

    const refLines = ["\n\n## References\n"];
    for (const [citeKey, num] of [...seen.entries()].sort((a, b) => a[1] - b[1])) {
      const ref = refMap.get(citeKey);
      if (ref) {
        const venue = ref.venue ? ` *${ref.venue}*.` : "";
        const doi = ref.doi ? ` doi:${ref.doi}` : "";
        refLines.push(`[${num}] ${ref.authors} (${ref.year}). ${ref.title}.${venue}${doi}`);
      } else {
        refLines.push(`[${num}] ${citeKey}`);
      }
    }

    return numbered + refLines.join("\n");
  }

  return text;
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(null);

  const [editorContent, setEditorContent] = useState("");
  const [previousContent, setPreviousContent] = useState("");
  const [appendixContent, setAppendixContent] = useState("");
  const [versions, setVersions] = useState<ContentVersion[]>([]);
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
  const [showOnboard, setShowOnboard] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);
  const [seedInput, setSeedInput] = useState("");

  // Ref to always have latest state in async callbacks
  const stateRef = useRef({ messages, editorContent, previousContent, appendixContent, versions, plan, pendingOutline });
  stateRef.current = { messages, editorContent, previousContent, appendixContent, versions, plan, pendingOutline };

  const abortRef = useRef<AbortController | null>(null);

  const newAbortSignal = () => {
    abortRef.current = new AbortController();
    return abortRef.current.signal;
  };

  const handleStop = () => abortRef.current?.abort();

  const isAbort = (err: any) => err?.name === "AbortError";

  // Last failed operation, retryable via the "__retry__" action without
  // re-adding the user's message bubble.
  const retryRef = useRef<(() => Promise<void>) | null>(null);

  useEffect(() => {
    checkHealth()
      .then((h) => {
        if (!h.configured) {
          setShowSetup(true);
        } else if (conversations.length === 0 && !localStorage.getItem("litscribe-onboarded")) {
          setShowOnboard(true);
        }
      })
      .catch(() => {});
  }, []);

  const dismissOnboard = () => {
    localStorage.setItem("litscribe-onboarded", "1");
    setShowOnboard(false);
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key === "s") {
        e.preventDefault();
        if (editorContent) handleExport("markdown");
      }
      if (mod && e.key === "l") {
        e.preventDefault();
        setSidebarCollapsed((v) => !v);
      }
      if (e.key === "Escape") {
        if (showSetup) setShowSetup(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [editorContent, showSetup]);

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
        appendixContent: s.appendixContent,
        versions: s.versions,
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
      appendixContent: "",
      versions: [],
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
    setAppendixContent(target.appendixContent || "");
    setVersions(target.versions || []);
    setPlan(target.plan);
    setPendingOutline(target.pendingOutline);
    setSelectionContext("");
  };

  const handleDeleteConversation = (id: string) => {
    setConversations((prev) => {
      const updated = prev.filter((c) => c.id !== id);
      saveConversations(updated);
      return updated;
    });
    if (id === activeId) {
      setActiveId(null);
      setEditorContent("");
      setPreviousContent("");
      setAppendixContent("");
      setVersions([]);
      setMessages([]);
      setPlan(null);
      setPendingOutline(null);
      setSelectionContext("");
    }
  };

  const handleNewReview = () => {
    persistConversation();
    setActiveId(null);
    setEditorContent("");
    setPreviousContent("");
    setAppendixContent("");
    setVersions([]);
    setMessages([]);
    setPlan(null);
    setPendingOutline(null);
    setSelectionContext("");
  };

  // --- Helpers ---

  const addMsg = (msg: Omit<ChatMessage, "id" | "timestamp">) =>
    setMessages((prev) => [...prev, { ...msg, id: uid(), timestamp: Date.now() }]);

  const reportFailure = (message: string) => {
    setMessages((prev) => [
      ...prev.filter((m) => m.type !== "progress"),
      { id: uid(), role: "assistant" as const, content: `Error: ${message}`, timestamp: Date.now() },
      {
        id: uid(),
        role: "assistant" as const,
        content: "",
        type: "actions" as const,
        data: {
          text: "请求失败（网络或服务端错误）。",
          actions: [{ label: "↻ Retry", value: "__retry__" }],
        },
        timestamp: Date.now(),
      },
    ]);
  };

  // Run a generation op with unified abort/failure handling; remembers it for retry.
  const runOp = async (op: () => Promise<void>) => {
    retryRef.current = op;
    setLoading(true);
    try {
      await op();
    } catch (err: any) {
      if (isAbort(err)) {
        setMessages((prev) => [
          ...prev.filter((m) => m.type !== "progress"),
          { id: uid(), role: "assistant", content: "⏹ Generation stopped.", timestamp: Date.now() },
        ]);
      } else {
        reportFailure(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const updateLastProgress = (content: string, data?: any) =>
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.type === "progress") {
        return [
          ...prev.slice(0, -1),
          { ...last, content, data: { ...last.data, ...data }, timestamp: Date.now() },
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

    addMsg({
      role: "user",
      content: `Generate ${enabled.length} sections${plan.constraints ? ` (constraints: ${plan.constraints.slice(0, 80)}...)` : ""}`,
    });

    await runOp(async () => {
      const sectionFilter = enabled.map((s) => s.number).join(",");
      const res = await startOutlineReview(
        plan.outlineText,
        plan.language,
        plan.maxPapers,
        plan.constraints,
        sectionFilter,
        newAbortSignal(),
      );
      await readSSE(res, handleOutlineEvent);
    });
  }, [plan]);

  function handleOutlineEvent(event: string, data: any) {
    if (event === "error") {
      reportFailure(data.message);
    } else if (event === "outline_parsed") {
      const total = data.selected_sections ?? data.total_sections;
      updateLastProgress(`Preparing ${total} sections`, {
        current: 0, total, title: "Preparing", stage: "Parsing outline…",
      });
    } else if (event === "section_start") {
      updateLastProgress(`[${data.index + 1}/${data.total}] ${data.title}`, {
        current: data.index + 1, total: data.total, title: data.title,
        papers: undefined, words: undefined, stage: "Starting…",
      });
    } else if (event === "section_search") {
      updateLastProgress(`[${data.index + 1}] ${data.title}`, { stage: "Searching papers…" });
    } else if (event === "section_read") {
      updateLastProgress(`Reading papers`, { stage: `Reading ${data.papers} papers…` });
    } else if (event === "section_synthesize") {
      updateLastProgress(`[${data.index + 1}] ${data.title}`, { stage: "Writing section…" });
    } else if (event === "section_done") {
      updateLastProgress(`[${data.index + 1}/${data.total}] ${data.title}`, {
        current: data.index + 1, total: data.total, title: data.title,
        papers: data.papers, words: data.words, stage: "",
      });
    } else if (event === "assembling") {
      updateLastProgress("Assembling document", { stage: "Assembling & consistency pass…" });
    } else if (event === "complete") {
      const { core, appendix } = splitReviewContent(data.text);
      setEditorContent(core);
      setAppendixContent(appendix);
      setVersions((prev) => [...prev, {
        content: core, appendix, timestamp: Date.now(),
        label: `v${prev.length + 1} — Outline review`,
      }]);
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
            { label: "写作分析", value: "__writing_analysis__" },
            { label: "写作模板", value: "__templates__" },
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
      if (message === "__retry__") {
        if (retryRef.current) await runOp(retryRef.current);
        return;
      }
      if (message === "__templates__") { setShowTemplates(true); return; }
      if (message === "__writing_analysis__") {
        setLoading(true);
        try {
          const data = await fetchWritingAnalysis();
          addMsg({ role: "assistant", content: "", type: "analysis", data });
        } catch (err: any) {
          reportFailure(err.message);
        } finally {
          setLoading(false);
        }
        return;
      }

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
        await runOp(() => handleFileUpload(attachment, message));
        return;
      }

      const isReview = /review|综述|generate|生成/i.test(message) && !message.startsWith("[Selected:");
      const isRefine = message.startsWith("[Selected:") || /改|修改|refine|rewrite|expand|缩减|展开/i.test(message);

      await runOp(async () => {
        if (isRefine && editorContent) await doRefine(message);
        else if (isReview && !editorContent) await doPlanFirst(message);
        else await doChat(message);
      });
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

    addMsg({ role: "user", content: `Generate all ${pendingOutline.sections.length} sections` });

    await runOp(async () => {
      const res = await startOutlineReview(pendingOutline.text, pendingOutline.language, 10, pendingOutline.constraints, "", newAbortSignal());
      await readSSE(res, handleOutlineEvent);
    });
  }

  // --- Review / Refine / Chat ---

  // Skeleton-first flow: plan a section outline, let the user confirm/edit it
  // in the plan card, then generate via the outline-review pipeline.
  async function doPlanFirst(question: string) {
    updateConversationTitle(question);

    const isChinese = /[一-鿿]/.test(question);
    const language = isChinese ? "zh" : "en";
    const data = await planReview(question, language, newAbortSignal());

    const planState: PlanState = {
      sections: data.sections as PlanSection[],
      constraints: "",
      language,
      maxPapers: 10,
      outlineText: data.outline_text,
    };
    setPlan(planState);

    addMsg({
      role: "assistant",
      content: isChinese
        ? `已规划综述骨架（领域：${data.domain || "通用"}，${data.sections.length} 节）。确认或调整章节后点击 **Generate** 开始逐节生成。`
        : `Proposed skeleton ready (domain: ${data.domain || "general"}, ${data.sections.length} sections). Toggle or adjust sections below, then hit **Generate**.`,
    });
    addMsg({ role: "assistant", content: "", type: "plan", data: planState });
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
    // Recent text turns give the model context for follow-up edits ("再正式一点", "改回去")
    const history = stateRef.current.messages
      .filter((m) => (m.type === "text" || !m.type) && m.content && !m.content.startsWith("Error:"))
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content.slice(0, 1000) }));
    const data = await refineReview(instruction || message, history, newAbortSignal());
    if (data.text) {
      const { core, appendix: newAppendix } = splitReviewContent(data.text);
      setPreviousContent(editorContent);
      setEditorContent(core);
      if (newAppendix) setAppendixContent(newAppendix);
      setVersions((prev) => [...prev, {
        content: core, appendix: newAppendix || appendixContent, timestamp: Date.now(),
        label: `v${prev.length + 1} — Refined`,
      }]);
      addMsg({
        role: "assistant",
        content: `Refined: ${data.word_count} words (${data.stats?.added || 0} added, ${data.stats?.removed || 0} removed). Review changes in the editor.`,
      });
    } else if (data.error) {
      addMsg({ role: "assistant", content: `Error: ${data.error}` });
    }
  }

  async function doChat(message: string) {
    const data = await sendChat(message, newAbortSignal());
    addMsg({ role: "assistant", content: data.response || data.error || "No response" });
  }

  // --- Export ---

  const handleExport = async (format: string, includeAppendix = true, citeFormat: CitationFormat = "bracket") => {
    if (format === "bibtex") {
      try {
        const data = await exportReview("bibtex");
        const blob = new Blob([data.content], { type: "text/plain" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "review.bib";
        a.click();
        URL.revokeObjectURL(a.href);
      } catch (err: any) {
        addMsg({ role: "assistant", content: `Export failed: ${err.message}` });
      }
      return;
    }
    try {
      let content = editorContent;
      if (includeAppendix && appendixContent) {
        content = content.trimEnd() + "\n\n" + appendixContent;
      }
      const refs = parseReferences(editorContent);
      content = convertCitations(content, citeFormat, refs);
      const blob = new Blob([content], { type: "text/plain" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "review.md";
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (err: any) {
      addMsg({ role: "assistant", content: `Export failed: ${err.message}` });
    }
  };

  const references = useMemo(() => parseReferences(editorContent), [editorContent]);

  const sidebarFiles = useMemo<SidebarFile[]>(() => {
    const list: SidebarFile[] = [];
    for (const m of messages) {
      if (m.attachment) {
        list.push({
          name: m.attachment.name,
          kind: "upload",
          detail: m.attachment.type || "uploaded file",
        });
      }
    }
    if (editorContent) {
      list.push({
        name: "review.md",
        kind: "generated",
        detail: `${(editorContent.length / 1024).toFixed(1)} KB · Markdown`,
        onDownload: () => handleExport("markdown"),
      });
    }
    if (appendixContent) {
      list.push({
        name: "appendix.md",
        kind: "generated",
        detail: `${(appendixContent.length / 1024).toFixed(1)} KB · supplementary`,
        onDownload: () => {
          const blob = new Blob([appendixContent], { type: "text/plain" });
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          a.download = "appendix.md";
          a.click();
          URL.revokeObjectURL(a.href);
        },
      });
    }
    if (references.length > 0) {
      list.push({
        name: "references.bib",
        kind: "generated",
        detail: `${references.length} entries · BibTeX`,
        onDownload: () => handleExport("bibtex"),
      });
    }
    return list;
  }, [messages, editorContent, appendixContent, references]);

  const handleApplyTemplate = (id: string, instructions: string, wordCount: number) => {
    addMsg({ role: "user", content: `应用模板: ${id}` });
    runOp(async () => {
      const data = await applyTemplate(id, instructions, wordCount, newAbortSignal());
      addMsg({ role: "assistant", content: data.text || "(empty)" });
    });
  };

  const handleRestoreVersion = (v: ContentVersion) => {
    setPreviousContent(editorContent);
    setEditorContent(v.content);
    if (v.appendix) setAppendixContent(v.appendix);
  };

  return (
    <div className="app">
      {showSetup && (
        <SetupWizard
          onComplete={() => setShowSetup(false)}
          onClose={() => setShowSetup(false)}
        />
      )}

      {showOnboard && !showSetup && (
        <Onboarding
          onClose={dismissOnboard}
          onPickExample={(q) => { setSeedInput(q); dismissOnboard(); }}
        />
      )}

      {showTemplates && (
        <TemplatesModal
          hasReview={!!editorContent}
          onClose={() => setShowTemplates(false)}
          onApply={handleApplyTemplate}
        />
      )}

      <Sidebar
        conversations={conversations}
        activeId={activeId}
        collapsed={sidebarCollapsed}
        versions={versions}
        files={sidebarFiles}
        references={references}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        onNewReview={handleNewReview}
        onSelect={handleSelectConversation}
        onDelete={handleDeleteConversation}
        onRestoreVersion={handleRestoreVersion}
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
          onContentChange={setEditorContent}
          previousContent={previousContent}
          appendixContent={appendixContent}
          references={references}
          versions={versions}
          onSelectionSend={setSelectionContext}
          onExport={handleExport}
          onAcceptChanges={() => setPreviousContent("")}
          onRevertChanges={() => {
            setEditorContent(previousContent);
            setPreviousContent("");
          }}
          onRestoreVersion={handleRestoreVersion}
        />
        <Chat
          messages={messages}
          onSend={handleSend}
          onStop={handleStop}
          onPlanUpdate={handlePlanUpdate}
          onPlanExecute={handlePlanExecute}
          selectionContext={selectionContext}
          onClearSelection={() => setSelectionContext("")}
          seedInput={seedInput}
          onSeedConsumed={() => setSeedInput("")}
          loading={loading}
        />
      </main>
    </div>
  );
}
