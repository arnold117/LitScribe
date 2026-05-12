import { useState, useCallback } from "react";
import Editor from "./components/Editor";
import Chat from "./components/Chat";
import type { ChatMessage, PlanSection, PlanState } from "./types";
import "./App.css";

const API = "";

function App() {
  const [editorContent, setEditorContent] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [selectionContext, setSelectionContext] = useState("");
  const [loading, setLoading] = useState(false);
  const [plan, setPlan] = useState<PlanState | null>(null);

  const addMsg = (msg: Omit<ChatMessage, "timestamp">) =>
    setMessages((prev) => [...prev, { ...msg, timestamp: Date.now() }]);

  const updateLastSystem = (content: string, data?: any) =>
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role === "system" || (last?.role === "assistant" && last?.type === "progress")) {
        return [...prev.slice(0, -1), { ...last, content, data, timestamp: Date.now() }];
      }
      return [...prev, { role: "assistant" as const, content, type: "progress" as const, data, timestamp: Date.now() }];
    });

  const handleSelectionSend = useCallback((text: string) => {
    setSelectionContext(text);
  }, []);

  const handlePlanUpdate = useCallback((sections: PlanSection[], constraints: string) => {
    setPlan((prev) => prev ? { ...prev, sections, constraints } : null);
  }, []);

  const handlePlanExecute = useCallback(async () => {
    if (!plan) return;
    const enabled = plan.sections.filter((s) => s.enabled);
    if (!enabled.length) return;

    setLoading(true);
    addMsg({
      role: "user",
      content: `Generate ${enabled.length} sections${plan.constraints ? ` with constraints: ${plan.constraints.slice(0, 100)}...` : ""}`,
    });

    try {
      const sectionFilter = enabled.map((s) => s.number).join(",");
      const body = {
        outline_text: plan.outlinePath,
        language: plan.language,
        max_papers_per_section: plan.maxPapers,
        constraints: plan.constraints,
        section_filter: sectionFilter,
      };

      const res = await fetch(`${API}/api/outline-review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop()!;

        let event: string | null = null;
        for (const line of lines) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ") && event) {
            const data = JSON.parse(line.slice(6));
            handleOutlineEvent(event, data);
            event = null;
          }
        }
      }
    } catch (err: any) {
      addMsg({ role: "assistant", content: `Error: ${err.message}` });
    } finally {
      setLoading(false);
    }
  }, [plan]);

  function handleOutlineEvent(event: string, data: any) {
    if (event === "section_start") {
      updateLastSystem(
        `[${data.index + 1}/${data.total}] ${data.title}`,
        { current: data.index + 1, total: data.total, title: data.title },
      );
    } else if (event === "section_done") {
      updateLastSystem(
        `[${data.index + 1}/${data.total}] ${data.title}`,
        { current: data.index + 1, total: data.total, title: data.title, papers: data.papers, words: data.words },
      );
    } else if (event === "complete") {
      setEditorContent(data.text);
      setMessages((prev) => [
        ...prev.filter((m) => m.type !== "progress"),
        {
          role: "assistant",
          content: `Done: ${data.total_words} words, ${data.total_papers} papers, ${data.total_sections} sections (${data.time}s)`,
          timestamp: Date.now(),
        },
      ]);

      if (data.coverage) {
        addMsg({
          role: "assistant",
          content: "",
          type: "coverage",
          data: data.coverage,
        });
      }

      addMsg({
        role: "assistant",
        content: "",
        type: "actions",
        data: {
          text: "What's next?",
          actions: [
            { label: "Check consistency", value: "Check cross-section consistency" },
            { label: "Export MD", value: "Export as markdown" },
            { label: "Show coverage", value: "Show species coverage report" },
          ],
        },
      });
    }
  }

  const handleSend = useCallback(
    async (message: string, attachment?: File) => {
      const userMsg: ChatMessage = {
        role: "user",
        content: message,
        timestamp: Date.now(),
        attachment: attachment ? { name: attachment.name, content: "" } : undefined,
      };
      setMessages((prev) => [...prev, userMsg]);

      // If file attached, parse outline and show plan
      if (attachment) {
        setLoading(true);
        try {
          await parseAndShowPlan(attachment, message);
        } catch (err: any) {
          addMsg({ role: "assistant", content: `Error parsing file: ${err.message}` });
        } finally {
          setLoading(false);
        }
        return;
      }

      // Intent detection
      const isReview = /review|综述|generate|生成/i.test(message) && !message.startsWith("[Selected:");
      const isRefine = message.startsWith("[Selected:") || /改|修改|refine|rewrite|expand|缩减|展开/i.test(message);
      const isCoverage = /coverage|覆盖|品种/i.test(message);
      const isConsistency = /consistency|一致|矛盾|冲突/i.test(message);

      setLoading(true);
      try {
        if (isRefine && editorContent) {
          await refineReview(message);
        } else if (isReview && !editorContent) {
          await streamReview(message);
        } else {
          await chatMessage(message);
        }
      } catch (err: any) {
        addMsg({ role: "assistant", content: `Error: ${err.message}` });
      } finally {
        setLoading(false);
      }
    },
    [editorContent],
  );

  async function parseAndShowPlan(file: File, userMessage: string) {
    let outlineText = "";

    if (file.name.endsWith(".md") || file.name.endsWith(".txt")) {
      outlineText = await file.text();
    } else if (file.name.endsWith(".docx")) {
      addMsg({
        role: "assistant",
        content: "For .docx files, please paste the outline text. I'll parse it into a plan.",
        type: "actions",
        data: {
          text: "Paste your outline text below, or switch to .md/.txt format.",
          actions: [],
        },
      });
      return;
    }

    // Parse outline via simple line analysis
    const lines = outlineText.split("\n").filter((l) => l.trim());
    const sections: PlanSection[] = [];
    const numPattern = /^(\d+(?:\.\d+)*)\s+(.+)/;

    for (const line of lines) {
      const m = line.trim().match(numPattern);
      if (m) {
        const num = m[1];
        const title = m[2].trim();
        const level = num.split(".").length;
        // Only leaf sections (deepest level)
        sections.push({ title, number: num, level, enabled: true });
      }
    }

    // Keep only leaf nodes
    const leafSections = sections.filter((s, i) => {
      const next = sections[i + 1];
      return !next || next.level <= s.level;
    });

    // Extract constraints from user message
    let constraints = "";
    const constraintMatch = userMessage.match(/(?:只|关注|focus|constraint|约束|品种|species)[：:]\s*(.+)/i)
      || userMessage.match(/(?:只关注|只写|focus on)\s+(.+)/i);
    if (constraintMatch) {
      constraints = constraintMatch[1].trim();
    }

    const planState: PlanState = {
      sections: leafSections,
      constraints,
      language: /中文|zh|chinese/i.test(userMessage) ? "zh" : "en",
      maxPapers: 10,
      outlinePath: outlineText,
    };
    setPlan(planState);

    addMsg({
      role: "assistant",
      content: "",
      type: "plan",
      data: planState,
    });
  }

  async function streamReview(question: string) {
    const res = await fetch(`${API}/api/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        max_papers: 15,
        language: "en",
        instructions: "",
      }),
    });

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop()!;

      let event: string | null = null;
      for (const line of lines) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ") && event) {
          const data = JSON.parse(line.slice(6));
          if (event === "status") {
            updateLastSystem(data.message);
          } else if (event === "complete") {
            setEditorContent(data.text);
            setMessages((prev) => [
              ...prev.filter((m) => m.role !== "system"),
              {
                role: "assistant",
                content: `Review complete: ${data.papers} papers, ${data.word_count} words, score ${data.score?.toFixed(2)}`,
                timestamp: Date.now(),
              },
            ]);
          }
          event = null;
        }
      }
    }
  }

  async function refineReview(message: string) {
    const instruction = message.replace(/\[Selected:.*?\]\s*/s, "").trim();
    const res = await fetch(`${API}/api/refine`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction: instruction || message }),
    });
    const data = await res.json();
    if (data.text) {
      setEditorContent(data.text);
      addMsg({
        role: "assistant",
        content: `Refined: ${data.word_count} words`,
      });
    }
  }

  async function chatMessage(message: string) {
    const res = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    addMsg({
      role: "assistant",
      content: data.response || data.error || "No response",
    });
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>LitScribe</h1>
        <span className="app-subtitle">AI-powered academic writing</span>
      </header>
      <main className="app-main">
        <Editor content={editorContent} onSelectionSend={handleSelectionSend} />
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

export default App;
