import { useEffect } from "react";
import { BookOpen, MessageSquare, ListChecks, PencilLine, X } from "lucide-react";

const EXAMPLES = [
  "综述大语言模型在医学影像诊断中的应用",
  "Review CRISPR knockout strategies in CHO cells",
  "Review transformer architectures for time series forecasting",
];

const STEPS = [
  {
    icon: <MessageSquare size={18} />,
    title: "提问或上传大纲",
    desc: "在右侧 Chat 输入研究问题，或拖入一个大纲文件（.docx / .md / .txt）。",
  },
  {
    icon: <ListChecks size={18} />,
    title: "确认综述骨架",
    desc: "系统先规划章节大纲，你可勾选 / 调整 / 加约束，确认后再逐节生成。",
  },
  {
    icon: <PencilLine size={18} />,
    title: "生成后对话式修改",
    desc: "综述生成在编辑器里；用自然语言继续 refine，每次改动自动存版本。",
  },
];

interface OnboardingProps {
  onClose: () => void;
  onPickExample: (q: string) => void;
}

export default function Onboarding({ onClose, onPickExample }: OnboardingProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="setup-overlay" onClick={onClose}>
      <div className="onboard-modal" onClick={(e) => e.stopPropagation()}>
        <button className="setup-close" onClick={onClose} title="跳过">
          <X size={16} />
        </button>

        <div className="onboard-brand">
          <BookOpen size={22} className="brand-icon" />
          <span className="brand-text">LitScribe</span>
        </div>
        <h2 className="onboard-title">欢迎使用 LitScribe</h2>
        <p className="onboard-sub">一个以对话驱动的 AI 文献综述工作台。三步上手：</p>

        <div className="onboard-steps">
          {STEPS.map((s, i) => (
            <div className="onboard-step" key={i}>
              <div className="onboard-step-num">{i + 1}</div>
              <div className="onboard-step-icon">{s.icon}</div>
              <div className="onboard-step-body">
                <div className="onboard-step-title">{s.title}</div>
                <div className="onboard-step-desc">{s.desc}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="onboard-examples-label">试试一个示例：</div>
        <div className="onboard-examples">
          {EXAMPLES.map((q) => (
            <button key={q} className="onboard-example" onClick={() => onPickExample(q)}>
              {q}
            </button>
          ))}
        </div>

        <button className="onboard-start" onClick={onClose}>
          我自己来 — 开始使用
        </button>
      </div>
    </div>
  );
}
