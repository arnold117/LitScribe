import { useMemo } from "react";
import { diffWords } from "diff";
import { Check, Undo2 } from "lucide-react";

interface DiffViewProps {
  oldText: string;
  newText: string;
  onAccept: () => void;
  onRevert: () => void;
}

export default function DiffView({ oldText, newText, onAccept, onRevert }: DiffViewProps) {
  const changes = useMemo(() => diffWords(oldText, newText), [oldText, newText]);

  const stats = useMemo(() => {
    let added = 0;
    let removed = 0;
    for (const c of changes) {
      if (c.added) added += c.value.split(/\s+/).filter(Boolean).length;
      if (c.removed) removed += c.value.split(/\s+/).filter(Boolean).length;
    }
    return { added, removed };
  }, [changes]);

  return (
    <div className="diff-container">
      <div className="diff-toolbar">
        <div className="diff-stats">
          <span className="diff-stat-added">+{stats.added} words</span>
          <span className="diff-stat-removed">-{stats.removed} words</span>
        </div>
        <div className="diff-actions">
          <button className="diff-btn diff-btn-revert" onClick={onRevert}>
            <Undo2 size={13} />
            Revert
          </button>
          <button className="diff-btn diff-btn-accept" onClick={onAccept}>
            <Check size={13} />
            Accept changes
          </button>
        </div>
      </div>
      <div className="diff-content">
        {changes.map((change, i) => {
          if (change.added) {
            return (
              <span key={i} className="diff-added">
                {change.value}
              </span>
            );
          }
          if (change.removed) {
            return (
              <span key={i} className="diff-removed">
                {change.value}
              </span>
            );
          }
          return <span key={i}>{change.value}</span>;
        })}
      </div>
    </div>
  );
}
