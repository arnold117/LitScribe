export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  type?: "text" | "plan" | "progress" | "coverage" | "actions" | "grounding" | "steps" | "analysis";
  data?: any;
  attachment?: { name: string; type: string };
}

export interface PlanSection {
  title: string;
  number: string;
  level: number;
  enabled: boolean;
}

export interface PlanState {
  sections: PlanSection[];
  constraints: string;
  language: string;
  maxPapers: number;
  outlineText: string;
}

export interface Session {
  session_id: string;
  question: string;
  domain: string;
  papers: number;
  papers_count: number;
  words: number;
  word_count: number;
  score: number;
  created_at: string;
}

export interface ContentVersion {
  content: string;
  appendix: string;
  timestamp: number;
  label: string;
}

export type CitationFormat = "bracket" | "apa" | "vancouver";

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  messages: ChatMessage[];
  editorContent: string;
  previousContent: string;
  appendixContent: string;
  versions: ContentVersion[];
  plan: PlanState | null;
  pendingOutline: {
    sections: PlanSection[];
    text: string;
    language: string;
    constraints: string;
  } | null;
  backendSessionId?: string;
  reviewMeta?: { papers: number; words: number; score: number };
}

export interface ReferenceEntry {
  key: string;
  authors: string;
  year: string;
  title: string;
  venue: string;
  doi: string;
}

export interface HealthStatus {
  configured: boolean;
  llm_key_set: boolean;
  llm_base_set: boolean;
  llm_model_set: boolean;
}

export interface PipelineStep {
  name: string;
  label: string;
  status: "pending" | "active" | "done";
  summary?: string;
  details?: any;
}

export interface SearchPaper {
  title: string;
  authors: string[];
  year: number;
  url: string;
}
