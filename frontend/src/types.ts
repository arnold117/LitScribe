export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  type?: "text" | "plan" | "progress" | "coverage" | "actions";
  data?: any;
  attachment?: { name: string; content: string };
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
  outlinePath: string;
}
