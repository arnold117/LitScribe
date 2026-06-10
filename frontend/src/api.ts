import type { HealthStatus, Session } from "./types";

const BASE = "";

// Parse a response as JSON; surface non-JSON bodies (proxy errors, plain-text
// 500s) and {"error": ...} payloads as readable Error messages.
async function asJson(res: Response): Promise<any> {
  const text = await res.text();
  let data: any;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text.slice(0, 200)}` : ""}`);
  }
  if (!res.ok && data?.error) throw new Error(data.error);
  return data;
}

export async function checkHealth(): Promise<HealthStatus> {
  const res = await fetch(`${BASE}/api/health`);
  return asJson(res);
}

export async function saveSetup(data: {
  api_key: string;
  api_base: string;
  model: string;
  ncbi_email?: string;
  ncbi_api_key?: string;
}) {
  const res = await fetch(`${BASE}/api/setup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return asJson(res);
}

export async function fetchSessions(): Promise<Session[]> {
  const res = await fetch(`${BASE}/api/sessions`);
  return asJson(res);
}

export async function fetchSession(id: string) {
  const res = await fetch(`${BASE}/api/sessions/${id}`);
  return asJson(res);
}

export function startReview(
  question: string,
  maxPapers: number,
  language: string,
  instructions: string,
) {
  return fetch(`${BASE}/api/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      max_papers: maxPapers,
      language,
      instructions,
    }),
  });
}

export async function refineReview(instruction: string) {
  const res = await fetch(`${BASE}/api/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instruction }),
  });
  return asJson(res);
}

export async function sendChat(message: string) {
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  return asJson(res);
}

export async function uploadOutline(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/upload-outline`, {
    method: "POST",
    body: form,
  });
  return asJson(res);
}

export function startOutlineReview(
  outlineText: string,
  language: string,
  maxPapers: number,
  constraints = "",
  sectionFilter = "",
) {
  return fetch(`${BASE}/api/outline-review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      outline_text: outlineText,
      language,
      max_papers_per_section: maxPapers,
      constraints,
      section_filter: sectionFilter,
    }),
  });
}

export async function exportReview(format: string) {
  const res = await fetch(`${BASE}/api/export/${format}`);
  return asJson(res);
}

export async function readSSE(
  response: Response,
  onEvent: (event: string, data: any) => void,
) {
  const reader = response.body!.getReader();
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
        try {
          onEvent(event, JSON.parse(line.slice(6)));
        } catch {
          /* skip malformed */
        }
        event = null;
      }
    }
  }
}
