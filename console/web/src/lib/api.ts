// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** The console API: same origin, JSON, and the header every change needs. */

export const REQUEST_HEADER = "X-Console-Request";

export class ApiError extends Error {
  status: number;
  /** The server's error code (``code``), named in the page's language. */
  code: string;
  body: Record<string, unknown>;
  constructor(status: number, message: string, body: Record<string, unknown>) {
    super(message);
    this.status = status;
    this.code = typeof body.code === "string" ? body.code : status === 404 ? "not_found" : "failed";
    this.body = body;
  }
}

type Translate = (key: string, vars?: Record<string, string | number>) => string;

/** What the page says of a failure inside the server (error class
 * ``failure``), whose detail the server keeps in its log: in the page's
 * language, with the ids the server sent (a question's ``query_id``, its
 * ``trace_id``) to find its log lines and trace by. */
export function failureHelp(t: Translate, ids: { query_id?: unknown; trace_id?: unknown }): string {
  const named = (value: unknown) => (typeof value === "string" && value ? value : undefined);
  const queryId = named(ids.query_id);
  const traceId = named(ids.trace_id);
  const help = queryId ? t("ask.error.failureHelp", { query_id: queryId }) : t("error.failure");
  return traceId ? `${help} ${t("error.traceId", { trace_id: traceId })}` : help;
}

/** An error as the page shows it: what kind of refusal it is, in the
 * page's language (a request's own kind, else its error class as the Ask
 * page names it), then the server's detail; a failure inside the server
 * as :func:`failureHelp` says it (its detail is the server log's). */
export function errorMessage(t: Translate, error: unknown): string {
  if (error instanceof ApiError) {
    if (error.code === "failure") return failureHelp(t, error.body);
    const key = `error.${error.code}`;
    const kind = t(key);
    if (kind !== key) return `${kind} ${error.message}`;
    const classKey = `ask.error.${error.code}`;
    const errorClass = t(classKey);
    return errorClass === classKey ? error.message : `${errorClass}: ${error.message}`;
  }
  if (error instanceof TypeError) return t("error.network");
  return error instanceof Error ? error.message : String(error);
}

async function parse<T>(response: Response): Promise<T> {
  const text = await response.text();
  let body: Record<string, unknown>;
  try {
    body = text ? (JSON.parse(text) as Record<string, unknown>) : {};
  } catch {
    body = { error: text };
  }
  if (!response.ok) {
    const message = String(body.error ?? body.detail ?? response.statusText);
    throw new ApiError(response.status, message, body);
  }
  return body as T;
}

export async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const query = params
    ? "?" +
      Object.entries(params)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join("&")
    : "";
  return parse<T>(await fetch(`/api/${path}${query}`, { headers: { Accept: "application/json" } }));
}

export async function send<T>(path: string, body?: unknown, method = "POST"): Promise<T> {
  return parse<T>(
    await fetch(`/api/${path}`, {
      method,
      headers: { "Content-Type": "application/json", [REQUEST_HEADER]: "1" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
}

export async function upload<T>(path: string, form: FormData): Promise<T> {
  return parse<T>(await fetch(`/api/${path}`, { method: "POST", headers: { [REQUEST_HEADER]: "1" }, body: form }));
}

/** Server-sent events of a POST: each `data:` line parsed as JSON. */
export async function* stream<T>(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<T> {
  const response = await fetch(`/api/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream", [REQUEST_HEADER]: "1" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    await parse(response);
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) yield JSON.parse(data) as T;
    }
  }
}
