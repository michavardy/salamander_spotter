"use client";

// API client (spec §4.2) — calls /api relative, reads any absolute base from
// /runtime-config.json, carries the CSRF cookie value as a header on mutations.

export type RuntimeConfig = {
  external_base_url: string | null;
  actor_label: string | null;
  features: Record<string, boolean>;
  version: string;
};

let _config: RuntimeConfig | null = null;
let _base = "";

export async function loadRuntimeConfig(): Promise<RuntimeConfig> {
  if (_config) return _config;
  const res = await fetch("/runtime-config.json", { cache: "no-store" });
  _config = (await res.json()) as RuntimeConfig;
  _base = _config.external_base_url?.replace(/\/$/, "") ?? "";
  return _config;
}

function csrfToken(): string | null {
  const m = document.cookie.match(/(?:^|;\s*)spotter_csrf=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

function actorLabel(): string | null {
  try {
    return localStorage.getItem("spotter_actor") || _config?.actor_label || null;
  } catch {
    return _config?.actor_label ?? null;
  }
}

export async function api<T = unknown>(
  path: string,
  opts: RequestInit & { json?: unknown } = {},
): Promise<T> {
  const { json, headers, ...rest } = opts;
  const method = (rest.method ?? (json !== undefined ? "POST" : "GET")).toUpperCase();
  const h = new Headers(headers);
  if (json !== undefined) h.set("content-type", "application/json");
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    const t = csrfToken();
    if (t) h.set("x-csrf-token", t);
  }
  const actor = actorLabel();
  if (actor) h.set("x-actor", actor);

  const res = await fetch(`${_base}/api${path}`, {
    ...rest,
    method,
    headers: h,
    body: json !== undefined ? JSON.stringify(json) : rest.body,
    credentials: "include",
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error((detail as any).detail || `${res.status} ${res.statusText}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function mediaUrl(rel: string): string {
  return `${_base}${rel}`;
}

/** Poll a background job (spec §5.3) until it settles — used for anything too
 * slow for one request/response (backup, full export, big imports). */
export async function pollJob(jobId: string, opts: { intervalMs?: number; timeoutMs?: number } = {}) {
  const { intervalMs = 1000, timeoutMs = 30 * 60_000 } = opts;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const job = await api<any>(`/jobs/${jobId}`);
    if (job.status === "done" || job.status === "failed" || job.status === "cancelled") return job;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(`job ${jobId} did not finish in time`);
}

export async function upload(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const h = new Headers();
  const t = csrfToken();
  if (t) h.set("x-csrf-token", t);
  const res = await fetch(`${_base}/api/uploads`, {
    method: "POST",
    body: fd,
    headers: h,
    credentials: "include",
  });
  if (!res.ok) throw new Error(`upload failed: ${res.status}`);
  return res.json();
}
