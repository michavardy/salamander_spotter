"use client";

import { useEffect, useRef, useState } from "react";
import { useJob, useJobLog, useInvalidate } from "@/lib/hooks";
import { useActiveTrainingJob } from "@/lib/trainingJob";

const ACTIVE = new Set(["queued", "running"]);

/** Notification-style floating card — visible from any page — showing the live
 * progress (fold-by-fold) and log of the currently running "Retrain all (full)"
 * job, if any. Persists across navigation and page reloads via useActiveTrainingJob. */
export function TrainingToast() {
  const { jobId, setJobId } = useActiveTrainingJob();
  const { data: job } = useJob(jobId);
  const [expanded, setExpanded] = useState(false);
  const { data: log } = useJobLog(jobId, !!job && ACTIVE.has(job.status));
  const inv = useInvalidate();
  const wasActive = useRef(false);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (!job) return;
    if (ACTIVE.has(job.status)) {
      wasActive.current = true;
    } else if (wasActive.current) {
      wasActive.current = false;
      inv("models", "dashboard");
    }
  }, [job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (expanded && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log?.text, expanded]);

  if (!jobId || !job) return null;

  const active = ACTIVE.has(job.status);
  const pct = Math.round((job.progress ?? 0) * 100);
  const title =
    job.status === "done" ? "Training complete" : job.status === "failed" ? "Training failed" : "Training run";

  return (
    <div className="fixed bottom-4 right-4 z-50 w-96 rounded-xl border border-black/10 bg-panel p-4 shadow-lg">
      <div className="flex items-center justify-between">
        <div className="font-head text-sm font-semibold">{title}</div>
        {!active && (
          <button
            className="text-xs text-ink/40 hover:text-ink"
            onClick={() => setJobId(null)}
            aria-label="Dismiss"
          >
            ✕
          </button>
        )}
      </div>

      <div className="mt-1 text-xs text-ink/60">{job.message || (active ? "Starting…" : "")}</div>

      {job.status !== "failed" && (
        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-black/10">
          <div
            className="h-full bg-accent transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
      {job.status === "failed" && (
        <p className="mt-2 text-xs text-danger">{String(job.error || "").split("\n")[0]}</p>
      )}

      <button
        className="mt-2 text-xs text-accent hover:underline"
        onClick={() => setExpanded((e) => !e)}
      >
        {expanded ? "Hide log" : "Show log"}
      </button>
      {expanded && (
        <pre
          ref={logRef}
          className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-black/90 p-2 font-mono text-[10px] leading-snug text-white/80"
        >
          {log?.text || "(no output yet)"}
        </pre>
      )}
    </div>
  );
}
