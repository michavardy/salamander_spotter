"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Panel, Button, Mono, Spinner } from "@/components/ui";
import { useModels, useInvalidate, useJob } from "@/lib/hooks";
import { useActiveTrainingJob } from "@/lib/trainingJob";

const METRIC_FIELDS = [
  ["r1", "R@1"],
  ["r5", "R@5"],
  ["r10", "R@10"],
  ["bal_acc", "Bal. acc"],
  ["novelty_auroc", "Novelty AUROC"],
  ["review_at_90", "Review@90"],
] as const;

export default function ModelsPage() {
  const { data, isLoading, refetch } = useModels();
  const inv = useInvalidate();
  const [importForm, setImportForm] = useState<Record<string, string>>({
    name: "", kind: "custom", source_weights_path: "", source_calibration_path: "",
  });
  const [importBusy, setImportBusy] = useState(false);
  const [importMsg, setImportMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const { jobId, setJobId } = useActiveTrainingJob();
  const { data: job } = useJob(jobId);
  const trainingActive = !!job && (job.status === "queued" || job.status === "running");

  // Reload recovery: a training run in progress when the page loads (or reloads mid-run,
  // since the real run can take hours) reattaches to its live job instead of losing the bar/log.
  useEffect(() => {
    const latest = data?.runs?.[0];
    if (!jobId && latest?.status === "running" && latest?.job_id) setJobId(latest.job_id);
  }, [data?.runs, jobId, setJobId]);

  if (isLoading || !data) return <Spinner />;

  async function startRetrain() {
    const { job_id } = await api<{ job_id: string }>("/models/retrain", { json: {} });
    setJobId(job_id);
  }

  async function submitImport() {
    setImportBusy(true);
    setImportMsg(null);
    const metrics: Record<string, number> = {};
    for (const [k] of METRIC_FIELDS) {
      const v = importForm[k];
      if (v !== undefined && v !== "") metrics[k] = Number(v);
    }
    try {
      const result = await api<any>("/models/import", {
        json: {
          name: importForm.name,
          kind: importForm.kind || "custom",
          source_weights_path: importForm.source_weights_path,
          source_calibration_path: importForm.source_calibration_path || undefined,
          metrics,
          make_active: !!importForm.make_active,
        },
      });
      setImportMsg({ ok: true, text: `Registered “${result.name}” — score ${Number(result.score).toFixed(3)}.` });
      setImportForm({ name: "", kind: "custom", source_weights_path: "", source_calibration_path: "" });
      inv("models", "dashboard");
      refetch();
    } catch (e: any) {
      setImportMsg({ ok: false, text: e.message });
    } finally {
      setImportBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div className="flex gap-4 text-sm">
          <Link href="/models" className="font-semibold">
            Models
          </Link>
          <Link href="/models/settings" className="text-ink/50 hover:text-ink">
            Settings
          </Link>
        </div>
        <div className="ml-auto flex gap-2">
          <Button onClick={() => api("/models/rollback", { json: {} }).then(() => refetch())}>Rollback</Button>
          <Button variant="primary" onClick={startRetrain} disabled={trainingActive}>
            {trainingActive ? "Training…" : "Retrain all (full)"}
          </Button>
        </div>
      </div>

      {data.training_due?.due && (
        <div className="rounded-lg border border-accent/30 bg-accent/5 px-4 py-2 text-sm">
          Training is due ({data.training_due.by_time ? "schedule" : `${data.training_due.new_images_since_last} new images`}).
        </div>
      )}
      {data.singleton_hint?.singletons > 0 && (
        <div className="rounded-lg border border-warn/30 bg-warn/5 px-4 py-2 text-sm">
          {data.singleton_hint.singletons} individual(s) would benefit from synthetic views.
        </div>
      )}

      <Panel title="Register an already-trained model (spec §7.6)">
        <p className="mb-3 text-sm text-ink/60">
          Point at a checkpoint you already produced from your research (e.g. a{" "}
          <code className="font-mono">pipeline/spot_transformer</code> sweep output). This copies the file onto
          the app's volume and catalogues it — zero GPU/LLM cost, no retraining. Metrics are optional: fill in
          whatever you already measured (from a bakeoff / <code className="font-mono">results.md</code> run) or
          leave them blank and add them later.
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Name">
            <input
              value={importForm.name}
              onChange={(e) => setImportForm((f) => ({ ...f, name: e.target.value }))}
              placeholder="e2e_transformer_fold0"
              className="rounded border border-black/10 px-2 py-1 font-mono text-xs"
            />
          </Field>
          <Field label="Kind (groups 'best per kind')">
            <input
              value={importForm.kind}
              onChange={(e) => setImportForm((f) => ({ ...f, kind: e.target.value }))}
              placeholder="aggregator"
              className="rounded border border-black/10 px-2 py-1 text-sm"
            />
          </Field>
          <Field label="Weights path (on the server)">
            <input
              value={importForm.source_weights_path}
              onChange={(e) => setImportForm((f) => ({ ...f, source_weights_path: e.target.value }))}
              placeholder="artifacts/spot_transformer/sweeps/strict/e2e_ckpt_fold0_transformer.pt"
              className="rounded border border-black/10 px-2 py-1 font-mono text-xs"
            />
          </Field>
          <Field label="Calibration path (optional)">
            <input
              value={importForm.source_calibration_path}
              onChange={(e) => setImportForm((f) => ({ ...f, source_calibration_path: e.target.value }))}
              placeholder="calibration.json"
              className="rounded border border-black/10 px-2 py-1 font-mono text-xs"
            />
          </Field>
          {METRIC_FIELDS.map(([k, label]) => (
            <Field key={k} label={label}>
              <input
                type="number"
                step="0.001"
                value={importForm[k] ?? ""}
                onChange={(e) => setImportForm((f) => ({ ...f, [k]: e.target.value }))}
                className="rounded border border-black/10 px-2 py-1"
              />
            </Field>
          ))}
        </div>
        <div className="mt-3 flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={!!importForm.make_active}
              onChange={(e) => setImportForm((f) => ({ ...f, make_active: e.target.checked ? "1" : "" }))}
            />
            Make active immediately
          </label>
          <Button
            variant="primary"
            disabled={importBusy || !importForm.name || !importForm.source_weights_path}
            onClick={submitImport}
          >
            {importBusy ? "Registering…" : "Register model"}
          </Button>
        </div>
        {importMsg && (
          <p className={`mt-2 text-sm ${importMsg.ok ? "text-confirm" : "text-danger"}`}>{importMsg.text}</p>
        )}
        <p className="mt-2 text-xs text-ink/50">
          Note: registering a model catalogues it and — once made active — is what a future training run
          compares against. Real-time matching against uploaded photos isn't wired to loaded weights yet
          (that lands with the M2/M3 pipeline bridges).
        </p>
      </Panel>

      <Panel title="Registered models">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-ink/50">
            <tr>
              <th className="py-2">Name</th>
              <th>Kind</th>
              <th>Status</th>
              <th>R@1</th>
              <th>Novelty</th>
              <th>Score</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data.models.map((m: any) => (
              <tr key={m.name} className="border-t border-black/5">
                <td className="py-2">
                  <Mono>{m.name}</Mono>
                </td>
                <td>{m.kind}</td>
                <td>
                  <span
                    className={`rounded px-2 py-0.5 text-xs ${
                      m.status === "active"
                        ? "bg-confirm/10 text-confirm"
                        : m.status === "best"
                        ? "bg-info/10 text-info"
                        : "bg-ink/10 text-ink/60"
                    }`}
                  >
                    {m.status}
                  </span>
                </td>
                <td className="font-mono">{fmt(m.r1)}</td>
                <td className="font-mono">{fmt(m.novelty_auroc)}</td>
                <td className="font-mono">{fmt(m.score)}</td>
                <td className="text-right">
                  {m.status !== "active" && (
                    <Button variant="ghost" onClick={() => api("/models/promote", { json: { name: m.name } }).then(() => refetch())}>
                      Promote
                    </Button>
                  )}
                </td>
              </tr>
            ))}
            {data.models.length === 0 && (
              <tr>
                <td colSpan={7} className="py-6 text-center text-ink/40">
                  No models registered yet — use the panel above, or run a training pass.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Panel>

      <Panel title="Training runs">
        <ul className="space-y-2 text-sm">
          {data.runs.map((r: any) => (
            <li key={r.id} className="flex justify-between border-b border-black/5 pb-2">
              <span>
                <Mono>{r.id}</Mono> · {r.trigger} · {r.status}
              </span>
              <span className="text-xs text-ink/50">
                {r.promotion_to ? `promoted ${r.promotion_to}` : "no promotion"}
              </span>
            </li>
          ))}
          {data.runs.length === 0 && <li className="text-ink/40">No runs yet.</li>}
        </ul>
      </Panel>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col text-sm">
      <span className="text-xs text-ink/50">{label}</span>
      {children}
    </label>
  );
}

const fmt = (n: number | null | undefined) => (n == null ? "—" : Number(n).toFixed(3));
