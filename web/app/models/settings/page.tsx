"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, mediaUrl, pollJob } from "@/lib/api";
import { Panel, Button, Spinner } from "@/components/ui";
import { useSettings, usePaths } from "@/lib/hooks";

const NUMERIC = [
  ["auto_approve_threshold", "Auto-approve confidence"],
  ["match_threshold", "Match suggestion threshold"],
  ["coverage_target", "Top-N coverage target"],
  ["max_candidates", "Max candidates"],
  ["quality_cutoff", "Quality cut-off"],
  ["min_spots_auto_accept", "Min spots for auto-accept"],
  ["daily_llm_budget", "Daily LLM call budget (0 = unlimited)"],
  ["retrain_every_days", "Retrain every N days"],
  ["retrain_after_images", "Retrain after N new images"],
  ["remote_logger_interval_minutes", "Remote logger interval (minutes)"],
] as const;

export default function SettingsPage() {
  const { data, isLoading, refetch } = useSettings();
  const paths = usePaths();
  const [form, setForm] = useState<Record<string, any>>({});
  const [secret, setSecret] = useState({ key: "llm_api_key", value: "" });
  const [datasetPath, setDatasetPath] = useState("datasets/all_sasa_norm_2026_23_07");
  const [importBusy, setImportBusy] = useState(false);
  const [importReport, setImportReport] = useState<any>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [dataJob, setDataJob] = useState<{ kind: "backup" | "export"; status: string; detail?: any } | null>(null);
  const [dataDirInput, setDataDirInput] = useState("");
  const [dataDirBusy, setDataDirBusy] = useState(false);
  const [dataDirMsg, setDataDirMsg] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    if (data) setForm(data.settings);
  }, [data]);

  useEffect(() => {
    if (paths.data && !dataDirInput) setDataDirInput(paths.data.data_dir.path);
  }, [paths.data]); // eslint-disable-line react-hooks/exhaustive-deps

  if (isLoading || !data) return <Spinner />;

  async function saveDataDir() {
    setDataDirBusy(true);
    setDataDirMsg(null);
    try {
      const result = await api<any>("/settings/data-dir", { json: { path: dataDirInput } });
      setDataDirMsg({
        ok: true,
        text: result.warning
          ? `Saved. ${result.warning}`
          : "Saved — restart the server for this to take effect.",
      });
    } catch (e: any) {
      setDataDirMsg({ ok: false, text: e.message });
    } finally {
      setDataDirBusy(false);
    }
  }

  async function save() {
    const values: Record<string, any> = {};
    for (const [k] of NUMERIC) values[k] = Number(form[k]);
    values.auto_promote = !!form.auto_promote;
    values.warn_on_override = !!form.warn_on_override;
    values.remote_logger_enabled = !!form.remote_logger_enabled;
    await api("/settings", { method: "PATCH", json: { values } });
    refetch();
  }

  async function runImport() {
    setImportBusy(true);
    setImportError(null);
    setImportReport(null);
    try {
      const report = await api<any>("/imports", { json: { dataset_path: datasetPath } });
      setImportReport(report);
    } catch (e: any) {
      setImportError(e.message);
    } finally {
      setImportBusy(false);
    }
  }

  async function runBackup() {
    setDataJob({ kind: "backup", status: "running" });
    const { job_id, dest } = await api<any>("/backup", { json: {} });
    const job = await pollJob(job_id);
    setDataJob({ kind: "backup", status: job.status, detail: { ...job.result, dest } });
  }

  async function runExportFull() {
    setDataJob({ kind: "export", status: "running" });
    const { job_id, download } = await api<any>("/exports/full", { json: {} });
    const job = await pollJob(job_id);
    setDataJob({ kind: "export", status: job.status, detail: { ...job.result, download } });
  }

  return (
    <div className="space-y-6">
      <div className="flex gap-4 text-sm">
        <Link href="/models" className="text-ink/50 hover:text-ink">
          Models
        </Link>
        <Link href="/models/settings" className="font-semibold">
          Settings
        </Link>
      </div>

      <Panel title="Import existing data (spec §7.9 A)">
        <p className="mb-3 text-sm text-ink/60">
          One-time / delta transfer of a packaged <code className="font-mono">datasets/&lt;name&gt;/</code> folder
          (raw images + <code className="font-mono">contours.db</code>) into this app. Copies files and derives
          the roster — zero LLM calls, safe to re-run, and a later run only imports what's new.
        </p>
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-1 min-w-[280px] flex-col text-sm">
            <span className="text-xs text-ink/50">Dataset path (on the server)</span>
            <input
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              className="rounded border border-black/10 px-2 py-1 font-mono text-xs"
            />
          </label>
          <Button variant="primary" onClick={runImport} disabled={importBusy}>
            {importBusy ? "Importing…" : "Import"}
          </Button>
        </div>
        {importError && <p className="mt-3 text-sm text-danger">{importError}</p>}
        {importReport && (
          <dl className="mt-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
            <Field k="Images added" v={importReport.images_added} />
            <Field k="Images skipped" v={importReport.images_skipped} />
            <Field k="Individuals added" v={importReport.individuals_added} />
            <Field k="Synthetic views" v={importReport.synthetic_views_added} />
            <Field k="Merges applied" v={importReport.merges_applied} />
            <Field k="Exclusions applied" v={importReport.exclusions_applied} />
            <Field k="LLM calls" v={importReport.llm_calls} />
            <Field k="Contours copied verbatim" v={String(importReport.contours_db_copied_verbatim)} />
          </dl>
        )}
      </Panel>

      <Panel title="Data & storage (spec §4.3, §2.2, §2.3)">
        <p className="mb-3 text-sm text-ink/60">
          Everything below lives under one data directory — images, databases, models, exports, logs. Change it
          here and restart the server to point at a different local dir (DuckDB allows one read-write connection
          per file, so this can't take effect live); the same layout is what an export archive carries to another
          host.
        </p>
        <div className="mb-4 flex flex-wrap items-end gap-3">
          <label className="flex flex-1 min-w-[320px] flex-col text-sm">
            <span className="text-xs text-ink/50">Data directory</span>
            <input
              value={dataDirInput}
              onChange={(e) => setDataDirInput(e.target.value)}
              placeholder="C:\Users\you\AppData\Roaming\SalamanderSpotter"
              className="rounded border border-black/10 px-2 py-1 font-mono text-xs"
            />
          </label>
          <Button variant="primary" onClick={saveDataDir} disabled={dataDirBusy || !dataDirInput}>
            {dataDirBusy ? "Saving…" : "Save (requires restart)"}
          </Button>
        </div>
        {dataDirMsg && (
          <p className={`mb-4 text-sm ${dataDirMsg.ok ? "text-confirm" : "text-danger"}`}>{dataDirMsg.text}</p>
        )}
        {paths.data?.env_override && (
          <p className="mb-4 text-sm text-warn">
            <code className="font-mono">SPOTTER_DATA_DIR</code> is set in the environment and overrides whatever
            is saved here until it's unset.
          </p>
        )}
        {paths.isLoading || !paths.data ? (
          <Spinner />
        ) : (
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase text-ink/50">
              <tr>
                <th className="py-1">Location</th>
                <th>Path</th>
                <th>Files</th>
                <th>Size</th>
              </tr>
            </thead>
            <tbody>
              {(
                [
                  ["Data dir", "data_dir"],
                  ["App database", "app_db"],
                  ["Contours database", "contours_db"],
                  ["Images — raw", "images_raw"],
                  ["Images — purple", "images_purple"],
                  ["Images — thumbnails", "images_thumb"],
                  ["Models", "models_dir"],
                  ["Exports", "exports_dir"],
                  ["Logs", "logs_dir"],
                  ["Secrets file", "secrets_file"],
                ] as const
              ).map(([label, key]) => {
                const p = paths.data[key];
                return (
                  <tr key={key} className="border-t border-black/5">
                    <td className="py-1.5">{label}</td>
                    <td className="max-w-[380px] truncate font-mono text-xs" title={p.path}>
                      {p.path}
                    </td>
                    <td>{p.exists ? p.n_files : "—"}</td>
                    <td>{p.exists ? formatBytes(p.bytes) : <span className="text-ink/30">missing</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}

        <div className="mt-4 flex flex-wrap gap-3 border-t border-black/5 pt-4">
          <Button onClick={runBackup} disabled={dataJob?.status === "running"}>
            {dataJob?.kind === "backup" && dataJob.status === "running" ? "Backing up…" : "Create backup now"}
          </Button>
          <Button variant="primary" onClick={runExportFull} disabled={dataJob?.status === "running"}>
            {dataJob?.kind === "export" && dataJob.status === "running"
              ? "Zipping…"
              : "Export full archive (for another host)"}
          </Button>
        </div>
        {dataJob && dataJob.status !== "running" && (
          <div className="mt-3 rounded-lg border border-black/5 bg-black/[0.02] p-3 text-sm">
            {dataJob.status === "failed" ? (
              <span className="text-danger">Failed — see server logs.</span>
            ) : dataJob.kind === "backup" ? (
              <>
                Backup written to <span className="font-mono text-xs">{dataJob.detail?.archive}</span> (
                {formatBytes(dataJob.detail?.bytes ?? 0)}).
              </>
            ) : (
              <>
                Archive ready ({formatBytes(dataJob.detail?.bytes ?? 0)}) —{" "}
                <a className="text-accent hover:underline" href={mediaUrl(dataJob.detail?.download)}>
                  download it
                </a>
                , then on the remote: <code className="font-mono text-xs">app import --full &lt;file&gt;</code>.
              </>
            )}
          </div>
        )}
        <p className="mt-2 text-xs text-ink/50">
          Both run live via the server's own DB connection (safe while it keeps serving traffic) — no need to
          stop anything first.
        </p>
      </Panel>

      <Panel title="Review & automation" right={<Button variant="primary" onClick={save}>Save</Button>}>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {NUMERIC.map(([k, label]) => (
            <label key={k} className="flex flex-col text-sm">
              <span className="text-xs text-ink/50">{label}</span>
              <input
                type="number"
                step="0.01"
                value={form[k] ?? ""}
                onChange={(e) => setForm((f) => ({ ...f, [k]: e.target.value }))}
                className="rounded border border-black/10 px-2 py-1"
              />
            </label>
          ))}
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={!!form.auto_promote}
              onChange={(e) => setForm((f) => ({ ...f, auto_promote: e.target.checked }))}
            />
            Auto-promote a better model
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={!!form.warn_on_override}
              onChange={(e) => setForm((f) => ({ ...f, warn_on_override: e.target.checked }))}
            />
            Warn before assigning against the suggestion
          </label>
        </div>
      </Panel>

      <Panel title="Remote logger" right={<Button variant="primary" onClick={save}>Save</Button>}>
        <p className="mb-3 text-sm text-ink/60">
          While a training run is active, append a log-tail snapshot to a relay file every N minutes
          (instead of only the live job log) — meant for a Claude Code session with Remote Control
          connected to watch and push you updates on a long run. Off by default; has no effect unless
          something is actually watching the relay file.
        </p>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={!!form.remote_logger_enabled}
            onChange={(e) => setForm((f) => ({ ...f, remote_logger_enabled: e.target.checked }))}
          />
          Enable remote logger during training runs
        </label>
      </Panel>

      <Panel title="Score formula">
        <p className="text-sm text-ink/60">
          Score = Σ coef·metric over letters a–f (a=R@1, b=R@5, c=R@10, d=bal.acc, e=novelty AUROC, f=review@90).
          Current: <span className="font-mono">{JSON.stringify(data.settings.score_coefficients)}</span>
        </p>
      </Panel>

      <Panel title="Pipeline & providers">
        <div className="flex flex-wrap items-end gap-3 text-sm">
          <label className="flex flex-col">
            <span className="text-xs text-ink/50">Secret</span>
            <select
              value={secret.key}
              onChange={(e) => setSecret((s) => ({ ...s, key: e.target.value }))}
              className="rounded border border-black/10 px-2 py-1"
            >
              {Object.keys(data.secrets).map((k) => (
                <option key={k}>{k}</option>
              ))}
            </select>
          </label>
          <input
            type="password"
            placeholder="value"
            value={secret.value}
            onChange={(e) => setSecret((s) => ({ ...s, value: e.target.value }))}
            className="rounded border border-black/10 px-2 py-1"
          />
          <Button onClick={() => api("/settings/secrets", { method: "PUT", json: secret }).then(() => { setSecret((s) => ({ ...s, value: "" })); refetch(); })}>
            Set
          </Button>
          <Button variant="ghost" onClick={() => api("/settings/test-llm", { json: {} }).then((r: any) => alert(JSON.stringify(r)))}>
            Test
          </Button>
        </div>
        <ul className="mt-3 text-xs text-ink/50">
          {Object.entries(data.secrets).map(([k, v]: any) => (
            <li key={k}>
              {k}: {v.set ? `set ••••${v.hint}` : "not set"}
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}

function Field({ k, v }: { k: string; v: any }) {
  return (
    <div>
      <dt className="text-xs text-ink/50">{k}</dt>
      <dd className="font-mono">{v ?? "—"}</dd>
    </div>
  );
}

function formatBytes(n: number): string {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}
