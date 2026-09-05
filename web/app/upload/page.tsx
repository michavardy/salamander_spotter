"use client";

import { useState } from "react";
import { upload, api } from "@/lib/api";
import { Panel, Button, TierBadge, Mono } from "@/components/ui";
import { useInvalidate } from "@/lib/hooks";

type Card = { name: string; image_id?: string; status?: string; ladder_tier?: string | null; n_spots?: number | null; error?: string };

export default function UploadPage() {
  const [cards, setCards] = useState<Card[]>([]);
  const [busy, setBusy] = useState(false);
  const [estimate, setEstimate] = useState<any>(null);
  const inv = useInvalidate();

  async function handleFiles(files: FileList | null) {
    if (!files || !files.length) return;
    const est = await api<any>("/uploads/estimate", { json: { count: files.length } });
    setEstimate(est);
    setBusy(true);
    for (const file of Array.from(files)) {
      setCards((c) => [{ name: file.name }, ...c]);
      try {
        const res = await upload(file);
        setCards((c) => c.map((x) => (x.name === file.name && !x.image_id ? { ...x, ...res } : x)));
      } catch (e: any) {
        setCards((c) => c.map((x) => (x.name === file.name && !x.image_id ? { ...x, error: e.message } : x)));
      }
    }
    setBusy(false);
    inv("dashboard", "review");
  }

  return (
    <div className="space-y-6">
      <Panel title="Upload images">
        <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-black/15 p-10 text-center hover:border-accent/50">
          <span className="font-head text-sm font-medium">Drop photos here or click to browse</span>
          <span className="mt-1 text-xs text-ink/50">JPG / PNG / HEIC · WhatsApp exports welcome</span>
          <input
            type="file"
            multiple
            accept="image/*,.heic"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
            disabled={busy}
          />
        </label>
        {estimate && (
          <p className="mt-3 text-xs text-ink/60">
            Est. {estimate.billed_calls_estimate} billed LLM calls
            {estimate.budget_remaining != null && ` · ${estimate.budget_remaining} left in today's budget`}
            {!estimate.within_budget && <span className="text-danger"> · over budget</span>}
          </p>
        )}
      </Panel>

      {cards.length > 0 && (
        <Panel title={`${cards.length} photo${cards.length > 1 ? "s" : ""}`}>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {cards.map((c, i) => (
              <div key={i} className="rounded-lg border border-black/5 p-3">
                <div className="truncate text-sm font-medium">{c.name}</div>
                <div className="mt-1 text-xs text-ink/50">
                  {c.image_id ? <Mono>{c.image_id}</Mono> : c.error ? <span className="text-danger">{c.error}</span> : "extracting…"}
                </div>
                <div className="mt-2 flex items-center justify-between">
                  <TierBadge tier={c.ladder_tier} />
                  {c.n_spots != null && <span className="text-xs text-ink/50">{c.n_spots} spots</span>}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4">
            <a href="/review" className="text-sm text-accent hover:underline">
              Review this batch now →
            </a>
          </div>
        </Panel>
      )}
    </div>
  );
}
