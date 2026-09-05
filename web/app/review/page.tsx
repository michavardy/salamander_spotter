"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api, mediaUrl } from "@/lib/api";
import { Panel, Button, TierBadge, Mono, Spinner } from "@/components/ui";
import { useReviewQueue, useReviewDetail, useRunMatch, useDecide, useBatches } from "@/lib/hooks";

export default function ReviewPage() {
  return (
    <Suspense fallback={<Spinner />}>
      <Review />
    </Suspense>
  );
}

function Review() {
  const params = useSearchParams();
  const id = params.get("id");
  return id ? <Detail imageId={id} /> : <Queue />;
}

function Queue() {
  const { data, isLoading } = useReviewQueue();
  const { data: batches } = useBatches();
  const router = useRouter();
  if (isLoading || !data) return <Spinner />;

  return (
    <div className="space-y-6">
      {batches?.batches?.[0] && (
        <Panel title="Batch">
          <div className="flex flex-wrap gap-4 text-sm">
            {batches.batches.map((b: any) => (
              <div key={b.id} className="rounded-lg border border-black/5 px-3 py-2">
                <div className="font-medium">
                  {b.name} <span className="text-xs text-ink/50">({b.status})</span>
                </div>
                <div className="text-xs text-ink/50">
                  {b.counts.decided}/{b.counts.total} decided
                </div>
                {b.status === "open" && (
                  <Button variant="ghost" onClick={() => api(`/batches/${b.id}/publish`, { json: {} }).then(() => location.reload())}>
                    Publish
                  </Button>
                )}
              </div>
            ))}
          </div>
        </Panel>
      )}

      <Panel title={`Review queue — ${data.count} sighting${data.count === 1 ? "" : "s"}`}>
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-ink/50">
            <tr>
              <th className="py-2">Sighting</th>
              <th>Status</th>
              <th>Tier</th>
              <th>Spots</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data.queue.map((r: any) => (
              <tr key={r.image_id} className="border-t border-black/5">
                <td className="py-2">
                  <Mono>{r.image_id}</Mono>
                </td>
                <td>{r.status}</td>
                <td>
                  <TierBadge tier={r.ladder_tier} />
                </td>
                <td>{r.n_spots ?? "—"}</td>
                <td className="text-right">
                  <Button variant="ghost" onClick={() => router.push(`/review?id=${r.image_id}`)}>
                    Open →
                  </Button>
                </td>
              </tr>
            ))}
            {data.queue.length === 0 && (
              <tr>
                <td colSpan={5} className="py-6 text-center text-ink/40">
                  Queue is clear.
                </td>
              </tr>
            )}
          </tbody>
        </table>
        {data.uncertain?.length > 0 && (
          <p className="mt-4 text-xs text-info">
            {data.uncertain.length} sighting(s) waiting in Uncertain — resolve from here or Names.
          </p>
        )}
      </Panel>
    </div>
  );
}

function Detail({ imageId }: { imageId: string }) {
  const { data, isLoading, refetch } = useReviewDetail(imageId);
  const match = useRunMatch();
  const decide = useDecide();
  const [showLines, setShowLines] = useState(false);
  const [override, setOverride] = useState<string | null>(null);
  if (isLoading || !data) return <Spinner />;

  const candidates = data.candidates ?? [];
  const chips: string[] = data.reason_chips ?? [];

  async function submit(verdict: string, chosen?: string, reason_chips?: string[]) {
    const res = await decide.mutateAsync({
      imageId,
      body: { verdict, chosen_individual_id: chosen, reason_chips, confirm_override: !!override },
    });
    if (res.needs_confirmation) {
      setOverride(res.warning);
      return;
    }
    setOverride(null);
    refetch();
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <a href="/review" className="text-sm text-accent hover:underline">
          ← Queue
        </a>
        <Mono>{imageId}</Mono>
        <TierBadge tier={data.image.ladder_tier} />
        <div className="ml-auto flex gap-2">
          <Button variant="ghost" onClick={() => setShowLines((v) => !v)}>
            {showLines ? "Cards" : "Match lines"}
          </Button>
          <Button onClick={() => match.mutate(imageId, { onSuccess: () => refetch() })}>
            {candidates.length ? "Re-run match" : "Run match"}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_1.2fr]">
        <Panel title="Query photo">
          <img src={mediaUrl(`/media/thumb/${imageId}.webp`)} alt={imageId} className="w-full rounded-lg" />
          <div className="mt-3 text-xs text-ink/60">
            {data.image.n_spots ?? "?"} spots · quality {fmt(data.image.q_overall)}
          </div>
        </Panel>

        <Panel title={`Candidates — top ${candidates.length}`}>
          {candidates.length === 0 && <p className="text-sm text-ink/50">Run the matcher to see candidates.</p>}
          <ul className="space-y-2">
            {candidates.map((c: any) => (
              <li key={c.rank} className="flex items-center gap-3 rounded-lg border border-black/5 p-2">
                <img
                  src={mediaUrl(`/media/thumb/${c.best_photo_id}.webp`)}
                  alt={c.individual_id}
                  className="h-14 w-14 rounded object-cover"
                />
                <div className="flex-1">
                  <Mono>{c.individual_id}</Mono>
                  <div className="text-xs text-ink/50">
                    sim {fmt(c.similarity)} · confidence {fmt(c.calibrated_confidence)}
                  </div>
                </div>
                {showLines ? (
                  <a
                    className="text-xs text-accent hover:underline"
                    href={`/review?id=${imageId}&lines=${c.individual_id}`}
                    onClick={(e) => {
                      e.preventDefault();
                      api(`/review/${imageId}/match-lines/${c.individual_id}`).then((r: any) =>
                        alert(`${r.links.length} spot links (geometric matcher)`),
                      );
                    }}
                  >
                    lines
                  </a>
                ) : (
                  <Button variant="primary" onClick={() => submit("confirm", c.individual_id, chips.slice(0, 1))}>
                    Confirm
                  </Button>
                )}
              </li>
            ))}
          </ul>

          {override && (
            <div className="mt-3 rounded-lg border border-warn/40 bg-warn/5 p-3 text-sm">
              {override}
              <div className="mt-2 flex gap-2">
                <Button variant="danger" onClick={() => submit("confirm", undefined)}>
                  Yes, override
                </Button>
                <Button variant="ghost" onClick={() => setOverride(null)}>
                  Cancel
                </Button>
              </div>
            </div>
          )}

          <div className="mt-4 flex flex-wrap gap-2 border-t border-black/5 pt-3">
            <Button onClick={() => submit("new")}>New individual</Button>
            <Button onClick={() => submit("uncertain")}>Uncertain — flag</Button>
            <Button variant="danger" onClick={() => submit("disqualify", undefined, ["possible duplicate frame"])}>
              Disqualify
            </Button>
          </div>
        </Panel>
      </div>

      <ExtractionEditor imageId={imageId} onDone={() => refetch()} />
    </div>
  );
}

function ExtractionEditor({ imageId, onDone }: { imageId: string; onDone: () => void }) {
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  return (
    <Panel title="Extraction editor">
      <div className="flex flex-wrap items-end gap-3 text-sm">
        <label className="flex flex-col">
          <span className="text-xs text-ink/50">join spot</span>
          <input className="w-20 rounded border border-black/10 px-2 py-1" value={a} onChange={(e) => setA(e.target.value)} />
        </label>
        <span className="pb-2">+</span>
        <label className="flex flex-col">
          <span className="text-xs text-ink/50">spot</span>
          <input className="w-20 rounded border border-black/10 px-2 py-1" value={b} onChange={(e) => setB(e.target.value)} />
        </label>
        <Button
          onClick={async () => {
            await api(`/images/${imageId}/extraction/edit`, {
              json: { ops: [{ op: "join_spots", spot_ids: [Number(a), Number(b)] }] },
            });
            setA("");
            setB("");
            onDone();
          }}
        >
          Join &amp; re-score
        </Button>
      </div>
    </Panel>
  );
}

const fmt = (n: number | null | undefined) => (n == null ? "—" : n.toFixed(2));
