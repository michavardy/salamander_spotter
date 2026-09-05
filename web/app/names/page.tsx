"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api, mediaUrl } from "@/lib/api";
import { Panel, Button, TierBadge, HealthDot, Mono, Spinner, Thumb } from "@/components/ui";
import { useIndividuals, useIndividual, useImage } from "@/lib/hooks";

export default function NamesPage() {
  return (
    <Suspense fallback={<Spinner />}>
      <Names />
    </Suspense>
  );
}

function Names() {
  const params = useSearchParams();
  const id = params.get("id");
  const img = params.get("img");
  if (img && id) return <ImageView individualId={id} imageId={img} />;
  if (id) return <IndividualView id={id} />;
  return <Roster />;
}

function Roster() {
  const [q, setQ] = useState("");
  const { data, isLoading } = useIndividuals(q || undefined);
  const router = useRouter();

  return (
    <Panel
      title="Names"
      right={
        <div className="flex gap-2">
          <input
            placeholder="Search…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="rounded border border-black/10 px-2 py-1 text-sm"
          />
          <Button onClick={() => api("/exports/census?fmt=xlsx", { json: {} }).then((r: any) => window.open(mediaUrl(r.download)))}>
            Export
          </Button>
        </div>
      }
    >
      {isLoading || !data ? (
        <Spinner />
      ) : (
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-ink/50">
            <tr>
              <th className="py-2"></th>
              <th>ID</th>
              <th>Nickname</th>
              <th>Status</th>
              <th>Photos</th>
              <th>Health</th>
              <th>Last seen</th>
            </tr>
          </thead>
          <tbody>
            {data.individuals.map((r: any) => (
              <tr
                key={r.individual_id}
                className="cursor-pointer border-t border-black/5 hover:bg-black/[0.02]"
                onClick={() => router.push(`/names?id=${r.individual_id}`)}
              >
                <td className="py-1.5">
                  <Thumb imageId={r.reference_image_id} size={36} />
                </td>
                <td>
                  <Mono>{r.display_id}</Mono>
                </td>
                <td>{r.nickname ?? "—"}</td>
                <td>{r.status}</td>
                <td>
                  {r.n_real}
                  {r.n_images !== r.n_real && <span className="text-ink/40"> (+{r.n_images - r.n_real})</span>}
                </td>
                <td>
                  <HealthDot health={r.health} />
                </td>
                <td className="text-xs text-ink/50">{r.last_seen ? new Date(r.last_seen).toLocaleDateString() : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="mt-2 text-xs text-ink/50">{data?.total ?? 0} individuals</div>
    </Panel>
  );
}

function IndividualView({ id }: { id: string }) {
  const { data, isLoading } = useIndividual(id);
  const router = useRouter();
  const [nickname, setNickname] = useState("");
  if (isLoading || !data) return <Spinner />;
  const ind = data.individual;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <a href="/names" className="text-sm text-accent hover:underline">
          ← Names
        </a>
        <Mono>{ind.display_id}</Mono>
        <span className="text-xs text-ink/50">{ind.status}</span>
      </div>

      <Panel
        title="Details"
        right={
          <div className="flex gap-2">
            <input
              placeholder={ind.nickname ?? "nickname"}
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
              className="rounded border border-black/10 px-2 py-1 text-sm"
            />
            <Button
              onClick={() =>
                api(`/individuals/${id}`, { method: "PATCH", json: { nickname, rev: ind.rev } }).then(() =>
                  location.reload(),
                )
              }
            >
              Save
            </Button>
          </div>
        }
      >
        <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
          <div>
            <dt className="text-xs text-ink/50">First seen</dt>
            <dd>{ind.first_seen ? new Date(ind.first_seen).toLocaleDateString() : "—"}</dd>
          </div>
          <div>
            <dt className="text-xs text-ink/50">Last seen</dt>
            <dd>{ind.last_seen ? new Date(ind.last_seen).toLocaleDateString() : "—"}</dd>
          </div>
          <div>
            <dt className="text-xs text-ink/50">Contributors</dt>
            <dd>{data.contributors.map((c: any) => c.name).join(", ") || "—"}</dd>
          </div>
        </dl>
      </Panel>

      <Panel title={`${data.images.length} photos`}>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
          {data.images.map((im: any) => (
            <button
              key={im.image_id}
              onClick={() => router.push(`/names?id=${id}&img=${im.image_id}`)}
              className="lift rounded-lg border border-black/5 p-2 text-left"
            >
              <img src={mediaUrl(`/media/thumb/${im.image_id}.webp`)} alt={im.image_id} className="aspect-square w-full rounded object-cover" />
              <div className="mt-1 truncate text-xs">
                <Mono>{im.image_id}</Mono>
              </div>
              <div className="mt-1 flex items-center justify-between">
                <TierBadge tier={im.ladder_tier} />
                {im.is_synthetic && <span className="text-[10px] text-info">synthetic</span>}
              </div>
            </button>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function ImageView({ individualId, imageId }: { individualId: string; imageId: string }) {
  const { data, isLoading } = useImage(imageId);
  if (isLoading || !data) return <Spinner />;
  const im = data.image;
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <a href={`/names?id=${individualId}`} className="text-sm text-accent hover:underline">
          ← {individualId}
        </a>
        <Mono>{imageId}</Mono>
        <TierBadge tier={im.ladder_tier} />
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel title="Photo">
          <img src={mediaUrl(`/media/raw/${imageId}${extGuess(im.source_filename)}`)} alt={imageId} className="w-full rounded-lg" onError={(e) => ((e.target as HTMLImageElement).src = mediaUrl(`/media/thumb/${imageId}.webp`))} />
        </Panel>
        <Panel title="Extraction">
          <dl className="grid grid-cols-2 gap-3 text-sm">
            <Field k="Spots" v={im.n_spots} />
            <Field k="Overall quality" v={fmt(im.q_overall)} />
            <Field k="Blur" v={fmt(im.q_blur)} />
            <Field k="Lighting" v={fmt(im.q_lighting)} />
            <Field k="Spot" v={fmt(im.q_spot)} />
            <Field k="Body" v={fmt(im.q_body)} />
            <Field k="Status" v={im.status} />
            <Field k="Origin" v={im.origin} />
          </dl>
          {data.corrections?.length > 0 && (
            <p className="mt-3 text-xs text-ink/50">{data.corrections.length} correction(s) on record.</p>
          )}
        </Panel>
      </div>
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

const fmt = (n: number | null | undefined) => (n == null ? "—" : Number(n).toFixed(2));
const extGuess = (name?: string) => (name && name.includes(".") ? name.slice(name.lastIndexOf(".")) : ".jpg");
