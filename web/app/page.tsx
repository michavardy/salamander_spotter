"use client";

import Link from "next/link";
import { useDashboard } from "@/lib/hooks";
import { Panel, StatCard, Spinner, TierBadge } from "@/components/ui";
import { api } from "@/lib/api";

export default function DashboardPage() {
  const { data, isLoading } = useDashboard();
  if (isLoading || !data) return <Spinner />;

  const s = data.stats;
  const d = data.deltas ?? {};
  return (
    <div className="space-y-6">
      {data.alert && (
        <div className="flex items-center justify-between rounded-xl border border-accent/30 bg-accent/5 px-4 py-3">
          <div>
            <div className="font-head text-sm font-semibold">{data.alert.title}</div>
            <div className="text-sm text-ink/70">{data.alert.body}</div>
          </div>
          <button
            className="text-xs text-ink/50 hover:text-ink"
            onClick={() => api(`/notifications/${data.alert.id}/dismiss`, { json: {} })}
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Individuals" value={s.individuals} delta={d.individuals ? `+${d.individuals} this month` : undefined} />
        <StatCard label="Images" value={s.images} delta={d.images ? `+${d.images} this month` : undefined} />
        <StatCard label="Contributors" value={s.contributors} />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel title="Recent activity">
          <ul className="space-y-3">
            {data.activity.slice(0, 12).map((a: any, i: number) => (
              <li key={i} className="text-sm">
                <div className="font-medium">{a.summary}</div>
                <div className="text-xs text-ink/50">
                  {a.kind} · {new Date(a.created_at).toLocaleString()}
                </div>
              </li>
            ))}
            {data.activity.length === 0 && <li className="text-sm text-ink/40">Nothing yet.</li>}
          </ul>
        </Panel>

        <div className="space-y-6">
          <Panel title="Census">
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <Stat k="Published individuals" v={data.census.published_individuals} />
              <Stat k="Confirmed sightings" v={data.census.confirmed_sightings} />
              <Stat k="Provisional" v={data.census.provisional_individuals} />
              <Stat k="In review queue" v={data.review_queue} />
            </dl>
            <Link href="/review" className="mt-3 inline-block text-sm text-accent hover:underline">
              Go to review →
            </Link>
          </Panel>

          <Panel title="Quality tiers">
            <div className="space-y-2">
              {Object.entries(data.tiers).map(([tier, n]) => (
                <div key={tier} className="flex items-center justify-between text-sm">
                  <TierBadge tier={tier} />
                  <span className="font-mono">{n as number}</span>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Stat({ k, v }: { k: string; v: number }) {
  return (
    <div>
      <dt className="text-xs text-ink/50">{k}</dt>
      <dd className="font-head text-xl font-semibold">{v}</dd>
    </div>
  );
}
