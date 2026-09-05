"use client";

import type { ReactNode } from "react";
import { useState } from "react";
import { PawPrint } from "lucide-react";
import { mediaUrl } from "@/lib/api";

/** A small, fixed-size salamander thumbnail — falls back to a neutral icon
 * when there's no reference photo yet or the thumbnail 404s. */
export function Thumb({
  imageId,
  size = 36,
  rounded = "rounded-md",
}: {
  imageId?: string | null;
  size?: number;
  rounded?: string;
}) {
  const [broken, setBroken] = useState(false);
  const style = { width: size, height: size, minWidth: size };

  if (!imageId || broken) {
    return (
      <div style={style} className={`flex shrink-0 items-center justify-center bg-black/5 text-ink/25 ${rounded}`}>
        <PawPrint size={Math.round(size * 0.55)} strokeWidth={1.5} />
      </div>
    );
  }
  return (
    <img
      src={mediaUrl(`/media/thumb/${imageId}.webp`)}
      alt={imageId}
      style={style}
      loading="lazy"
      onError={() => setBroken(true)}
      className={`shrink-0 object-cover ${rounded}`}
    />
  );
}

export function Panel({ title, children, right }: { title?: ReactNode; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="rounded-xl border border-black/5 bg-panel shadow-card">
      {title && (
        <header className="flex items-center justify-between border-b border-black/5 px-4 py-3">
          <h2 className="font-head text-sm font-semibold">{title}</h2>
          {right}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function StatCard({ label, value, delta }: { label: string; value: ReactNode; delta?: string }) {
  return (
    <div className="lift rounded-xl border border-black/5 bg-panel p-4 shadow-card">
      <div className="text-xs uppercase tracking-wide text-ink/50">{label}</div>
      <div className="mt-1 font-head text-3xl font-semibold">{value}</div>
      {delta && <div className="mt-1 text-xs text-confirm">{delta}</div>}
    </div>
  );
}

const TIER_STYLES: Record<string, string> = {
  auto_accept: "bg-confirm/10 text-confirm",
  needs_a_look: "bg-warn/10 text-warn",
  hand_correction: "bg-danger/10 text-danger",
  failed: "bg-ink/10 text-ink/60",
};

export function TierBadge({ tier }: { tier?: string | null }) {
  if (!tier) return <span className="text-ink/30">—</span>;
  return (
    <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${TIER_STYLES[tier] ?? "bg-ink/10"}`}>
      {tier.replace(/_/g, " ")}
    </span>
  );
}

export function HealthDot({ health }: { health?: string }) {
  const c = health === "strong" ? "bg-confirm" : health === "weak" ? "bg-danger" : "bg-warn";
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${c}`} title={health} />;
}

export function Mono({ children }: { children: ReactNode }) {
  return <span className="font-mono text-[0.85em]">{children}</span>;
}

export function Button({
  children,
  onClick,
  variant = "default",
  disabled,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "default" | "primary" | "danger" | "ghost";
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  const styles = {
    default: "border border-black/10 bg-panel hover:bg-black/5",
    primary: "bg-accent text-white hover:bg-accent/90",
    danger: "bg-danger text-white hover:bg-danger/90",
    ghost: "hover:bg-black/5",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md px-3 py-1.5 text-sm font-medium disabled:opacity-40 ${styles}`}
    >
      {children}
    </button>
  );
}

export function Spinner() {
  return <div className="animate-pulse text-sm text-ink/40">Loading…</div>;
}
