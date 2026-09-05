"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { Bell, Home, Upload, ListChecks, Tags, Boxes, Settings } from "lucide-react";
import { api } from "@/lib/api";
import { TrainingToast } from "@/components/TrainingToast";

const NAV = [
  { href: "/", label: "Overview", icon: Home },
  { href: "/upload", label: "Upload", icon: Upload },
  { href: "/review", label: "Review", icon: ListChecks },
  { href: "/names", label: "Names", icon: Tags },
  { href: "/models", label: "Models", icon: Boxes },
  { href: "/models/settings", label: "Settings", icon: Settings },
];

export function Chrome({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { data } = useQuery({
    queryKey: ["notifications"],
    queryFn: () => api<{ unread: number }>("/notifications"),
    refetchInterval: 20_000,
  });

  return (
    <div className="flex min-h-screen">
      <aside className="w-[220px] shrink-0 bg-sidebar text-white/90 flex flex-col">
        <div className="px-5 pt-5 pb-1">
          <div className="flex h-9 items-center justify-center overflow-hidden">
            <img
              src="/logo.png"
              alt="Salamander Spotter logo"
              className="rotate-90 object-contain"
              style={{ height: 128, width: 36 }}
            />
          </div>
        </div>
        <div className="px-5 pb-5 font-head text-lg font-semibold tracking-tight">
          Salamander<span className="text-accent"> Spotter</span>
        </div>
        <nav className="flex-1 px-2 space-y-1">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm ${
                  active ? "bg-white/10 text-white" : "hover:bg-white/5"
                }`}
              >
                <Icon size={17} strokeWidth={1.7} />
                {label}
              </Link>
            );
          })}
        </nav>
        <div className="px-5 py-4 text-xs text-white/40 font-mono">v0.1.0</div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col">
        <header className="flex items-center justify-between border-b border-black/5 bg-panel px-6 py-3">
          <div className="font-head text-sm text-ink/60">{titleFor(pathname)}</div>
          <button className="relative rounded-md p-2 hover:bg-black/5" aria-label="Notifications">
            <Bell size={18} strokeWidth={1.7} />
            {data && data.unread > 0 && (
              <span className="absolute -right-0.5 -top-0.5 grid h-4 min-w-4 place-items-center rounded-full bg-danger px-1 text-[10px] font-semibold text-white">
                {data.unread}
              </span>
            )}
          </button>
        </header>
        <main className="flex-1 p-6">{children}</main>
      </div>
      <TrainingToast />
    </div>
  );
}

function titleFor(path: string): string {
  if (path === "/") return "Overview";
  const seg = path.split("/").filter(Boolean);
  return seg.map((s) => s[0].toUpperCase() + s.slice(1)).join(" › ");
}
