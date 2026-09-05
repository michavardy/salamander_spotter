"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { loadRuntimeConfig } from "@/lib/api";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { staleTime: 10_000, refetchOnWindowFocus: false } },
      }),
  );
  const [ready, setReady] = useState(false);

  useEffect(() => {
    loadRuntimeConfig().finally(() => setReady(true));
  }, []);

  if (!ready) return null;
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
