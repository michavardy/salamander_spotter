"use client";

// Shares the "currently active training job id" across the whole app shell (the
// notification-style toast in Chrome.tsx, the Retrain button on the Models page)
// without a new Context — piggybacks on the TanStack Query cache that's already
// provided at the root, backed by localStorage so a page reload during a run that
// can take hours doesn't lose track of it.

import { useQuery, useQueryClient } from "@tanstack/react-query";

const KEY = ["activeTrainingJob"];
const STORAGE_KEY = "spotter_active_training_job";

function readStored(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function useActiveTrainingJob() {
  const qc = useQueryClient();
  const { data } = useQuery<string | null>({
    queryKey: KEY,
    queryFn: () => null,
    initialData: readStored,
    staleTime: Infinity,
    refetchOnMount: false,
  });

  function setJobId(id: string | null) {
    qc.setQueryData(KEY, id);
    try {
      if (id) localStorage.setItem(STORAGE_KEY, id);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      // localStorage unavailable — the in-memory query cache still works for this tab
    }
  }

  return { jobId: data ?? null, setJobId };
}
