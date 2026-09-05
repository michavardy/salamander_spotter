"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";

export const useDashboard = () =>
  useQuery({ queryKey: ["dashboard"], queryFn: () => api<any>("/dashboard") });

export const useIndividuals = (q?: string) =>
  useQuery({ queryKey: ["individuals", q ?? ""], queryFn: () => api<any>(`/individuals${q ? `?q=${encodeURIComponent(q)}` : ""}`) });

export const useIndividual = (id: string | null) =>
  useQuery({ enabled: !!id, queryKey: ["individual", id], queryFn: () => api<any>(`/individuals/${id}`) });

export const useImage = (id: string | null) =>
  useQuery({ enabled: !!id, queryKey: ["image", id], queryFn: () => api<any>(`/images/${id}`) });

export const useReviewQueue = () =>
  useQuery({ queryKey: ["review"], queryFn: () => api<any>("/review"), refetchInterval: 15_000 });

export const useReviewDetail = (id: string | null) =>
  useQuery({ enabled: !!id, queryKey: ["review", id], queryFn: () => api<any>(`/review/${id}`) });

export const useModels = () =>
  useQuery({ queryKey: ["models"], queryFn: () => api<any>("/models") });

const JOB_ACTIVE_STATUSES = new Set(["queued", "running"]);

/** Poll a background job (spec §5.3) while it's active; stop once it settles. */
export const useJob = (jobId: string | null) =>
  useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api<any>(`/jobs/${jobId}`),
    enabled: !!jobId,
    refetchInterval: (query) => (JOB_ACTIVE_STATUSES.has(query.state.data?.status) ? 3000 : false),
  });

/** Poll a job's on-disk log while it's active. */
export const useJobLog = (jobId: string | null, active: boolean) =>
  useQuery({
    queryKey: ["jobLog", jobId],
    queryFn: () => api<{ text: string }>(`/jobs/${jobId}/log`),
    enabled: !!jobId,
    refetchInterval: active ? 3000 : false,
  });

export const useSettings = () =>
  useQuery({ queryKey: ["settings"], queryFn: () => api<any>("/settings") });

export const usePaths = () =>
  useQuery({ queryKey: ["paths"], queryFn: () => api<any>("/settings/paths") });

export const useBatches = () =>
  useQuery({ queryKey: ["batches"], queryFn: () => api<any>("/batches") });

export function useInvalidate() {
  const qc = useQueryClient();
  return (...keys: string[]) => keys.forEach((k) => qc.invalidateQueries({ queryKey: [k] }));
}

export function useRunMatch() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (imageId: string) => api<any>(`/review/${imageId}/match`, { json: {} }),
    onSuccess: () => inv("review"),
  });
}

export function useDecide() {
  const inv = useInvalidate();
  return useMutation({
    mutationFn: (p: { imageId: string; body: any }) => api<any>(`/review/${p.imageId}/decision`, { json: p.body }),
    onSuccess: () => inv("review", "dashboard", "individuals"),
  });
}
