import type {
  BrowseResult,
  DoctorReport,
  RunDetail,
  RunSummary,
  SyncResults,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  init?: RequestInit & { json?: unknown },
): Promise<T> {
  const { json, ...rest } = init ?? {};
  const res = await fetch(path, {
    ...rest,
    headers: {
      Accept: "application/json",
      ...(json !== undefined ? { "Content-Type": "application/json" } : {}),
      ...rest.headers,
    },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });

  if (!res.ok) {
    let detail: string = res.statusText;
    try {
      const body = await res.json();
      // FastAPI validation errors arrive as a list of objects.
      detail = Array.isArray(body?.detail)
        ? body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ")
        : (body?.detail ?? detail);
    } catch {
      // not JSON; keep the status text
    }
    throw new ApiError(detail, res.status);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  doctor: () => request<DoctorReport>("/api/doctor"),

  listRuns: () => request<RunSummary[]>("/api/runs"),
  createRun: (name: string) =>
    request<RunSummary>("/api/runs", { method: "POST", json: { name } }),
  getRun: (runId: string) => request<RunDetail>(`/api/runs/${runId}`),
  deleteRun: (runId: string) =>
    request<void>(`/api/runs/${runId}`, { method: "DELETE" }),

  addClip: (
    runId: string,
    body: {
      source_path: string;
      role: string;
      camera_group: string;
      segment_id: string;
      segment_kind?: "rig" | "single" | "independent";
      kind?: "video" | "photos";
    },
  ) => request<unknown>(`/api/runs/${runId}/clips`, { method: "POST", json: body }),
  updateClip: (
    runId: string,
    clipId: string,
    body: Partial<{
      role: string;
      camera_group: string;
      segment_id: string;
      time_offset_s: number;
      enabled: boolean;
    }>,
  ) =>
    request<unknown>(`/api/runs/${runId}/clips/${clipId}`, {
      method: "PATCH",
      json: body,
    }),
  deleteClip: (runId: string, clipId: string) =>
    request<void>(`/api/runs/${runId}/clips/${clipId}`, { method: "DELETE" }),
  posterUrl: (runId: string, clipId: string) =>
    `/api/runs/${runId}/clips/${clipId}/poster`,

  autoSync: (runId: string, segmentId: string) =>
    request<SyncResults>(`/api/runs/${runId}/sync/auto`, {
      method: "POST",
      json: { segment_id: segmentId },
    }),

  browse: (path?: string) =>
    request<BrowseResult>(
      `/api/fs/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`,
    ),

  runStage: (runId: string, stageId: string) =>
    request<unknown>(`/api/runs/${runId}/stages/${stageId}/run`, { method: "POST" }),
  cancelStage: (runId: string, stageId: string) =>
    request<unknown>(`/api/runs/${runId}/stages/${stageId}/cancel`, { method: "POST" }),
};
