import { useEffect, useRef, useState } from "react";

export interface StageEvent {
  seq: number;
  ts: string;
  type: string;
  stage?: string;
  fraction?: number | null;
  message?: string;
  current?: number | null;
  total?: number | null;
  line?: string;
  level?: string;
  error?: string;
  duration_s?: number;
  warnings?: string[];
  metrics?: Record<string, unknown>;
}

export interface StreamState {
  connected: boolean;
  progress: { fraction: number | null; message: string } | null;
  log: string[];
  lastLifecycle: StageEvent | null;
}

const MAX_LOG_LINES = 500;

/**
 * Subscribes to a run's server-sent event stream.
 *
 * EventSource reconnects on its own and replays via Last-Event-ID, so refreshing
 * the page mid-job picks up where it left off rather than showing a dead progress
 * bar. Log lines are live-only by design; history comes from the log file endpoint.
 */
export function useEventStream(runId: string | undefined, stageId?: string): StreamState {
  const [state, setState] = useState<StreamState>({
    connected: false,
    progress: null,
    log: [],
    lastLifecycle: null,
  });
  const logRef = useRef<string[]>([]);

  useEffect(() => {
    if (!runId) return;

    const source = new EventSource(`/api/runs/${runId}/events`);
    const onOpen = () => setState((s) => ({ ...s, connected: true }));
    const onError = () => setState((s) => ({ ...s, connected: false }));

    function handle(event: MessageEvent) {
      let data: StageEvent;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }
      if (stageId && data.stage && data.stage !== stageId) return;

      if (data.type === "stage.log" && data.line) {
        logRef.current = [...logRef.current, data.line].slice(-MAX_LOG_LINES);
        setState((s) => ({ ...s, log: logRef.current }));
        return;
      }
      if (data.type === "stage.progress") {
        setState((s) => ({
          ...s,
          progress: { fraction: data.fraction ?? null, message: data.message ?? "" },
        }));
        return;
      }
      if (data.type.startsWith("stage.") || data.type.startsWith("sync.")) {
        setState((s) => ({
          ...s,
          lastLifecycle: data,
          progress: data.type === "stage.started" ? { fraction: 0, message: "" } : null,
        }));
      }
    }

    // Named SSE events do not fire the generic "message" handler, so each type has
    // to be registered explicitly.
    const types = [
      "stage.queued",
      "stage.started",
      "stage.progress",
      "stage.log",
      "stage.finished",
      "stage.failed",
      "stage.cancelled",
      "stage.params",
      "sync.updated",
      "clip.added",
      "clip.updated",
      "clip.removed",
    ];
    types.forEach((t) => source.addEventListener(t, handle as EventListener));
    source.addEventListener("open", onOpen);
    source.addEventListener("error", onError);

    return () => {
      types.forEach((t) => source.removeEventListener(t, handle as EventListener));
      source.removeEventListener("open", onOpen);
      source.removeEventListener("error", onError);
      source.close();
    };
  }, [runId, stageId]);

  return state;
}
