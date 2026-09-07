import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { RunSummary } from "../api/types";

const STAGE_ORDER = [
  "extract",
  "select",
  "mask",
  "sparse",
  "dense",
  "mesh",
  "export",
];

const STATE_CLASS: Record<string, string> = {
  done: "ok",
  running: "accent",
  stale: "warn",
  failed: "bad",
};

function StageStrip({ run }: { run: RunSummary }) {
  return (
    <div className="stage-strip">
      {STAGE_ORDER.map((id) => {
        const stage = run.stages?.[id];
        const cls = stage ? (STATE_CLASS[stage.state] ?? "") : "";
        return (
          <span
            key={id}
            className={`pip ${cls}`}
            title={`${id}: ${stage?.state ?? "unknown"}`}
          />
        );
      })}
    </div>
  );
}

export default function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();

  const load = useCallback(async () => {
    try {
      setRuns(await api.listRuns());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function create(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const run = await api.createRun(name || "bust");
      navigate(`/runs/${run.run_id}/clips`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function remove(runId: string) {
    if (!confirm(`Delete run ${runId} and all of its data?`)) return;
    await api.deleteRun(runId);
    void load();
  }

  return (
    <div className="main">
      <div className="page-header">
        <h2>Runs</h2>
        <p>Each run holds one capture session and everything derived from it.</p>
      </div>

      <form className="panel inline-form" onSubmit={create}>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="New run name, e.g. bust-01"
        />
        <button type="submit" className="primary" disabled={busy}>
          {busy ? "Creating…" : "New run"}
        </button>
      </form>

      {error && <div className="banner bad">{error}</div>}

      {runs === null ? (
        <div className="loading">Loading…</div>
      ) : runs.length === 0 ? (
        <div className="stub-note">
          No runs yet. Create one above, then add your camera clips to it.
        </div>
      ) : (
        <div className="panel">
          <div className="panel-body">
            {runs.map((run) => (
              <div className="run-row" key={run.run_id}>
                <div>
                  <Link to={`/runs/${run.run_id}/clips`} className="run-name">
                    {run.name || run.run_id}
                  </Link>
                  <div className="muted mono-sm">{run.run_id}</div>
                </div>
                <div className="muted">
                  {run.clip_count} clip{run.clip_count === 1 ? "" : "s"}
                  {run.total_slots > 0 && ` · ${run.total_slots} slots`}
                </div>
                <StageStrip run={run} />
                <button className="refresh" onClick={() => void remove(run.run_id)}>
                  Delete
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
