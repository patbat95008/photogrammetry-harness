import { useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";

interface PlannedStage {
  label: string;
  summary: string;
  plan: string[];
  note: string | null;
}

interface StubDetail {
  stage_id: string;
  state: string;
  blocked_by: string[];
  skippable?: boolean;
  planned: PlannedStage | null;
}

const STAGE_TITLE: Record<string, string> = {
  select: "Select frames",
  mask: "Mask",
  sparse: "Align",
  dense: "Dense cloud",
  mesh: "Mesh",
  export: "Export",
};

/**
 * A stage that has not been built yet.
 *
 * Shows what the stage will actually do and what it is waiting on, rather than a
 * bare "not implemented" — the point is that the shape of the pipeline stays
 * legible while most of it is still empty. Notes marked as capture-dependent come
 * from the server, because they change with how the run was shot.
 */
export default function StageStub({ stageId }: { stageId: string }) {
  const { runId = "" } = useParams();
  const [detail, setDetail] = useState<StubDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const res = await fetch(`/api/runs/${runId}/stages/${stageId}`);
    if (res.ok) setDetail(await res.json());
  }, [runId, stageId]);

  useEffect(() => {
    void load();
  }, [load]);

  const skipped = detail?.state === "skipped";

  async function toggleSkip() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/runs/${runId}/stages/${stageId}/${skipped ? "unskip" : "skip"}`,
        { method: "POST" },
      );
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const planned = detail?.planned;

  return (
    <div className="main">
      <div className="page-header">
        <h2>{planned?.label ?? STAGE_TITLE[stageId] ?? stageId}</h2>
        <p>{planned?.summary ?? "This stage is not wired up yet."}</p>
      </div>

      {error && <div className="banner bad">{error}</div>}

      {skipped ? (
        <div className="banner">
          <span className="dot ok">●</span>
          <div>
            <strong>Skipped</strong>
            <span className="muted">
              Later stages treat this as satisfied and will run without it.
            </span>
          </div>
          <button className="refresh" disabled={busy} onClick={toggleSkip}>
            Un-skip
          </button>
        </div>
      ) : (
        <div className="banner">
          <span className="dot warn">●</span>
          <div>
            <strong>Not built yet</strong>
            <span className="muted">
              {detail?.blocked_by.length
                ? `It will also need these to finish first: ${detail.blocked_by.join(", ")}.`
                : "Its inputs are ready; only the stage itself is missing."}
            </span>
          </div>
          {detail?.skippable && (
            <button className="refresh" disabled={busy} onClick={toggleSkip}>
              Skip this stage
            </button>
          )}
        </div>
      )}

      {planned?.note && (
        <div className="banner note">
          <span className="dot accent-dot">●</span>
          <div>
            <strong>Because of how this run was captured</strong>
            {planned.note}
          </div>
        </div>
      )}

      {planned && planned.plan.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <h3>What this stage will do</h3>
          </div>
          <ol className="plan-list">
            {planned.plan.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
