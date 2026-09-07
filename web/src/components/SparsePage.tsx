import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import StageShell, { StageDetail } from "./StageShell";
import PointCloudViewer from "./PointCloudViewer";

interface Registration {
  name: string;
  registered: boolean;
  num_points: number;
}

/**
 * Which images failed to register, and how weakly the rest are tied in.
 *
 * COLMAP drops images it cannot place without saying so in its exit code, so this
 * table is the only place the omission is visible. A view registered with very few
 * points is nearly as bad as one that failed outright -- it is in the model, but
 * barely constrained by it.
 */
function RegistrationTable({ runId }: { runId: string }) {
  const [rows, setRows] = useState<Registration[] | null>(null);

  useEffect(() => {
    fetch(`/api/runs/${runId}/artifacts/sparse/registration.jsonl`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then((text) =>
        setRows(
          text
            .split("\n")
            .filter((line) => line.trim())
            .map((line) => JSON.parse(line) as Registration),
        ),
      )
      .catch(() => setRows([]));
  }, [runId]);

  if (!rows) return null;

  const failed = rows.filter((r) => !r.registered);
  const weak = rows.filter((r) => r.registered && r.num_points < 40);
  if (failed.length === 0 && weak.length === 0) {
    return (
      <p className="stage-note">
        All {rows.length} submitted frames registered, and none is weakly tied to the
        model.
      </p>
    );
  }

  return (
    <div className="registration-panel">
      {failed.length > 0 && (
        <details open>
          <summary>
            {failed.length} of {rows.length} frames did not register
          </summary>
          <ul className="registration-list">
            {failed.slice(0, 60).map((r) => (
              <li key={r.name}>{r.name}</li>
            ))}
            {failed.length > 60 && <li>…and {failed.length - 60} more</li>}
          </ul>
        </details>
      )}
      {weak.length > 0 && (
        <details>
          <summary>{weak.length} frames registered with fewer than 40 points</summary>
          <ul className="registration-list">
            {weak.slice(0, 60).map((r) => (
              <li key={r.name}>
                {r.name} <span className="muted">({r.num_points} points)</span>
              </li>
            ))}
            {weak.length > 60 && <li>…and {weak.length - 60} more</li>}
          </ul>
        </details>
      )}
    </div>
  );
}

export default function SparsePage() {
  const { runId } = useParams<{ runId: string }>();
  if (!runId) return null;

  return (
    <StageShell runId={runId} stageId="sparse">
      {(detail: StageDetail) =>
        detail.state === "done" || detail.state === "stale" ? (
          <div className="stage-results">
            <h3>Cameras and sparse points</h3>
            <p className="stage-note">
              Each pyramid is where one frame was taken from, pointing the way it
              looked; the line joins them in capture order. A closed orbit reads as a
              closed ring. Cameras scattered rather than arced usually means the solve
              locked onto the background instead of the subject.
            </p>
            <PointCloudViewer
              cloudUrl={`/api/runs/${runId}/artifacts/sparse/preview.ply`}
              posesUrl={`/api/runs/${runId}/artifacts/sparse/poses.json`}
            />
            <RegistrationTable runId={runId} />
          </div>
        ) : null
      }
    </StageShell>
  );
}
