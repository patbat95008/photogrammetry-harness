import { useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import StageShell from "./StageShell";

interface FrameRecord {
  slot: number;
  camera_group: string;
  timeline_time_s: number;
  sync_residual_s: number;
  sync_suspect: boolean;
  duplicate_of: number | null;
  mean_luma: number;
  w: number;
  h: number;
}

/**
 * The contact sheet, grouped by camera.
 *
 * Laid out as one column per slot so the two cameras line up vertically: the
 * whole point of the slot naming is that a column is a single instant seen from
 * two heights, and that is only obvious if you can see it.
 */
function FrameGrid({ runId }: { runId: string }) {
  const [frames, setFrames] = useState<FrameRecord[] | null>(null);
  const [groups, setGroups] = useState<string[]>([]);
  const [showFull, setShowFull] = useState<{ group: string; slot: number } | null>(null);

  const load = useCallback(async () => {
    const res = await fetch(`/api/runs/${runId}/frames?limit=4000`);
    if (!res.ok) return;
    const data = await res.json();
    setFrames(data.frames);
    setGroups(data.groups);
  }, [runId]);

  useEffect(() => {
    void load();
  }, [load]);

  if (!frames || frames.length === 0) return null;

  const slots = [...new Set(frames.map((f) => f.slot))].sort((a, b) => a - b);
  const byGroupSlot = new Map<string, FrameRecord>();
  frames.forEach((f) => byGroupSlot.set(`${f.camera_group}/${f.slot}`, f));

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Frames</h3>
        <span className="muted" style={{ fontSize: 12 }}>
          each column is one instant · {slots.length} slots × {groups.length} cameras
        </span>
      </div>

      <div className="sheet-scroll">
        {groups.map((group) => (
          <div className="sheet-row" key={group}>
            <div className="sheet-label mono-sm">{group}</div>
            <div className="sheet-strip">
              {slots.map((slot) => {
                const record = byGroupSlot.get(`${group}/${slot}`);
                if (!record) return <div className="sheet-gap" key={slot} />;
                return (
                  <button
                    key={slot}
                    className={`sheet-cell ${record.duplicate_of !== null ? "dup" : ""} ${
                      record.sync_suspect ? "suspect" : ""
                    }`}
                    title={`slot ${slot} · t=${record.timeline_time_s.toFixed(3)}s${
                      record.duplicate_of !== null ? " · duplicate" : ""
                    }${record.sync_suspect ? " · sync suspect" : ""}`}
                    onClick={() => setShowFull({ group, slot })}
                  >
                    <img
                      src={`/api/runs/${runId}/frames/${group}/${slot}?size=thumb`}
                      loading="lazy"
                      alt={`${group} slot ${slot}`}
                    />
                    <span className="sheet-slot">{slot}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {showFull && (
        <div className="modal-backdrop" onClick={() => setShowFull(null)}>
          <div className="modal wide" onClick={(e) => e.stopPropagation()}>
            <div className="panel-head">
              <h3>
                {showFull.group} · slot {showFull.slot}
              </h3>
              <button className="refresh" onClick={() => setShowFull(null)}>
                Close
              </button>
            </div>
            <div className="compare">
              {groups.map((group) => (
                <figure key={group}>
                  <img
                    src={`/api/runs/${runId}/frames/${group}/${showFull.slot}?size=full`}
                    alt={`${group} slot ${showFull.slot}`}
                  />
                  <figcaption className="mono-sm muted">{group}</figcaption>
                </figure>
              ))}
            </div>
            <div className="muted" style={{ padding: "0 16px 14px", fontSize: 12.5 }}>
              Both images are slot {showFull.slot} — the same instant, from each camera.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ExtractPage() {
  const { runId = "" } = useParams();
  return (
    <StageShell runId={runId} stageId="extract">
      {(detail) =>
        detail.state === "done" || detail.state === "stale" ? (
          <FrameGrid runId={runId} />
        ) : null
      }
    </StageShell>
  );
}
