import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import StageShell, { StageDetail } from "./StageShell";

interface Decision {
  slot: number;
  camera_group: string;
  file: string;
  sharpness: number;
  sharpness_percentile: number;
  selected: boolean;
  reason: string;
  overridden: boolean;
}

type Override = "keep" | "reject";

const REASON_LABEL: Record<string, string> = {
  duplicate: "near-identical to the previous frame",
  blurry: "below the sharpness floor",
  coverage: "thinned out to keep coverage even",
  manual: "rejected by hand",
  unreadable: "could not be read from disk",
};

/**
 * The contact sheet, with a reason attached to every rejection.
 *
 * A selection you cannot inspect is a selection you have to trust. Showing why each
 * frame was dropped is what makes the parameters meaningful -- "43 rejected" says
 * nothing, but a strip of visibly soft frames tinted as blurry says whether the
 * floor is set sensibly.
 *
 * Clicking a frame overrides the automatic decision. Overrides live in the stage's
 * params, so they change its fingerprint and mark it out of date, exactly as moving
 * a slider would.
 */
function ContactSheet({
  runId,
  detail,
  reload,
}: {
  runId: string;
  detail: StageDetail;
  reload: () => Promise<void>;
}) {
  const [rows, setRows] = useState<Decision[] | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "selected" | "rejected">("all");

  const overrides = (detail.record.params.overrides ?? {}) as Record<string, Override>;

  useEffect(() => {
    fetch(`/api/runs/${runId}/artifacts/select/selection.jsonl`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then((text) =>
        setRows(
          text
            .split("\n")
            .filter((line) => line.trim())
            .map((line) => JSON.parse(line) as Decision),
        ),
      )
      .catch(() => setRows([]));
  }, [runId, detail.record.params]);

  const setOverride = useCallback(
    async (key: string, next: Override | null) => {
      setSaving(true);
      setError(null);
      const merged = { ...overrides };
      if (next === null) delete merged[key];
      else merged[key] = next;
      try {
        const res = await fetch(`/api/runs/${runId}/stages/select/params`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...detail.record.params, overrides: merged }),
        });
        if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
        await reload();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setSaving(false);
      }
    },
    [runId, detail.record.params, overrides, reload],
  );

  const groups = useMemo(
    () => [...new Set((rows ?? []).map((r) => r.camera_group))],
    [rows],
  );

  if (!rows || rows.length === 0) return null;

  const overrideCount = Object.keys(overrides).length;

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Selection</h3>
        <span className="muted" style={{ fontSize: 12 }}>
          click a frame to keep or reject it by hand
          {overrideCount > 0 && ` · ${overrideCount} override${overrideCount > 1 ? "s" : ""}`}
        </span>
      </div>

      {error && <div className="banner bad">{error}</div>}

      <div className="sheet-filters">
        {(["all", "selected", "rejected"] as const).map((f) => (
          <button
            key={f}
            className={`chip ${filter === f ? "on" : ""}`}
            onClick={() => setFilter(f)}
          >
            {f}
          </button>
        ))}
        <span className="muted" style={{ fontSize: 12 }}>
          {rows.filter((r) => r.selected).length} of {rows.length} kept
        </span>
      </div>

      <div className="sheet-scroll">
        {groups.map((group) => {
          const groupRows = rows
            .filter((r) => r.camera_group === group)
            .filter((r) =>
              filter === "all"
                ? true
                : filter === "selected"
                  ? r.selected
                  : !r.selected,
            );
          return (
            <div className="sheet-row" key={group}>
              <div className="sheet-label mono-sm">{group}</div>
              <div className="sheet-strip">
                {groupRows.map((row) => {
                  const key = `${row.camera_group}/${row.slot}`;
                  const override = overrides[key];
                  const title = [
                    `slot ${row.slot}`,
                    `sharpness ${row.sharpness} (${row.sharpness_percentile}th percentile)`,
                    row.selected ? "kept" : `rejected: ${REASON_LABEL[row.reason] ?? row.reason}`,
                    override ? `overridden to ${override}` : "click to override",
                  ].join(" · ");
                  return (
                    <button
                      key={key}
                      className={`sheet-cell select-cell ${row.selected ? "keep" : "drop"} ${
                        override ? "overridden" : ""
                      }`}
                      title={title}
                      disabled={saving}
                      onClick={() =>
                        setOverride(
                          key,
                          override ? null : row.selected ? "reject" : "keep",
                        )
                      }
                    >
                      <img
                        src={`/api/runs/${runId}/frames/${group}/${row.slot}?size=thumb`}
                        loading="lazy"
                        alt={`slot ${row.slot}`}
                      />
                      <span className="sheet-slot">{row.slot}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      <p className="stage-note">
        Green frames are kept, dimmed ones are dropped, and a marked corner means you
        overrode the decision. Overriding changes this stage's parameters, so it will
        show as out of date until you run it again.
      </p>
    </div>
  );
}

export default function SelectPage() {
  const { runId = "" } = useParams();
  return (
    <StageShell runId={runId} stageId="select">
      {(detail, reload) =>
        detail.state === "done" || detail.state === "stale" ? (
          <ContactSheet runId={runId} detail={detail} reload={reload} />
        ) : null
      }
    </StageShell>
  );
}
