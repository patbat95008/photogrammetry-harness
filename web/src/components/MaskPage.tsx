import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import StageShell, { StageDetail } from "./StageShell";

/**
 * Masking: click the subject, look at what came back.
 *
 * Two halves, and the order matters. The prompt editor renders whatever state the
 * stage is in -- unlike every other stage page, whose results only exist after a run,
 * the clicks here are the stage's *input* and have to be placeable before the first
 * run. The review strip appears once there is something to review.
 *
 * The preview is what makes the parameters mean anything. A single click on the cup
 * landed on one of the photographs printed on it rather than on the cup, and SAM 2
 * ranked the actual mug second -- which was invisible until a full 241-frame
 * propagation had finished. One click, one round trip, and you can see it.
 */

interface Prompt {
  x: number;
  y: number;
  include: boolean;
}

interface MaskRow {
  slot: number;
  camera_group: string;
  overlay: string;
  area_fraction: number;
  empty: boolean;
  touches_border: string[];
  iou_prev: number | null;
}

type View = "overlay" | "cutout" | "mask";

function key(group: string, slot: number) {
  return `${group}/${slot}`;
}

/** Click the subject on one frame. Coordinates are normalised, as the params are. */
function PromptEditor({
  runId,
  group,
  slot,
  prompts,
  onChange,
  busy,
  params,
}: {
  runId: string;
  group: string;
  slot: number;
  prompts: Prompt[];
  onChange: (next: Prompt[]) => void;
  busy: boolean;
  params: Record<string, unknown>;
}) {
  const [exclude, setExclude] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [area, setArea] = useState<number | null>(null);
  const [checking, setChecking] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const url = `/api/runs/${runId}/artifacts/frames/${group}/${String(slot).padStart(6, "0")}.jpg`;

  async function check() {
    setChecking(true);
    setPreviewError(null);
    try {
      const res = await fetch(`/api/runs/${runId}/mask/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_group: group, slot, prompts, params }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      setArea(Number(res.headers.get("X-Mask-Area") ?? 0));
      const blob = await res.blob();
      setPreview((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return URL.createObjectURL(blob);
      });
    } catch (err) {
      setPreviewError(err instanceof Error ? err.message : String(err));
    } finally {
      setChecking(false);
    }
  }

  function place(event: React.MouseEvent<HTMLImageElement>) {
    if (busy) return;
    const rect = event.currentTarget.getBoundingClientRect();
    onChange([
      ...prompts,
      {
        x: (event.clientX - rect.left) / rect.width,
        y: (event.clientY - rect.top) / rect.height,
        include: !exclude,
      },
    ]);
  }

  return (
    <div className="mask-editor">
      <div className="mask-frame">
        <img src={url} alt={`${group} slot ${slot}`} onClick={place} />
        {prompts.map((p, index) => (
          <button
            key={index}
            type="button"
            className={`mask-dot ${p.include ? "include" : "exclude"}`}
            style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }}
            title={
              p.include
                ? "On the subject. Click to remove."
                : "Off the subject. Click to remove."
            }
            onClick={() => onChange(prompts.filter((_, i) => i !== index))}
          />
        ))}
      </div>
      <div className="mask-controls">
        <strong>{group}</strong>
        <label>
          <input
            type="checkbox"
            checked={exclude}
            onChange={(e) => setExclude(e.target.checked)}
          />
          Next click marks something to leave out
        </label>
        <span className="muted">
          {prompts.length === 0
            ? "Click the subject."
            : `${prompts.filter((p) => p.include).length} on, ${
                prompts.filter((p) => !p.include).length
              } off — click a dot to remove it`}
        </span>
        <button
          className="refresh"
          disabled={busy || checking || prompts.length === 0}
          onClick={() => void check()}
        >
          {checking ? "Segmenting…" : "Check this frame"}
        </button>
        {previewError && <span className="cloud-warn">{previewError}</span>}
        {area !== null && (
          <span className="muted">
            covers {(area * 100).toFixed(1)}% of the frame
            {area > 0.75 && " — that is most of the picture, so the clicks probably " +
              "landed on the room rather than the subject"}
          </span>
        )}
      </div>
      {preview && (
        <div className="mask-frame">
          <img src={url} alt="" />
          <img className="mask-preview-overlay" src={preview} alt="preview mask" />
        </div>
      )}
    </div>
  );
}

/**
 * One canvas, three views: the tinted overlay, the subject cut out, and the mask
 * alone. A canvas rather than stacked images with mix-blend-mode, because the cutout
 * needs real per-pixel alpha and blend modes render differently between browsers.
 */
function MaskCanvas({
  frameUrl,
  maskUrl,
  view,
  opacity,
}: {
  frameUrl: string;
  maskUrl: string;
  view: View;
  opacity: number;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setError(false);

    async function draw() {
      const load = (src: string) =>
        new Promise<HTMLImageElement>((resolve, reject) => {
          const image = new Image();
          image.onload = () => resolve(image);
          image.onerror = reject;
          image.src = src;
        });

      try {
        const [frame, mask] = await Promise.all([load(frameUrl), load(maskUrl)]);
        if (cancelled) return;
        const canvas = ref.current;
        if (!canvas) return;

        const width = 420;
        const height = Math.round((frame.naturalHeight / frame.naturalWidth) * width);
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.clearRect(0, 0, width, height);
        if (view !== "mask") ctx.drawImage(frame, 0, 0, width, height);

        const scratch = document.createElement("canvas");
        scratch.width = width;
        scratch.height = height;
        const sctx = scratch.getContext("2d");
        if (!sctx) return;
        // Nearest neighbour: the mask is binary, and smoothing it invents the
        // partial values the whole pipeline is careful not to produce.
        sctx.imageSmoothingEnabled = false;
        sctx.drawImage(mask, 0, 0, width, height);
        const maskData = sctx.getImageData(0, 0, width, height).data;

        const base = ctx.getImageData(0, 0, width, height);
        for (let i = 0; i < maskData.length; i += 4) {
          const keep = maskData[i] > 127;
          if (view === "mask") {
            const value = keep ? 255 : 30;
            base.data[i] = base.data[i + 1] = base.data[i + 2] = value;
            base.data[i + 3] = 255;
          } else if (view === "cutout") {
            base.data[i + 3] = keep ? 255 : 0;
          } else if (keep) {
            base.data[i + 1] = base.data[i + 1] * (1 - opacity) + 255 * opacity;
            base.data[i] = base.data[i] * (1 - opacity);
          }
        }
        ctx.putImageData(base, 0, 0);
      } catch {
        if (!cancelled) setError(true);
      }
    }

    void draw();
    return () => {
      cancelled = true;
    };
  }, [frameUrl, maskUrl, view, opacity]);

  if (error) return <div className="cloud-status">could not load that frame</div>;
  return <canvas ref={ref} className={`mask-canvas ${view}`} />;
}

export default function MaskPage() {
  const { runId = "" } = useParams();
  const [rows, setRows] = useState<MaskRow[]>([]);
  const [selected, setSelected] = useState<MaskRow | null>(null);
  const [view, setView] = useState<View>("overlay");
  const [opacity, setOpacity] = useState(0.45);
  const [problemsOnly, setProblemsOnly] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const loadRows = useCallback(
    async (state: string) => {
      if (state !== "done" && state !== "stale") {
        setRows([]);
        return;
      }
      const res = await fetch(`/api/runs/${runId}/artifacts/mask/masks.jsonl`);
      if (!res.ok) return;
      const text = await res.text();
      setRows(
        text
          .split("\n")
          .filter((line) => line.trim())
          .map((line) => JSON.parse(line) as MaskRow),
      );
    },
    [runId],
  );

  return (
    <StageShell runId={runId} stageId="mask">
      {(detail: StageDetail, reload: () => Promise<void>) => {
        // eslint-disable-next-line react-hooks/rules-of-hooks
        return (
          <MaskBody
            detail={detail}
            reload={reload}
            runId={runId}
            rows={rows}
            loadRows={loadRows}
            selected={selected}
            setSelected={setSelected}
            view={view}
            setView={setView}
            opacity={opacity}
            setOpacity={setOpacity}
            problemsOnly={problemsOnly}
            setProblemsOnly={setProblemsOnly}
            saving={saving}
            setSaving={setSaving}
            saveError={saveError}
            setSaveError={setSaveError}
          />
        );
      }}
    </StageShell>
  );
}

function MaskBody({
  detail,
  reload,
  runId,
  rows,
  loadRows,
  selected,
  setSelected,
  view,
  setView,
  opacity,
  setOpacity,
  problemsOnly,
  setProblemsOnly,
  saving,
  setSaving,
  saveError,
  setSaveError,
}: {
  detail: StageDetail;
  reload: () => Promise<void>;
  runId: string;
  rows: MaskRow[];
  loadRows: (state: string) => Promise<void>;
  selected: MaskRow | null;
  setSelected: (row: MaskRow | null) => void;
  view: View;
  setView: (v: View) => void;
  opacity: number;
  setOpacity: (v: number) => void;
  problemsOnly: boolean;
  setProblemsOnly: (v: boolean) => void;
  saving: boolean;
  setSaving: (v: boolean) => void;
  saveError: string | null;
  setSaveError: (v: string | null) => void;
}) {
  const params = (detail.record.params ?? {}) as Record<string, unknown>;
  const prompts = (params.prompts ?? {}) as Record<string, Prompt[]>;
  const groups = useMemo(() => {
    const fromRows = Array.from(new Set(rows.map((r) => r.camera_group)));
    if (fromRows.length) return fromRows.sort();
    const fromPrompts = Object.keys(prompts).map((k) => k.split("/")[0]);
    return Array.from(new Set(fromPrompts)).sort();
  }, [rows, prompts]);

  useEffect(() => {
    void loadRows(detail.state);
    // Re-read after a run finishes; the sidecar is rewritten wholesale.
  }, [detail.state, detail.record.duration_s, loadRows]);

  const threshold = Number(detail.record.metrics?.drift_threshold ?? 0);
  const problems = rows.filter(
    (r) =>
      r.empty ||
      r.touches_border.length > 0 ||
      (threshold > 0 && r.iou_prev !== null && r.iou_prev < threshold),
  );
  const shown = problemsOnly ? problems : rows;

  async function savePrompts(next: Record<string, Prompt[]>) {
    setSaving(true);
    setSaveError(null);
    try {
      const res = await fetch(`/api/runs/${runId}/stages/mask/params`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...params, prompts: next }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      await reload();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  // Which frame each group is prompted on. The first chunk has to be conditioned,
  // so this defaults to the earliest frame rather than to whatever was clicked last.
  const editing: Record<string, number> = {};
  for (const group of groups) {
    const slots = Object.keys(prompts)
      .filter((k) => k.startsWith(`${group}/`))
      .map((k) => Number(k.split("/")[1]))
      .filter((n) => Number.isFinite(n));
    editing[group] = slots.length ? Math.min(...slots) : 0;
  }

  return (
    <div className="stage-results">
      <h3>Click the subject</h3>
      <p className="stage-note">
        One click per camera is usually enough, but check it before running: SAM 2
        picks the most confident object under the point, which is not always the one
        you meant. On the test cup, a single click chose one of the photographs
        printed on the mug and the mug itself came second. Add points until the
        highlight covers the whole subject.
      </p>

      {saveError && <div className="banner bad">{saveError}</div>}

      {groups.length === 0 && (
        <p className="stage-note cloud-warn">
          No camera groups yet — run the extract stage first.
        </p>
      )}

      {groups.map((group) => {
        const slot = editing[group];
        const at = key(group, slot);
        return (
          <PromptEditor
            key={group}
            runId={runId}
            group={group}
            slot={slot}
            prompts={prompts[at] ?? []}
            busy={saving}
            params={params}
            onChange={(next) => {
              const merged = { ...prompts };
              if (next.length) merged[at] = next;
              else delete merged[at];
              void savePrompts(merged);
            }}
          />
        );
      })}

      {rows.length > 0 && (
        <>
          <h3>What came back</h3>
          <div className="mask-review-controls">
            <label>
              View{" "}
              <select value={view} onChange={(e) => setView(e.target.value as View)}>
                <option value="overlay">Overlay</option>
                <option value="cutout">Cutout</option>
                <option value="mask">Mask only</option>
              </select>
            </label>
            {view === "overlay" && (
              <label>
                Tint{" "}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                />
              </label>
            )}
            <label>
              <input
                type="checkbox"
                checked={problemsOnly}
                onChange={(e) => setProblemsOnly(e.target.checked)}
              />
              Only the {problems.length} worth looking at
            </label>
          </div>

          {selected && (
            <div className="mask-detail">
              <MaskCanvas
                frameUrl={`/api/runs/${runId}/artifacts/frames/${selected.camera_group}/${String(
                  selected.slot,
                ).padStart(6, "0")}.jpg`}
                maskUrl={`/api/runs/${runId}/artifacts/mask/canonical/${selected.camera_group}/${String(
                  selected.slot,
                ).padStart(6, "0")}.png`}
                view={view}
                opacity={opacity}
              />
              <div>
                <strong>
                  {selected.camera_group}/{String(selected.slot).padStart(6, "0")}
                </strong>
                <p className="muted">
                  covers {(selected.area_fraction * 100).toFixed(1)}% of the frame
                  {selected.iou_prev !== null &&
                    `, agrees ${(selected.iou_prev * 100).toFixed(0)}% with the frame before`}
                  {selected.touches_border.length > 0 &&
                    `, runs off the ${selected.touches_border.join(" and ")} edge`}
                </p>
                <p className="stage-note">
                  Re-prompting fixes this frame and the ones after it in the same
                  chunk. Frames before it keep the track they already had, so if the
                  mask went wrong earlier than this, click the earliest frame that
                  looks wrong rather than the worst one.
                </p>
                <button
                  className="refresh"
                  disabled={saving}
                  onClick={() => {
                    const at = key(selected.camera_group, selected.slot);
                    if (!prompts[at]) void savePrompts({ ...prompts, [at]: [] });
                  }}
                >
                  Re-prompt this frame
                </button>
              </div>
            </div>
          )}

          <div className="mask-strip">
            {shown.map((row) => {
              const bad = row.empty;
              const iffy =
                !bad &&
                (row.touches_border.length > 0 ||
                  (threshold > 0 && row.iou_prev !== null && row.iou_prev < threshold));
              return (
                <button
                  key={`${row.camera_group}/${row.slot}`}
                  type="button"
                  className={`mask-cell ${bad ? "bad" : iffy ? "iffy" : ""} ${
                    selected?.slot === row.slot &&
                    selected?.camera_group === row.camera_group
                      ? "on"
                      : ""
                  }`}
                  title={`${row.camera_group}/${row.slot} — ${(
                    row.area_fraction * 100
                  ).toFixed(1)}%`}
                  onClick={() => setSelected(row)}
                >
                  <img
                    loading="lazy"
                    src={`/api/runs/${runId}/artifacts/${row.overlay}`}
                    alt={`${row.camera_group} ${row.slot}`}
                  />
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
