import { useEffect, useState } from "react";
import type { RunManifest } from "../api/types";

const MODES = [
  {
    value: "subject_rotates",
    title: "Subject rotates, cameras fixed",
    blurb:
      "You spin in a chair while the cameras stay put. Masking the background becomes mandatory, and cameras on fixed mounts can be treated as one rigid rig.",
  },
  {
    value: "camera_orbits",
    title: "Cameras orbit, subject still",
    blurb:
      "The classic walk-around. The background is rigid with the subject, so it helps rather than hurts and is best left unmasked. Handheld cameras are not a rig.",
  },
] as const;

/**
 * How the capture was performed.
 *
 * This is not bookkeeping: it inverts whether background masking is required or
 * harmful, and decides whether a rigid camera rig can honestly be declared. Later
 * stages read it directly.
 */
export default function CapturePanel({
  runId,
  manifest,
  onChange,
}: {
  runId: string;
  manifest: RunManifest;
  onChange: () => void;
}) {
  const capture = manifest.capture as RunManifest["capture"] & {
    mode?: string;
    revolutions?: number | null;
  };
  const [baseline, setBaseline] = useState(String(capture.baseline_mm ?? ""));
  const [revolutions, setRevolutions] = useState(String(capture.revolutions ?? ""));
  const mode = capture.mode ?? "subject_rotates";

  useEffect(() => {
    setBaseline(String(capture.baseline_mm ?? ""));
    setRevolutions(String(capture.revolutions ?? ""));
  }, [capture.baseline_mm, capture.revolutions]);

  async function patch(body: Record<string, unknown>) {
    await fetch(`/api/runs/${runId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    onChange();
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>How this was captured</h3>
      </div>

      <div className="mode-choices">
        {MODES.map((m) => (
          <button
            key={m.value}
            className={`mode-choice ${mode === m.value ? "active" : ""}`}
            onClick={() => void patch({ capture_mode: m.value })}
          >
            <span className="mode-title">{m.title}</span>
            <span className="mode-blurb">{m.blurb}</span>
          </button>
        ))}
      </div>

      <div className="params-grid" style={{ paddingTop: 4 }}>
        <label className="param">
          <span className="param-name">Camera baseline (mm)</span>
          <input
            value={baseline}
            placeholder="measure it"
            onChange={(e) => setBaseline(e.target.value)}
            onBlur={() => {
              const v = Number(baseline);
              if (baseline !== "" && Number.isFinite(v)) void patch({ baseline_mm: v });
            }}
          />
          <span className="param-help">
            Lens-to-lens distance between the two fixed cameras. The only route to real
            millimetres — photogrammetry recovers shape but never size. Meaningless for
            handheld cameras, whose separation changes constantly.
          </span>
        </label>

        <label className="param">
          <span className="param-name">Revolutions in the take</span>
          <input
            value={revolutions}
            placeholder="e.g. 1.5"
            onChange={(e) => setRevolutions(e.target.value)}
            onBlur={() => {
              const v = Number(revolutions);
              if (revolutions !== "" && Number.isFinite(v))
                void patch({ revolutions: v });
            }}
          />
          <span className="param-help">
            How many full turns the footage covers. Without it the rotation-rate check
            has nothing to calibrate against, so it stays silent rather than guessing.
          </span>
        </label>
      </div>
    </div>
  );
}
