import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import StageShell, { StageDetail } from "./StageShell";
import MeshViewer from "./MeshViewer";

/**
 * The turntable, as an auto-advancing strip.
 *
 * A rendered orbit is a record of what was actually exported, after cleanup, orientation
 * and decimation -- which is not the same object the mesh stage showed. Playing it is the
 * quickest way to see that the model is standing up and in one piece.
 */
function Turntable({ runId, frames }: { runId: string; frames: number }) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing || frames < 2) return;
    const timer = window.setInterval(
      () => setIndex((i) => (i + 1) % frames),
      1000 / 12,
    );
    return () => window.clearInterval(timer);
  }, [playing, frames]);

  if (!frames) return null;
  const name = `frame_${String(index).padStart(3, "0")}.png`;

  return (
    <div className="turntable">
      <img
        src={`/api/runs/${runId}/artifacts/export/turntable/${name}`}
        alt={`turntable frame ${index + 1} of ${frames}`}
        onClick={() => setPlaying((p) => !p)}
        title={playing ? "Click to pause" : "Click to play"}
      />
      <label className="cloud-control">
        <input
          type="range"
          min={0}
          max={frames - 1}
          value={index}
          onChange={(e) => {
            setPlaying(false);
            setIndex(Number(e.target.value));
          }}
        />
        {index + 1}/{frames}
      </label>
    </div>
  );
}

function Dimensions({ metrics }: { metrics: Record<string, unknown> }) {
  const dims = metrics.dimensions as number[] | undefined;
  if (!dims || dims.length !== 3) return null;
  const units = String(metrics.dimension_units ?? "model units");
  const unscaled = units !== "mm";

  return (
    <p className={unscaled ? "stage-note cloud-warn" : "stage-note"}>
      {dims.map((d) => d.toFixed(unscaled ? 3 : 1)).join(" × ")} {units}
      {unscaled && " — nothing in this capture could give the model a real size, so it " +
        "will print at whatever size the slicer guesses."}
    </p>
  );
}

export default function ExportPage() {
  const { runId = "" } = useParams();

  return (
    <StageShell runId={runId} stageId="export">
      {(detail: StageDetail) => {
        if (detail.state !== "done" && detail.state !== "stale") return null;

        const artifacts = detail.record.artifacts ?? {};
        const metrics = detail.record.metrics ?? {};
        const downloads = Object.entries(artifacts).filter(([key]) =>
          key.startsWith("model_"),
        );
        // An OBJ needs its .mtl and its texture to be an OBJ at all. They are not
        // models, so they are listed apart rather than sitting beside GLB and STL.
        const sidecars = Object.entries(artifacts).filter(([key]) =>
          key.startsWith("sidecar_"),
        );
        const glb = artifacts.model_glb;
        const modelBytes = Number(metrics.model_bytes ?? 0) || undefined;

        return (
          <div className="stage-results">
            <h3>Exported model</h3>
            <p className="stage-note">
              This is the finished object: cleaned, stood up from the plane the cameras
              lie in, and turned the way the Flip toggle said. Check it is the right way
              up before printing — the orientation is recovered rather than measured, and
              a model exported upside down still opens perfectly happily.
            </p>

            <Dimensions metrics={metrics} />

            {glb && (
              <MeshViewer
                meshUrl={`/api/runs/${runId}/artifacts/${glb}`}
                format="glb"
                runId={runId}
                faces={Number(metrics.faces_out ?? 0) || undefined}
                // Without this the 40 MB "load it?" gate can never fire here, and
                // this is the page that needs it: the cup export is 68 MB against a
                // 23 MB mesh, because Blender writes normals the mesh stage's file
                // does not have.
                sizeBytes={modelBytes}
                allowFlip={false}
                height={480}
              />
            )}

            <Turntable runId={runId} frames={Number(metrics.turntable_frames ?? 0)} />

            {downloads.length > 0 && (
              <div className="export-downloads">
                {downloads.map(([key, path]) => (
                  <a key={key} href={`/api/runs/${runId}/artifacts/${path}`} download>
                    {key.replace("model_", "").toUpperCase()}
                  </a>
                ))}
              </div>
            )}

            {sidecars.length > 0 && (
              <p className="stage-note">
                The OBJ needs these beside it, or it opens untextured:{" "}
                {sidecars.map(([key, path], index) => (
                  <span key={key}>
                    {index > 0 && ", "}
                    <a href={`/api/runs/${runId}/artifacts/${path}`} download>
                      {String(path).split("/").pop()}
                    </a>
                  </span>
                ))}
              </p>
            )}
          </div>
        );
      }}
    </StageShell>
  );
}
