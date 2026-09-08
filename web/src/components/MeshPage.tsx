import { useParams } from "react-router-dom";
import StageShell, { StageDetail } from "./StageShell";
import MeshViewer from "./MeshViewer";

/** The extension decides the loader, and the stage chose it. Do not infer it twice. */
function formatOf(path: string): "glb" | "gltf" | "obj" | "ply" {
  const suffix = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
  return suffix === "gltf" || suffix === "obj" || suffix === "ply" ? suffix : "glb";
}

export default function MeshPage() {
  const { runId = "" } = useParams();

  return (
    <StageShell runId={runId} stageId="mesh">
      {(detail: StageDetail) => {
        if (detail.state !== "done" && detail.state !== "stale") return null;

        // The filename depends on export_type, so it is read from the artifacts the
        // stage recorded rather than assumed -- the mesh is discovered, not named.
        const mesh = detail.record.artifacts?.mesh;
        if (!mesh) return null;

        const metrics = detail.record.metrics ?? {};
        const bytes =
          Number(metrics.mesh_bytes ?? 0) + Number(metrics.texture_bytes ?? 0);

        return (
          <div className="stage-results">
            <h3>Mesh</h3>
            <p className="stage-note">
              Look at it matte before you believe the texture. A photograph baked onto a
              lumpy or doubled surface still looks like a photograph of the real thing,
              and the matte shading with the light raking across it is what shows the
              difference. Holes that were closed rather than measured — the crown of a
              head, the underside of anything — read as smooth patches with no detail.
            </p>
            <MeshViewer
              meshUrl={`/api/runs/${runId}/artifacts/${mesh}`}
              format={formatOf(mesh)}
              runId={runId}
              sizeBytes={bytes || undefined}
              faces={Number(metrics.faces ?? 0) || undefined}
              height={560}
            />
          </div>
        );
      }}
    </StageShell>
  );
}
