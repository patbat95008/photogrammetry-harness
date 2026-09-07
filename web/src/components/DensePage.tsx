import { useParams } from "react-router-dom";
import StageShell, { StageDetail } from "./StageShell";
import PointCloudViewer from "./PointCloudViewer";

export default function DensePage() {
  const { runId = "" } = useParams();

  return (
    <StageShell runId={runId} stageId="dense">
      {(detail: StageDetail) =>
        detail.state === "done" || detail.state === "stale" ? (
          <div className="stage-results">
            <h3>Dense point cloud</h3>
            <p className="stage-note">
              A decimated preview — the full cloud stays on disk. Look for surfaces
              that are doubled, and for floating fragments where a reflective or
              transparent surface fooled the depth estimate. Raising “views that must
              agree” thins both of those, and thins genuine detail with them.
            </p>
            <PointCloudViewer
              cloudUrl={`/api/runs/${runId}/artifacts/dense/preview.ply`}
              runId={runId}
              height={520}
            />
          </div>
        ) : null
      }
    </StageShell>
  );
}
