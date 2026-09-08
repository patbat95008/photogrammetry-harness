import { NavLink, Navigate, Route, Routes, useParams } from "react-router-dom";
import ClipsPage from "./components/ClipsPage";
import DensePage from "./components/DensePage";
import DoctorPage from "./components/DoctorPage";
import ExtractPage from "./components/ExtractPage";
import MeshPage from "./components/MeshPage";
import RunsPage from "./components/RunsPage";
import SelectPage from "./components/SelectPage";
import SparsePage from "./components/SparsePage";
import StageStub from "./components/StageStub";

/**
 * The seven pipeline stages, in DAG order. Those with a page of their own are listed
 * in BUILT_STAGES below; the rest render as stubs inside the same shell so the shape
 * of the pipeline stays visible from the start.
 */
const BUILT_STAGES: Record<string, () => JSX.Element | null> = {
  extract: ExtractPage,
  select: SelectPage,
  sparse: SparsePage,
  dense: DensePage,
  mesh: MeshPage,
};

const STAGES = [
  { id: "extract", label: "Extract frames" },
  { id: "select", label: "Select frames" },
  { id: "mask", label: "Mask" },
  { id: "sparse", label: "Align" },
  { id: "dense", label: "Dense cloud" },
  { id: "mesh", label: "Mesh" },
  { id: "export", label: "Export" },
];

function Sidebar() {
  const { runId } = useParams();

  return (
    <nav className="sidebar">
      <h1>Photogrammetry Harness</h1>

      <div className="nav-section">Environment</div>
      <NavLink
        to="/doctor"
        className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
      >
        <span className="step">·</span> Doctor
      </NavLink>
      <NavLink
        to="/runs"
        end
        className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
      >
        <span className="step">·</span> Runs
      </NavLink>

      {runId && (
        <>
          <div className="nav-section">Capture</div>
          <NavLink
            to={`/runs/${runId}/clips`}
            className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
          >
            <span className="step">·</span> Clips &amp; sync
          </NavLink>

          <div className="nav-section">Pipeline</div>
          {STAGES.map((stage, i) => (
            <NavLink
              key={stage.id}
              to={`/runs/${runId}/stages/${stage.id}`}
              className={({ isActive }) =>
                `nav-item ${isActive ? "active" : ""}`
              }
            >
              <span className="step">{i + 1}</span> {stage.label}
            </NavLink>
          ))}
        </>
      )}
    </nav>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="app">
      <Sidebar />
      {children}
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/runs" replace />} />
      <Route
        path="/doctor"
        element={
          <Shell>
            <DoctorPage />
          </Shell>
        }
      />
      <Route
        path="/runs"
        element={
          <Shell>
            <RunsPage />
          </Shell>
        }
      />
      <Route
        path="/runs/:runId/clips"
        element={
          <Shell>
            <ClipsPage />
          </Shell>
        }
      />
      {Object.entries(BUILT_STAGES).map(([id, Page]) => (
        <Route
          key={id}
          path={`/runs/:runId/stages/${id}`}
          element={
            <Shell>
              <Page />
            </Shell>
          }
        />
      ))}
      {STAGES.filter((s) => !(s.id in BUILT_STAGES)).map((stage) => (
        <Route
          key={stage.id}
          path={`/runs/:runId/stages/${stage.id}`}
          element={
            <Shell>
              <StageStub stageId={stage.id} />
            </Shell>
          }
        />
      ))}
    </Routes>
  );
}
