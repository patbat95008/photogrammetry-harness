import { NavLink, Navigate, Route, Routes, useParams } from "react-router-dom";
import ClipsPage from "./components/ClipsPage";
import DoctorPage from "./components/DoctorPage";
import ExtractPage from "./components/ExtractPage";
import RunsPage from "./components/RunsPage";
import StageStub from "./components/StageStub";

/**
 * The seven pipeline stages, in DAG order. Only `extract` becomes real at M5;
 * the rest render as stubs inside the same shell so the shape of the pipeline is
 * visible from the start.
 */
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
      <Route
        path="/runs/:runId/stages/extract"
        element={
          <Shell>
            <ExtractPage />
          </Shell>
        }
      />
      {STAGES.filter((s) => s.id !== "extract").map((stage) => (
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
