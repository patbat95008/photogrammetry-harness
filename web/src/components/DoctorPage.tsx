import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { DoctorReport, ToolStatus } from "../api/types";

function Dot({ state }: { state: "ok" | "warn" | "bad" }) {
  const glyph = state === "ok" ? "●" : state === "warn" ? "●" : "○";
  return <span className={`dot ${state}`}>{glyph}</span>;
}

function toolState(tool: ToolStatus): "ok" | "warn" | "bad" {
  if (!tool.found) return tool.required ? "bad" : "warn";
  return tool.notes.length > 0 ? "warn" : "ok";
}

function ToolRow({ tool }: { tool: ToolStatus }) {
  return (
    <>
      <div className="row">
        <Dot state={toolState(tool)} />
        <div className="row-label">{tool.label}</div>
        <div className="row-value">
          {tool.found ? (tool.version ?? tool.path) : "not found"}
        </div>
        <div>
          {tool.cuda === true && <span className="tag ok">CUDA</span>}
          {tool.cuda === false && <span className="tag">no CUDA</span>}
          {!tool.required && !tool.found && <span className="tag">optional</span>}
        </div>
      </div>
      {tool.notes.map((note) => (
        <div key={note} className="row">
          <span />
          <div className="row-note">{note}</div>
        </div>
      ))}
    </>
  );
}

export default function DoctorPage() {
  const [report, setReport] = useState<DoctorReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setReport(await api.doctor());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading && !report) return <div className="main loading">Probing environment…</div>;
  if (error && !report)
    return (
      <div className="main error">
        Could not reach the harness server: {error}
        <div className="muted" style={{ marginTop: 12 }}>
          Is uvicorn running on port 8756?
        </div>
      </div>
    );
  if (!report) return null;

  const { torch, gpu, storage, sam2, colmap_dialect: dialect } = report;

  return (
    <div className="main">
      <div className="page-header">
        <h2>Doctor</h2>
        <p>Every external dependency the pipeline relies on, probed live.</p>
      </div>

      <div className={`banner ${report.healthy ? "ok" : "bad"}`}>
        <Dot state={report.healthy ? "ok" : "bad"} />
        <div>
          <strong>
            {report.healthy
              ? "Environment is ready."
              : `${report.blocking.length} blocking issue${report.blocking.length === 1 ? "" : "s"}`}
          </strong>
          {report.healthy ? (
            <span className="muted">
              All tools resolved, CUDA available, SAM 2 weights present.
            </span>
          ) : (
            <ul>
              {report.blocking.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          )}
        </div>
        <button className="refresh" onClick={() => void load()} style={{ marginLeft: "auto" }}>
          {loading ? "…" : "Re-probe"}
        </button>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h3>External tools</h3>
        </div>
        <div className="panel-body">
          {report.tools.map((tool) => (
            <ToolRow key={tool.key} tool={tool} />
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h3>GPU &amp; PyTorch</h3>
        </div>
        <div className="stat-grid">
          <div className="stat">
            <div className="stat-label">Device</div>
            <div className="stat-value" style={{ fontSize: 13 }}>
              {gpu.available ? gpu.name : "—"}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">VRAM free</div>
            <div className="stat-value">
              {gpu.available
                ? `${(gpu.vram_free_mb! / 1024).toFixed(1)} / ${(gpu.vram_total_mb! / 1024).toFixed(0)} GB`
                : "—"}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">torch</div>
            <div className="stat-value">{torch.version ?? "not installed"}</div>
          </div>
          <div className="stat">
            <div className="stat-label">torch CUDA</div>
            <div className="stat-value" style={{ color: torch.cuda_available ? "var(--ok)" : "var(--bad)" }}>
              {torch.installed ? (torch.cuda_available ? `cu${torch.cuda_version}` : "unavailable") : "—"}
            </div>
          </div>
        </div>
        {torch.warning && (
          <div className="row">
            <span />
            <div className="row-note">{torch.warning}</div>
          </div>
        )}
      </div>

      <div className="panel">
        <div className="panel-head">
          <h3>SAM 2 checkpoints</h3>
          <span className="muted" style={{ fontSize: 12 }}>
            checkpoint ↔ config pairs are pinned in the registry
          </span>
        </div>
        <div className="panel-body">
          {sam2.models.map((m) => (
            <div className="row" key={m.key}>
              <Dot state={m.checkpoint_present && m.config_present ? "ok" : "warn"} />
              <div className="row-label">
                {m.label}
                {m.is_default && (
                  <span className="tag accent" style={{ marginLeft: 8 }}>
                    default
                  </span>
                )}
              </div>
              <div className="row-value">
                {m.config}
                {!m.config_present && " (config missing)"}
              </div>
              <div className="row-value">
                {m.checkpoint_present ? `${m.checkpoint_mb} MB` : "absent"}
              </div>
            </div>
          ))}
          {!sam2.package_importable && (
            <div className="row">
              <span />
              <div className="row-note">
                sam2 package not importable: {sam2.import_error}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h3>COLMAP option dialect</h3>
          <span className="muted" style={{ fontSize: 12 }}>
            both namespaces coexist in 4.2 — argv builders read this rather than guess
          </span>
        </div>
        <div className="panel-body">
          {dialect.available ? (
            [
              ["FeatureMatching.* namespace", dialect.feature_matching_ns],
              ["SiftMatching.* namespace", dialect.sift_matching_ns],
              ["FeatureExtraction.* namespace", dialect.feature_extraction_ns],
              ["SiftExtraction.* namespace", dialect.sift_extraction_ns],
              ["ImageReader.single_camera_per_folder", dialect.single_camera_per_folder],
              ["ImageReader.mask_path", dialect.mask_path],
              ["TwoViewGeometry.filter_stationary_matches", dialect.filter_stationary_matches],
            ].map(([label, present]) => (
              <div className="row" key={label as string}>
                <Dot state={present ? "ok" : "warn"} />
                <div className="row-label" style={{ gridColumn: "2 / -1" }}>
                  <span className="row-value">{label as string}</span>
                </div>
              </div>
            ))
          ) : (
            <div className="row">
              <span />
              <div className="row-note">COLMAP not found — dialect unknown</div>
            </div>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h3>Storage</h3>
        </div>
        <div className="stat-grid">
          <div className="stat">
            <div className="stat-label">Runs root</div>
            <div className="stat-value" style={{ fontSize: 13 }}>
              {storage.runs_root}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Free</div>
            <div className="stat-value">
              {storage.free_gb != null ? `${storage.free_gb} GB` : "—"}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Long paths</div>
            <div className="stat-value">
              {storage.long_paths_enabled ? "enabled" : "disabled"}
            </div>
          </div>
          <div className="stat">
            <div className="stat-label">Python</div>
            <div className="stat-value">{report.python}</div>
          </div>
        </div>
        {storage.notes?.map((n) => (
          <div className="row" key={n}>
            <span />
            <div className="row-note">{n}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
