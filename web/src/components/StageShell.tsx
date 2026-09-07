import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { useEventStream } from "../hooks/useEventStream";

interface StageDetail {
  stage_id: string;
  label: string;
  description: string;
  implemented: boolean;
  record: {
    state: string;
    params: Record<string, unknown>;
    metrics: Record<string, unknown>;
    warnings: string[];
    error: string | null;
    duration_s: number | null;
  };
  schema: JsonSchema | null;
  defaults: Record<string, unknown>;
  state: string;
  stale_reason: string | null;
  blocked_by: string[];
  runnable: boolean;
  preflight: string[];
}

interface JsonSchema {
  properties?: Record<string, SchemaProp>;
  $defs?: Record<string, { enum?: string[] }>;
}

interface SchemaProp {
  type?: string;
  title?: string;
  description?: string;
  enum?: string[];
  anyOf?: { type?: string; enum?: string[]; $ref?: string }[];
  allOf?: { $ref?: string }[];
  $ref?: string;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
}

/** Resolve the enum options a property allows, following $ref into $defs. */
function enumOptions(prop: SchemaProp, schema: JsonSchema): string[] | null {
  if (prop.enum) return prop.enum;
  const ref = prop.$ref ?? prop.allOf?.[0]?.$ref ?? prop.anyOf?.find((a) => a.$ref)?.$ref;
  if (ref) {
    const name = ref.split("/").pop()!;
    const def = schema.$defs?.[name];
    if (def?.enum) return def.enum;
  }
  const inline = prop.anyOf?.find((a) => a.enum)?.enum;
  return inline ?? null;
}

function isNumeric(prop: SchemaProp): boolean {
  if (prop.type === "number" || prop.type === "integer") return true;
  return (prop.anyOf ?? []).some((a) => a.type === "number" || a.type === "integer");
}

function isBoolean(prop: SchemaProp): boolean {
  return prop.type === "boolean";
}

function ParamsForm({
  schema,
  values,
  onChange,
  disabled,
}: {
  schema: JsonSchema;
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
  disabled: boolean;
}) {
  return (
    <div className="params-grid">
      {Object.entries(schema.properties ?? {}).map(([key, prop]) => {
        const options = enumOptions(prop, schema);
        const value = values[key];
        return (
          <label className="param" key={key}>
            <span className="param-name">{prop.title ?? key}</span>
            {options ? (
              <select
                value={String(value ?? "")}
                disabled={disabled}
                onChange={(e) => onChange(key, e.target.value)}
              >
                {options.map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            ) : isBoolean(prop) ? (
              <input
                type="checkbox"
                checked={Boolean(value)}
                disabled={disabled}
                onChange={(e) => onChange(key, e.target.checked)}
              />
            ) : (
              <input
                type={isNumeric(prop) ? "number" : "text"}
                step="any"
                value={value === null || value === undefined ? "" : String(value)}
                disabled={disabled}
                placeholder={isNumeric(prop) ? "auto" : ""}
                onChange={(e) => {
                  const raw = e.target.value;
                  if (raw === "") return onChange(key, null);
                  onChange(key, isNumeric(prop) ? Number(raw) : raw);
                }}
              />
            )}
            {prop.description && <span className="param-help">{prop.description}</span>}
          </label>
        );
      })}
    </div>
  );
}

const STATE_LABEL: Record<string, string> = {
  done: "Up to date",
  stale: "Out of date",
  pending: "Not run yet",
  running: "Running",
  failed: "Failed",
  cancelled: "Cancelled",
};

const STALE_EXPLANATION: Record<string, string> = {
  upstream: "an earlier stage changed, so these results no longer match its output",
  params_or_inputs: "a parameter or source clip changed since this last ran",
};

export default function StageShell({
  runId,
  stageId,
  children,
}: {
  runId: string;
  stageId: string;
  /** Rendered below the shell once the stage has results. */
  children?: (detail: StageDetail) => React.ReactNode;
}) {
  const [detail, setDetail] = useState<StageDetail | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const stream = useEventStream(runId, stageId);

  const load = useCallback(async () => {
    try {
      const res = await fetch(`/api/runs/${runId}/stages/${stageId}`);
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
      const data: StageDetail = await res.json();
      setDetail(data);
      setParams(
        Object.keys(data.record.params).length ? data.record.params : data.defaults,
      );
      setDirty(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [runId, stageId]);

  useEffect(() => {
    void load();
  }, [load]);

  // A finished or failed job means the server-side state changed underneath us.
  const lifecycle = stream.lastLifecycle?.type;
  useEffect(() => {
    if (
      lifecycle &&
      ["stage.finished", "stage.failed", "stage.cancelled"].includes(lifecycle)
    ) {
      void load();
    }
  }, [lifecycle, stream.lastLifecycle?.seq, load]);

  const running = detail?.state === "running" || Boolean(stream.progress);

  async function save() {
    setError(null);
    try {
      const res = await fetch(`/api/runs/${runId}/stages/${stageId}/params`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(
          Array.isArray(body.detail)
            ? body.detail.map((d: { msg: string }) => d.msg).join("; ")
            : body.detail,
        );
      }
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function start() {
    setError(null);
    try {
      if (dirty) await save();
      await api.runStage(runId, stageId);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const metricRows = useMemo(
    () => Object.entries(detail?.record.metrics ?? {}),
    [detail],
  );

  if (!detail) {
    return (
      <div className="main">
        {error ? <div className="banner bad">{error}</div> : <div className="loading">Loading…</div>}
      </div>
    );
  }

  const blocked = detail.blocked_by.length > 0 || detail.preflight.length > 0;

  return (
    <div className="main">
      <div className="page-header">
        <h2>{detail.label}</h2>
        <p>{detail.description}</p>
      </div>

      {error && <div className="banner bad">{error}</div>}

      <div className="panel">
        <div className="panel-head">
          <h3>
            {STATE_LABEL[detail.state] ?? detail.state}
            {detail.record.duration_s != null && detail.state === "done" && (
              <span className="muted" style={{ marginLeft: 10, textTransform: "none" }}>
                took {detail.record.duration_s}s
              </span>
            )}
          </h3>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span className={`conn ${stream.connected ? "on" : "off"}`}>
              {stream.connected ? "live" : "offline"}
            </span>
            {dirty && (
              <button className="refresh" onClick={() => void save()}>
                Save params
              </button>
            )}
            {running ? (
              <button
                className="refresh"
                onClick={() => void api.cancelStage(runId, stageId)}
              >
                Cancel
              </button>
            ) : (
              <button className="primary" disabled={blocked} onClick={() => void start()}>
                {detail.state === "done" ? "Re-run" : "Run"}
              </button>
            )}
          </div>
        </div>

        {detail.state === "stale" && (
          <div className="row">
            <span />
            <div className="row-note">
              These results are out of date:{" "}
              {STALE_EXPLANATION[detail.stale_reason ?? ""] ?? detail.stale_reason}. The
              files are still on disk; re-run to bring them up to date.
            </div>
          </div>
        )}

        {detail.preflight.map((p) => (
          <div className="row" key={p}>
            <span />
            <div className="row-note">{p}</div>
          </div>
        ))}
        {detail.blocked_by.length > 0 && (
          <div className="row">
            <span />
            <div className="row-note">
              Waiting on: {detail.blocked_by.join(", ")}
            </div>
          </div>
        )}

        {running && (
          <div className="progress-wrap">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${((stream.progress?.fraction ?? 0) * 100).toFixed(1)}%`,
                }}
              />
            </div>
            <div className="muted mono-sm">{stream.progress?.message ?? "starting…"}</div>
          </div>
        )}

        {detail.record.error && (
          <div className="row">
            <span />
            <div className="row-note" style={{ color: "var(--bad)" }}>
              {detail.record.error}
            </div>
          </div>
        )}
      </div>

      {detail.schema && (
        <div className="panel">
          <div className="panel-head">
            <h3>Parameters</h3>
          </div>
          <ParamsForm
            schema={detail.schema}
            values={params}
            disabled={running}
            onChange={(k, v) => {
              setParams((p) => ({ ...p, [k]: v }));
              setDirty(true);
            }}
          />
        </div>
      )}

      {stream.log.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <h3>Log</h3>
          </div>
          <pre className="log-view">{stream.log.join("\n")}</pre>
        </div>
      )}

      {detail.record.warnings.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <h3>Warnings</h3>
          </div>
          <div className="panel-body">
            {detail.record.warnings.map((w) => (
              <div className="row" key={w}>
                <span className="dot warn">●</span>
                <div style={{ gridColumn: "2 / -1" }}>{w}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {metricRows.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <h3>Results</h3>
          </div>
          <div className="panel-body">
            {metricRows.map(([key, value]) => (
              <div className="row" key={key}>
                <span />
                <div className="row-label">{key.replace(/_/g, " ")}</div>
                <div className="row-value">
                  {typeof value === "object" ? JSON.stringify(value) : String(value)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {children?.(detail)}
    </div>
  );
}
