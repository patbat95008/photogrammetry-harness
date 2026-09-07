import { useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import type { Clip, RunDetail, SyncResults } from "../api/types";
import CapturePanel from "./CapturePanel";
import DiskUsage from "./DiskUsage";
import FileBrowser from "./FileBrowser";

function ClipCard({
  runId,
  clip,
  onChange,
}: {
  runId: string;
  clip: Clip;
  onChange: () => void;
}) {
  const p = clip.probe;
  const [offset, setOffset] = useState(String(clip.time_offset_s));

  useEffect(() => setOffset(String(clip.time_offset_s)), [clip.time_offset_s]);

  async function commitOffset() {
    const value = Number(offset);
    if (Number.isFinite(value) && value !== clip.time_offset_s) {
      await api.updateClip(runId, clip.clip_id, { time_offset_s: value });
      onChange();
    }
  }

  return (
    <div className={`clip-card ${clip.enabled ? "" : "disabled"}`}>
      <img
        className="clip-poster"
        src={api.posterUrl(runId, clip.clip_id)}
        alt={`${clip.clip_id} poster frame`}
        loading="lazy"
      />
      <div className="clip-body">
        <div className="clip-title">
          <strong>{clip.clip_id}</strong>
          <span className="tag">{clip.camera_group}</span>
          {p.is_hdr && <span className="tag warn-tag">HDR</span>}
          {p.is_vfr_suspected && <span className="tag warn-tag">VFR</span>}
          {!p.has_audio && <span className="tag warn-tag">no audio</span>}
          {p.effective_rotation !== 0 && (
            <span className="tag">rot {p.effective_rotation}°</span>
          )}
        </div>

        <div className="clip-meta mono-sm">
          {p.width}×{p.height} · {p.avg_frame_rate?.toFixed(2)} fps ·{" "}
          {p.duration_s?.toFixed(1)}s · {p.codec_name} · {p.nb_frames} frames
        </div>
        <div className="clip-path mono-sm muted">{clip.source_path}</div>

        <div className="clip-controls">
          <label>
            offset
            <input
              className="mono-sm"
              value={offset}
              onChange={(e) => setOffset(e.target.value)}
              onBlur={() => void commitOffset()}
              size={9}
            />
            s
          </label>
          {clip.sync_confidence != null && (
            <span
              className={`tag ${clip.sync_confidence >= 25 ? "ok" : "warn-tag"}`}
              title="Peak-to-sidelobe ratio of the audio cross-correlation"
            >
              conf {clip.sync_confidence}
            </span>
          )}
          {clip.sync_method && <span className="muted mono-sm">{clip.sync_method}</span>}
          <button
            className="refresh"
            onClick={async () => {
              await api.updateClip(runId, clip.clip_id, { enabled: !clip.enabled });
              onChange();
            }}
          >
            {clip.enabled ? "Disable" : "Enable"}
          </button>
          <button
            className="refresh"
            onClick={async () => {
              if (!confirm(`Remove ${clip.clip_id}?`)) return;
              await api.deleteClip(runId, clip.clip_id);
              onChange();
            }}
          >
            Remove
          </button>
        </div>
      </div>
    </div>
  );
}

export default function ClipsPage() {
  const { runId = "" } = useParams();
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [browsing, setBrowsing] = useState<null | {
    segment: string;
    kind: "rig" | "single" | "independent";
  }>(null);
  // Which kind of source this run is built from. A run is one or the other: the
  // slot clock that pairs frames across cameras comes from video timestamps, and
  // photographs have none.
  const [sourceKind, setSourceKind] = useState<"video" | "photos">("video");
  const [error, setError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<SyncResults | null>(null);

  const load = useCallback(async () => {
    try {
      setDetail(await api.getRun(runId));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [runId]);

  useEffect(() => {
    void load();
  }, [load]);

  async function pick(path: string) {
    const target = browsing;
    setBrowsing(null);
    if (!target) return;
    setError(null);
    try {
      const name = path.split(/[\\/]/).pop() ?? "clip";
      const role = name.replace(/\.[^.]+$/, "");
      await api.addClip(runId, {
        source_path: path,
        role,
        camera_group: role,
        segment_id: target.segment,
        segment_kind: sourceKind === "photos" ? "single" : target.kind,
        kind: sourceKind,
      });
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function runSync(segmentId: string) {
    setSyncing(true);
    setError(null);
    try {
      setSyncResult(await api.autoSync(runId, segmentId));
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSyncing(false);
    }
  }

  if (!detail) return <div className="main loading">Loading run…</div>;

  const { manifest } = detail;
  const segments = manifest.segments.length
    ? manifest.segments
    : [{ segment_id: "seg0", label: "Main spin", kind: "rig" as const, clip_ids: [], slot_base: 0, slot_count: 0 }];

  return (
    <div className="main">
      <div className="page-header">
        <h2>{manifest.name || manifest.run_id}</h2>
        <p>
          Source clips for this capture. Clips recorded at the same time share a
          segment and are paired frame-for-frame; a separate pass gets its own.
        </p>
      </div>

      {error && <div className="banner bad">{error}</div>}

      {manifest.clips.length === 0 && (
        <div className="source-toggle">
          <button
            className={sourceKind === "video" ? "on" : ""}
            onClick={() => setSourceKind("video")}
          >
            <strong>Video clips</strong>
            <span>
              A chair spin or a handheld orbit. Frames are cut on a shared timeline,
              so two cameras can be paired instant by instant.
            </span>
          </button>
          <button
            className={sourceKind === "photos" ? "on" : ""}
            onClick={() => setSourceKind("photos")}
          >
            <strong>Photo set</strong>
            <span>
              A folder of stills. No rolling shutter, no frame-rate guessing, and real
              EXIF — the cleanest input there is, and the simplest thing to try first.
            </span>
          </button>
        </div>
      )}

      <CapturePanel runId={runId} manifest={manifest} onChange={() => void load()} />

      {segments.map((segment) => {
        const clips = manifest.clips.filter((c) => c.segment_id === segment.segment_id);
        const enabled = clips.filter((c) => c.enabled);
        return (
          <div className="panel" key={segment.segment_id}>
            <div className="panel-head">
              <h3>
                {segment.label || segment.segment_id}
                <span className="tag" style={{ marginLeft: 10 }}>
                  {segment.kind === "rig"
                    ? "fixed mounts · rig"
                    : segment.kind === "independent"
                      ? "simultaneous · handheld"
                      : "separate pass"}
                </span>
              </h3>
              <div style={{ display: "flex", gap: 8 }}>
                {segment.kind !== "single" && enabled.length >= 2 && (
                  <button
                    className="primary"
                    disabled={syncing}
                    onClick={() => void runSync(segment.segment_id)}
                  >
                    {syncing ? "Correlating…" : "Auto-sync audio"}
                  </button>
                )}
                <button
                  className="refresh"
                  onClick={() =>
                    setBrowsing({ segment: segment.segment_id, kind: segment.kind })
                  }
                >
                  Add clip
                </button>
              </div>
            </div>

            <div className="panel-body">
              {clips.length === 0 ? (
                <div className="stub-note" style={{ margin: 16 }}>
                  No clips in this segment yet.
                </div>
              ) : (
                clips.map((clip) => (
                  <ClipCard
                    key={clip.clip_id}
                    runId={runId}
                    clip={clip}
                    onChange={() => void load()}
                  />
                ))
              )}
            </div>

            {segment.kind === "rig" && enabled.length === 1 && (
              <div className="row">
                <span />
                <div className="row-note">
                  Only one clip here. If you filmed with two cameras at once, add the
                  second so their frames can be paired into a rig.
                </div>
              </div>
            )}
            {segment.kind === "independent" && (
              <div className="row">
                <span />
                <div className="row-note">
                  These cameras were handheld, so their relative pose changes every
                  frame. Frames still share a timeline, but no rigid rig will be
                  declared — they are solved as independent cameras.
                </div>
              </div>
            )}
          </div>
        );
      })}

      <div className="panel">
        <div className="panel-head">
          <h3>Add another pass</h3>
        </div>
        <div className="panel-body" style={{ padding: 12 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              className="refresh"
              onClick={() =>
                setBrowsing({ segment: `seg${manifest.segments.length}`, kind: "single" })
              }
            >
              Separate pass (e.g. the crown pass)
            </button>
            <button
              className="refresh"
              onClick={() =>
                setBrowsing({
                  segment: `seg${manifest.segments.length}`,
                  kind: "independent",
                })
              }
            >
              Simultaneous handheld cameras
            </button>
          </div>
          <div className="muted" style={{ marginTop: 8, fontSize: 12.5 }}>
            A pass filmed at a different time gets its own segment: it shares no
            instants with the main spin, so its frames are not rig-paired. Handheld
            cameras share a timeline but not a fixed relative pose, so they are paired
            in time yet solved independently.
          </div>
        </div>
      </div>

      {syncResult && (
        <div className={`banner ${syncResult.warnings.length ? "bad" : "ok"}`}>
          <div>
            <strong>
              Synced against {syncResult.reference_clip_id}
            </strong>
            {Object.entries(syncResult.results).map(([id, r]) => (
              <div key={id} className="mono-sm">
                {id}: {r.offset_s >= 0 ? "+" : ""}
                {r.offset_s.toFixed(4)}s
                {r.confidence != null && ` (confidence ${r.confidence})`}
                {r.reference && " — reference"}
              </div>
            ))}
            {syncResult.warnings.map((w) => (
              <div key={w} className="row-note">
                {w}
              </div>
            ))}
          </div>
        </div>
      )}

      <DiskUsage usage={detail.disk_usage ?? {}} />

      {browsing && (
        <FileBrowser
          mode={sourceKind}
          onPick={pick}
          onClose={() => setBrowsing(null)}
        />
      )}
    </div>
  );
}
