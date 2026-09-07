const COLOURS: Record<string, string> = {
  frames: "#4c9aff",
  thumbs: "#7a5cff",
  masks: "#3fb950",
  probe: "#6b7885",
  sparse: "#d29922",
  dense: "#f85149",
  mesh: "#e06ebd",
  export: "#42c8b0",
};

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

/**
 * Where this run's bytes are going.
 *
 * Worth showing because the dense stage produces tens of gigabytes of depth maps
 * that are useless once fusion has run — and nobody deletes what they cannot see.
 */
export default function DiskUsage({ usage }: { usage: Record<string, number> }) {
  const entries = Object.entries(usage).filter(([, bytes]) => bytes > 0);
  if (entries.length === 0) return null;

  const total = entries.reduce((sum, [, bytes]) => sum + bytes, 0);

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Disk usage</h3>
        <span className="muted mono-sm">{formatBytes(total)}</span>
      </div>

      <div className="disk-bar" style={{ marginTop: 14 }}>
        {entries.map(([name, bytes]) => (
          <div
            key={name}
            className="disk-seg"
            title={`${name}: ${formatBytes(bytes)}`}
            style={{
              width: `${(bytes / total) * 100}%`,
              background: COLOURS[name] ?? "var(--border-strong)",
            }}
          />
        ))}
      </div>

      <div className="disk-legend">
        {entries.map(([name, bytes]) => (
          <span key={name}>
            <span
              className="swatch"
              style={{ background: COLOURS[name] ?? "var(--border-strong)" }}
            />
            {name} {formatBytes(bytes)}
          </span>
        ))}
      </div>
    </div>
  );
}
