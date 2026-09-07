import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { BrowseResult } from "../api/types";

function formatBytes(bytes: number): string {
  if (bytes > 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

/**
 * Picks a video file from the server's own disk.
 *
 * Deliberately not a browser upload: these clips are multi-gigabyte, and pushing
 * one through the browser to a server running on the same machine would copy it
 * for no reason. Browsing is restricted server-side to the configured ingest roots.
 */
export default function FileBrowser({
  onPick,
  onClose,
}: {
  onPick: (path: string) => void;
  onClose: () => void;
}) {
  const [listing, setListing] = useState<BrowseResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (path?: string) => {
    setError(null);
    try {
      setListing(await api.browse(path));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="panel-head">
          <h3>Choose a video</h3>
          <button className="refresh" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="browser-path mono-sm">{listing?.path ?? "Ingest roots"}</div>

        {error && <div className="row-note" style={{ padding: "8px 16px" }}>{error}</div>}

        <div className="browser-list">
          {listing?.parent && (
            <button className="browser-item" onClick={() => void load(listing.parent!)}>
              <span className="glyph">↰</span> ..
            </button>
          )}
          {listing?.directories.map((dir) => (
            <button
              key={dir.path}
              className="browser-item"
              onClick={() => void load(dir.path)}
            >
              <span className="glyph">▸</span> {dir.name}
            </button>
          ))}
          {listing?.files.map((file) => (
            <button
              key={file.path}
              className="browser-item file"
              onClick={() => onPick(file.path)}
            >
              <span className="glyph">▪</span> {file.name}
              <span className="muted mono-sm">{formatBytes(file.size_bytes)}</span>
            </button>
          ))}
          {listing &&
            listing.directories.length === 0 &&
            listing.files.length === 0 && (
              <div className="muted" style={{ padding: 16 }}>
                No folders or video files here.
              </div>
            )}
        </div>
      </div>
    </div>
  );
}
