import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { BrowseResult } from "../api/types";

function formatBytes(bytes: number): string {
  if (bytes > 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

/**
 * Picks a source from the server's own disk: a video file, or a folder of photographs.
 *
 * Deliberately not a browser upload: these sources are gigabytes, and pushing one
 * through the browser to a server running on the same machine would copy it for no
 * reason. Browsing is restricted server-side to the configured ingest roots.
 *
 * In photo mode the thing being picked is the folder itself, so directories become
 * choosable as well as enterable -- hence the split row.
 */
export default function FileBrowser({
  onPick,
  onClose,
  mode = "video",
}: {
  onPick: (path: string) => void;
  onClose: () => void;
  mode?: "video" | "photos";
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
          <h3>{mode === "photos" ? "Choose a folder of photographs" : "Choose a video"}</h3>
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
            <div key={dir.path} className="browser-row">
              <button className="browser-item" onClick={() => void load(dir.path)}>
                <span className="glyph">▸</span> {dir.name}
                {mode === "photos" && (dir.photo_count ?? 0) > 0 && (
                  <span className="muted mono-sm">{dir.photo_count} photos</span>
                )}
              </button>
              {mode === "photos" && (dir.photo_count ?? 0) > 0 && (
                <button className="refresh" onClick={() => onPick(dir.path)}>
                  Use
                </button>
              )}
            </div>
          ))}
          {mode === "photos" && listing?.path && (
            <button className="browser-item file" onClick={() => onPick(listing.path!)}>
              <span className="glyph">▪</span> Use this folder
            </button>
          )}
          {mode === "video" && listing?.files.map((file) => (
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
                {mode === "photos"
                  ? "No folders here."
                  : "No folders or video files here."}
              </div>
            )}
        </div>
      </div>
    </div>
  );
}
