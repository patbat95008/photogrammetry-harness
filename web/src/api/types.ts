export interface ToolStatus {
  key: string;
  label: string;
  found: boolean;
  path: string | null;
  version: string | null;
  required: boolean;
  cuda: boolean | null;
  notes: string[];
}

export interface Sam2Model {
  key: string;
  label: string;
  checkpoint: string;
  checkpoint_present: boolean;
  checkpoint_mb: number | null;
  config: string;
  config_present: boolean;
  is_default: boolean;
}

export interface DoctorReport {
  project_root: string;
  python: string;
  tools: ToolStatus[];
  colmap_dialect: {
    available: boolean;
    feature_matching_ns?: boolean;
    sift_matching_ns?: boolean;
    feature_extraction_ns?: boolean;
    sift_extraction_ns?: boolean;
    single_camera_per_folder?: boolean;
    mask_path?: boolean;
    filter_stationary_matches?: boolean;
  };
  sam2: {
    package_importable: boolean;
    import_error: string | null;
    default_model: string;
    models: Sam2Model[];
  };
  torch: {
    installed: boolean;
    error?: string;
    version?: string;
    cuda_available?: boolean;
    cuda_version?: string | null;
    device_name?: string;
    vram_total_gb?: number;
    warning?: string;
  };
  gpu: {
    available: boolean;
    name?: string;
    vram_total_mb?: number;
    vram_used_mb?: number;
    vram_free_mb?: number;
    driver_version?: string;
  };
  storage: {
    runs_root: string;
    runs_root_exists?: boolean;
    free_gb?: number;
    total_gb?: number;
    long_paths_enabled?: boolean | null;
    notes?: string[];
    error?: string;
  };
  blocking: string[];
  healthy: boolean;
}

// -- runs, clips, stages -----------------------------------------------------

export type StageState =
  | "pending"
  | "running"
  | "done"
  | "stale"
  | "failed"
  | "cancelled";

export interface StageSummary {
  state: StageState;
  stale_reason: string | null;
  blocked_by: string[];
  implemented: boolean;
  runnable?: boolean;
  fingerprint?: string;
}

export interface RunSummary {
  run_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  clip_count: number;
  total_slots: number;
  stages: Record<string, StageSummary>;
  error?: string;
}

export interface ClipProbe {
  duration_s: number | null;
  codec_name: string | null;
  pix_fmt: string | null;
  width: number | null;
  height: number | null;
  avg_frame_rate: number | null;
  nb_frames: number | null;
  color_transfer: string | null;
  rotation: number;
  effective_rotation: number;
  is_hdr: boolean;
  is_vfr_suspected: boolean;
  has_audio: boolean;
}

export interface Clip {
  clip_id: string;
  camera_group: string;
  role: string;
  segment_id: string;
  source_path: string;
  probe: ClipProbe;
  time_offset_s: number;
  sync_confidence: number | null;
  sync_method: string | null;
  enabled: boolean;
}

export interface Segment {
  segment_id: string;
  label: string;
  kind: "rig" | "single" | "independent";
  clip_ids: string[];
  slot_base: number;
  slot_count: number;
}

export interface RunManifest {
  run_id: string;
  name: string;
  created_at: string;
  capture: {
    mode: "subject_rotates" | "camera_orbits";
    baseline_mm: number | null;
    revolutions: number | null;
    scale_reference_note: string;
    notes: string;
  };
  clips: Clip[];
  segments: Segment[];
  timeline: {
    fps: number | null;
    total_slots: number;
    sync_residual_p95_s: number | null;
    degrees_per_second: number | null;
  };
  stages: Record<string, unknown>;
}

export interface RunDetail {
  manifest: RunManifest;
  evaluation: Record<string, StageSummary>;
  active_jobs: unknown[];
  event_seq: number;
  disk_usage: Record<string, number>;
}

export interface SyncResults {
  reference_clip_id: string;
  results: Record<
    string,
    {
      offset_s: number;
      confidence: number | null;
      peak?: number;
      trustworthy?: boolean;
      reference: boolean;
    }
  >;
  warnings: string[];
}

export interface BrowseResult {
  path: string | null;
  parent: string | null;
  directories: { name: string; path: string }[];
  files: { name: string; path: string; size_bytes: number }[];
}
