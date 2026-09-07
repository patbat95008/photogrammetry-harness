# Photogrammetry Harness — Handover

**Status:** Stage 1 (video → frames) complete and verified. Stages 2–7 designed, stubbed, not built.
**Last verified:** 2026-09-07, against both synthetic ground-truth footage and a real phone clip.
**Tests:** 69 passing (`.venv\Scripts\python.exe -m pytest server/tests -q`).

---

## 1. What this is

A local web harness that turns video into a 3D model, built around existing
reconstruction engines rather than replacing them. The end goal is a printable 3D
bust of the owner's head, captured by rotating in an office chair while two phone
cameras record from different heights.

The harness owns the parts those engines do badly: ingest, frame selection,
masking, quality control, and making failures legible. COLMAP and OpenMVS do the
actual reconstruction.

```
FFmpeg → OpenCV/SAM 2 → COLMAP → OpenMVS → Blender
extract   select/mask    align    dense/mesh  cleanup
  ✅         ⬜ ⬜         ⬜        ⬜   ⬜       ⬜
```

---

## 2. Run it

```
launch.bat          double-click. Builds the UI if stale, serves API + UI on
                    http://127.0.0.1:8756, opens a browser when it's actually up.
                    Closing the window shuts everything down.

dev.bat             two windows: Vite on :5173 with hot reload, auto-reloading
                    API on :8756. Use this while editing.
```

Tests: `.venv\Scripts\python.exe -m pytest server/tests -q`

**Environment is fully provisioned and verified green on the Doctor page:** ffmpeg
(gyan full build, on PATH), COLMAP 4.2.0 CUDA, OpenMVS 2.4.0 CUDA prebuilt,
Blender 5.2 LTS, torch 2.14.0+cu130 on an RTX 4090 (24 GB), all four SAM 2.1
checkpoints. Python 3.12.9 in `.venv`. Runs live in `D:\pgh-runs`.

---

## 3. Where things are

```
launch.bat  dev.bat  requirements.txt
scripts/needs-build.ps1        UI freshness check (see §6.9)
docs/capture-checklist.md      read before filming — capture decides most outcomes
docs/HANDOVER.md               this file

server/pgh/
  config.py       tool paths, SAM2 checkpoint↔config pairs, ingest roots
  registry.py     doctor probes: versions, CUDA, COLMAP option dialect
  manifest.py     project.json models — StageId/State, CaptureMode, Segment, Clip
  fingerprint.py  the staleness engine  ← load-bearing, read this first
  timeline.py     shared slot clock across clips  ← the core of Stage 1
  jobs.py         single-worker queue, per-stage logs, lifecycle events
  proc.py         the ONLY module allowed to import subprocess (see §6.4)
  events.py       append-only event log + SSE bus
  store.py        run directories, crash recovery
  sync.py         audio cross-correlation
  files.py        source identity, ingest-root guard
  api/            runs clips stages sync fs artifacts events doctor deps
  stages/
    base.py       Stage contract: params, external_inputs, preflight, run
    registry.py   StageResolver the staleness engine runs against
    extract.py    Stage 1 ✅
    planned.py    copy for the six unbuilt stages, varies by capture mode
web/src/          Vite + React + TS. StageShell auto-builds param forms from the
                  pydantic JSON schema — a new stage gets a working UI for free.
```

### Run directory (`D:\pgh-runs\<run_id>\`)

```
project.json          state only, single source of truth
events.ndjson         seq-numbered; powers SSE replay on reconnect
probe/                verbatim ffprobe JSON, packet timelines, poster frames
frames/<group>/000042.jpg     slot-indexed, one folder per camera
frames/frames.jsonl   per-frame: dHash, luma, sync residual, duplicate flag
thumbs/<group>/       256px webp
logs/                 per-stage run logs
.partial/             stage scratch; committed by atomic rename
```

`D:\pgh-runs` rather than the project folder is deliberate: the dense stage builds
paths like `dense/undistorted/stereo/depth_maps/cam_high/000042.jpg.geometric.bin`,
and COLMAP and OpenMVS may not honour long-path support even though Windows has it
enabled here.

---

## 4. The two ideas everything else hangs off

### 4.1 Slot naming is a shared clock

`frames/cam_high/000042.jpg` and `frames/cam_eye/000042.jpg` are **the same instant**.
Not the same source frame number — the same moment in time.

This exists because two cameras on fixed mounts, in the *subject's* reference frame,
orbit the head in lockstep at a constant relative pose. That is a rigid two-sensor
rig, and COLMAP 4.2 can exploit it (`rig_configurator`, `--Mapper.ba_refine_sensor_from_rig`),
which constrains bundle adjustment hard and helps the orbit close. COLMAP groups rig
frames by **matching filename suffix across per-camera folders**, so the naming *is*
the mechanism.

Nothing depends on it yet — rigs land at M8 — but Stage 1 was built so that adding
them later needs no re-extraction. `timeline.py` is where this lives.

Segments carry it: clips recorded simultaneously share a segment and a timeline;
a separate pass (the crown shot) gets its own disjoint block of slot indices.

### 4.2 Capture mode inverts the pipeline

`manifest.capture.mode` is not bookkeeping. It flips two later decisions:

| | `subject_rotates` | `camera_orbits` |
|---|---|---|
| Background masking | **Mandatory.** The room is static in the world but moves relative to the head, so the solver locks onto it and the head never resolves. | **Skip it.** Background is rigid with the subject: extra features, better loop closure. |
| Rigid rig | Yes, on fixed mounts | No — handheld cameras have no constant relative pose |
| Scale source | Measured lens-to-lens baseline | Needs a ruler in frame |

`SegmentKind` has three values: `RIG` (fixed mounts, simultaneous), `INDEPENDENT`
(simultaneous but handheld — paired in time, solved as independent cameras), and
`SINGLE` (a separate pass).

### 4.3 Staleness, and why it's pleasant

`fingerprint.py` hashes each stage from its params (minus cosmetic ones), its
declared external inputs, and its upstream stages' fingerprints. Because it is
content-derived rather than a run counter, **changing a param and changing it back
leaves downstream work valid.** Verified on the live system, not just in tests.

Mark a param cosmetic with `json_schema_extra={"affects_fingerprint": False}` —
`thumbnail_px` is the example.

---

## 5. What's verified, and how

**Synthetic ground truth (the important one).** Two clips were cut from a single
master at frame-exact offsets (2.0 s and 1.4 s), so correctly-aligned slots must come
from the *same master frame*. Result: all 68 slot pairs differed by exactly
−0.6000 s, identical dHash, mean pixel difference 0.25/255. Audio sync recovered
that −0.600 s offset from a clap alone at confidence 363.

**Real footage (`cup-1`, see §7).** Portrait HEVC with a −90° display matrix
extracted upright at 1080×2320; 241 frames in 22.6 s; VFR correctly detected;
exposure drift and duplicate frames flagged.

**Cancellation.** Killing a stage that shelled out to a child kills the whole
process tree — verified with `tasklist`, no orphan survived.

**Crash recovery.** A stage left `RUNNING` by a killed server is marked failed on
startup and its scratch discarded.

---

## 6. Expensive lessons — do not re-derive these

**6.1 `fps=...:start_time=` does not phase-shift the sampling grid.** *(the big one)*
It anchors at zero and merely skips earlier frames. Asking for `start_time=0.6` at
6 fps yields frames at 0.667, 0.833… Two cameras with different offsets would each
snap to the same absolute grid and sample **different instants under the same slot
number** — a silent 67 ms misalignment that survives to reconstruction. Fix: shift
the timeline with `setpts=PTS-<start>/TB` *before* `fps`. See `_build_filters`.

**6.2 `-t` with `-copyts` measures from zero,** not from the first frame, silently
truncating the tail. Bound the range with a `trim` filter instead.

**6.3 The `fps` filter with `round=near` needs half an output period of lookahead**
to commit a frame; at end-of-stream it discards the pending one. `timeline.py`
therefore holds the window back by one source frame + `0.5/fps`. Without it you get
one frame fewer than planned and extraction refuses to map slots.

**6.4 OpenMVS prints *nothing* to stdout or stderr** — not even `--help`. The version
banner and all output go only to `<Tool>-<timestamp>.log` written to the **current
working directory**. Hence: every subprocess gets an explicit `cwd`, and `proc.py` is
the only module permitted to import `subprocess` (also because the project path
contains spaces and ` - `, so argv must always be a list with `shell=False`).

**6.5 COLMAP logs to stderr via glog** (`--log_target` defaults to `stderr_and_file`).
Pass `--log_target stdout --log_color 0` from the argv builder, not call sites.
Progress is `[%d/%d]` counters, not percentages.

**6.6 COLMAP 4.2 has *both* option namespaces.** `--FeatureMatching.use_gpu` and
`--SiftMatching.max_ratio` are both valid — pipeline/GPU options moved, algorithm
tuning didn't. Do not blanket-rename. The doctor records which dialect the binary
speaks; read it rather than guessing.

**6.7 Masks are three conventions, not one.** COLMAP wants `<name>.<ext>.png`
(`.png` appended to the *full* filename including `.jpg`); OpenMVS wants
`<name>.mask.png`. Store one canonical form and generate per-engine hardlink views.

**6.8 COLMAP does not undistort masks,** but OpenMVS consumes undistorted images.
Run `image_undistorter` a second time over the mask view with **identical** scale/ROI
flags, then threshold at 127. Drifting flags produce subtly wrong masks exactly where
ears and hair live.

**6.9 Batch escaping.** A PowerShell pipeline inside `for /f` has to survive two
levels of escaping; cmd hands PowerShell a literal `^|` and the check fails oddly.
That logic lives in `scripts/needs-build.ps1` for this reason. Also: this machine has
`NoDefaultCurrentDirectoryInExePath=1`, so `cmd /c launch.bat` fails from a shell —
use `.\launch.bat`. Double-clicking is unaffected.

**6.10 Thresholds were calibrated, not guessed.** Sync confidence: uncorrelated room
tone scores ~7 (the max of a few hundred thousand noise samples always sits several
sigma above their RMS), real claps score 190–310. Threshold is 25. An initial guess of
4 would have called pure noise a confident match.

---

## 7. The smoke test: `cup-1`

**Run:** `D:\pgh-runs\20260907-cup-1-2` · **Source:** `C:\Users\patru\Downloads\20260907_Coffee-Cup-Orbit.mp4`

Handheld single-camera orbit of a mug on a glass patio table. 60.3 s, 1.5 circuits,
HEVC 2320×1080 portrait (−90° display matrix), 29.92 fps, bt709, has audio.
Configured as `camera_orbits`, `revolutions=1.5`, one `SINGLE` segment.

Extract at 4 fps produced **241 frames, correctly oriented**, in 22.6 s.

| Reading | Value | Verdict |
|---|---|---|
| Rotation rate | 9.0 deg/s | Good — well under the 15 deg/s rolling-shutter threshold |
| Sync residual p95 | 0.011 s | Fine (single camera, so not load-bearing) |
| Duplicates | 15 (6%) | Normal; selection will drop them |
| Mean luma range | 17% | **Auto-exposure was hunting** — expect texture seams |
| VFR | suspected | Real phone footage; handled |

**Known weaknesses of this capture, in rough order of severity.** None block using it
as a pipeline test; all matter if the mesh looks wrong.

1. **Glass table.** Transparent and specular. Reflections move with the camera and
   violate the rigid-scene assumption, so expect phantom geometry below the mug.
2. **Exposure hunting** (17%), plus a bright window blowing highlights in part of the
   orbit. Lock AE/AWB next time.
3. **Single elevation ring.** Like the chair spin, this sees one band — the top of the
   mug and the underside of the rim will be weak.
4. **Thin objects** (pens/brushes in the mug) will not reconstruct cleanly.

**Why this run is strategically useful:** it is `camera_orbits`, so **masking is not
required at all**. That means the fastest route to a first end-to-end reconstruction
is **M7 → M8 → M10 → M11 → M12, skipping M9 (SAM 2) entirely.** The whole COLMAP and
OpenMVS chain can be de-risked before any masking code exists. Do that first.

Also on disk: `D:\pgh-test\{master,cam_high,cam_eye}.mp4` — the synthetic
ground-truth pair from §5. Keep them; they are the regression test for slot
alignment. `D:\pgh-runs\20260907-extract-test` is their run.

---

## 8. Next steps

Milestones from the approved plan, unchanged except where the cup run reorders them.

**M7 — Select.** Sharpness (variance of Laplacian, restricted to the subject region,
not the whole frame), drop the already-flagged duplicates, keep angular coverage even,
target-count knob. Contact sheet with accept/reject and a reason per frame; manual
overrides must fold into the stage fingerprint.

**M8 — Align (sparse).** Hardlink farm of selected frames preserving per-camera
folders → `feature_extractor --ImageReader.single_camera_per_folder 1` →
`sequential_matcher` with loop detection → `mapper`. Add
`--TwoViewGeometry.filter_stationary_matches 1`. Report unregistered images. r3f
viewer with camera frusta.

*For a chair spin, deliberately run this without masks first* — it will likely
produce a background-locked model, and seeing that failure in the viewer is what
makes the mask stage's parameters comprehensible. *For `cup-1` it should simply
work*, which is a clean signal that the COLMAP wrapper is sound.

**M10 — Dense.** `image_undistorter` → mask undistortion (§6.8) → `InterfaceCOLMAP`
→ `DensifyPointCloud`. Expose `--resolution-level` prominently: this stage exhausts
the 64 GB of system RAM long before it troubles the 24 GB of VRAM. Offer to delete
depth maps after fusion — they are ~90% of the tens of gigabytes a run consumes and
are never needed again.

**M11 — Mesh.** `ReconstructMesh` → `RefineMesh` → `TextureMesh`. Show texture-shaded,
matte-shaded and wireframe: a surface problem hidden by a convincing texture is the
usual failure.

**M12 — Export.** Headless Blender: largest connected component, decimate, scale from
the measured baseline, orient/centre, GLB/OBJ/STL, turntable render.

**M9 — Mask.** SAM 2.1 large (`sam2.1_hiera_l.yaml` — the pairing is pinned in
`config.py`; a mismatched pair loads *without raising* and produces poor masks).
Click-to-prompt on frame 0 per camera, propagate in timeline order, **chunked at
~150 frames** carrying the last mask forward as the next chunk's seed — the video
predictor's inference state grows with sequence length and will OOM 24 GB otherwise.
Design progress as `chunks × frames` from the start.

---

## 9. Known issues and loose ends

- **`degrees_per_second` needs `capture.revolutions` set by hand.** It stays silent
  otherwise, on purpose — a guessed revolution count produces a confident number
  derived from nothing.
- **dHash is 8×8 and can be fooled by highly regular patterns.** Synthetic `testsrc2`
  colour bars register as 100% duplicates because their 8×8 signature never changes,
  even though full-resolution frames differ. Real footage is unaffected (cup-1: 6%).
- **`opencv-5.0.0-windows.exe`** (186 MB) is still in the project root and unused —
  OpenCV comes from pip. Safe to delete.
- **`openMVS - Source/`** is a source checkout at `develop`, ~7 months ahead of the
  v2.4.0 prebuilt binaries actually in use. Do not mix scene files between versions.
  Only needed if `CreateStructure`/`ExtractKeyframes` are ever wanted.
- **No git repository.** Worth `git init` before the next phase.
- **Sprite sheets are not implemented.** The contact sheet currently loads individual
  thumbnails; fine at 241 frames, will need batching at 3600.
- **Blender is optional in the doctor** and only used at M12.
