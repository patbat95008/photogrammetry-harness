# Photogrammetry Harness — Handover

**Status:** Stages 1, 2, 4 and 5 (ingest → select → align → dense) complete and
verified end to end on real footage. Masking, meshing and export designed, stubbed,
not built.
**Last verified:** 2026-09-07, on the `cup-1` orbit, plus synthetic ground-truth footage.
**Tests:** 182 passing (`.venv\Scripts\python.exe -m pytest server/tests -q`).

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
  ✅         ✅ ⬜         ✅        ✅   ⬜       ⬜
```

Sources are video **or** a folder of photographs; both land in the same slot-indexed
frame layout, and nothing downstream knows which produced them. See §10.

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
scripts/fetch-vocab-tree.ps1   COLMAP vocabulary tree (but read §6.11 first)
data/                          vocabulary tree; machine-local, gitignored
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
  photos.py       reading a folder of stills: natural order, EXIF  ← see §10
  ply.py          point clouds in and previews out  ← see §6.13
  vendor/
    ffmpeg.py     argv builders and probing
    colmap.py     argv builders (dialect-aware, §6.6) + sparse model parsing
    openmvs.py    argv builders, flags read off the v2.4.0 binaries  ← see §6.4
  api/            runs clips stages sync fs artifacts events doctor deps
  stages/
    base.py       Stage contract: params, external_inputs, preflight, run
    shell.py      run_tool, COLMAP counter parsing, OpenMVS log tailing  ← §6.4
    registry.py   StageResolver the staleness engine runs against
    extract.py    Stage 1 ✅ — video frames, and the stills branch (§10)
    select.py     Stage 2 ✅ — sharpness, duplicates, coverage, overrides
    sparse.py     Stage 4 ✅ — COLMAP align
    dense.py      Stage 5 ✅ — undistort, InterfaceCOLMAP, DensifyPointCloud
    planned.py    copy for the three unbuilt stages, varies by capture mode
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
                      (timing fields are null for a photo set — no clock)
thumbs/<group>/       256px webp
select/selection.jsonl  per-frame keep/reject with a reason and a sharpness score
sparse/images/<group>/  hardlink farm of just the selected frames
sparse/model/0/         COLMAP model; model_best in artifacts names the winner
sparse/poses.json       camera intrinsics and positions, for the frusta overlay
sparse/registration.jsonl  which frames registered, and how many points each holds
sparse/preview.ply      normalised cloud the viewer loads
dense/undistorted/      rectified images plus the pinhole model OpenMVS needs
dense/scene_dense.ply   the full dense cloud
dense/preview.ply       decimated to a cap, so the browser can hold it
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

**The full chain, on `cup-1`.** The tower stands up. 241 extracted frames became a
738,015-point dense cloud in which the mug, its handle, the brushes standing in it and
the patio table are all plainly recognisable.

| Stage | Result | Time |
|---|---|---|
| Select | 241 → **183** kept (43 too soft, 15 duplicates) | 3 s |
| Align | **183 of 183 registered**, one model, 50,189 points | 11m 52s |
| Dense | **738,015 points**, 4,033 per view, 2.6 GB of depth maps reclaimed | 3m 47s |

Align in detail: mean reprojection error **1.162 px**, mean track length **6.04**, a
single submodel — the 1.5-revolution orbit closed. The camera trajectory is a smooth
unbroken arc with no jumps. COLMAP started from a guessed focal length of 2784 px
(1.2 × the long edge, because video frames carry no EXIF) and bundle adjustment pulled
it to **1683 px**, a 40% correction it had to find unaided. That is the strongest
argument for the stills path in §10.

Everything §7 predicted about this capture showed up: the glass table produced a haze
of phantom points beneath the mug, and the brushes came out as streaks rather than
cylinders. Neither is a bug.

Timings are for 183 images at 1080×2320 on the 4090. Matching was exhaustive (§6.11),
and at 6.3 minutes it is over half the total.

**Cancellation, again, for silent children.** Cancelling a stage whose child prints
nothing — the OpenMVS case — kills the whole process tree and leaves no orphan, now
covered by a test that uses a deliberately silent child rather than one that narrates.

**Staleness, on the live run.** Change `resolution_level`, dense goes stale; change it
back, dense returns to done without re-running. Change `target_count` on select and
align and dense both go stale citing "upstream"; revert it and all three come back.
The cosmetic `preview_point_cap` invalidates nothing.

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

**6.11 The published COLMAP vocabulary trees do not work with this build.** *(cost a
full matching run to find)* COLMAP replaced FLANN with faiss for its visual index in
May 2025; the trees still served from demuc.de are the old format. Handing one to
4.2.0 does not produce an error — it aborts the process with
`STATUS_STACK_BUFFER_OVERRUN` (`0xC0000409`) partway through matching, *after*
feature extraction has been paid for. Four bytes distinguish them: the faiss-era file
opens with a version field of 1 or 2, a FLANN tree with its word count (32762 for the
32K tree). `sparse.vocab_tree_status()` checks that before use and the matcher falls
back to exhaustive; the doctor reports which you have. No faiss-format tree has been
published yet.

**6.12 Exhaustive matching is fine at this scale, and cannot miss a loop.** 183 frames
is 16,653 pairs and took 6.3 minutes on the 4090 — against a whole-run budget measured
in tens of minutes, that is not the bottleneck. It also closes the orbit by
construction, which is the thing loop detection exists to do. Reach for sequential
matching when frame counts climb past roughly 500, not before.

**6.13 OpenMVS point clouds have variable-length records.** `scene_dense.ply` carries
a per-point *list* of the views that saw it (`property list uchar int views`), so the
record length differs from point to point and nothing can be memory-mapped. Reading it
at a fixed stride does not fail — it returns normals and view indices interpreted as
coordinates, which looks like a plausible cloud and is not one. `ply.py` detects the
list in the header and walks records individually in that case, and both stages write
a **normalised preview** (`x y z` float32 + `rgb` uchar, decimated to a cap) beside the
full file so three.js's stock `PLYLoader` never has to guess a layout. The dense stage
logs the header before parsing, because scratch is discarded on failure and that log
line is then the only surviving evidence of what the engine wrote.

**6.14 COLMAP's output arrives in bursts, not a stream.** glog block-buffers when
stdout is a pipe rather than a console, so a matcher that ran for six minutes delivered
its entire log in one flush at exit. `[%d/%d]` counters are still parsed and still
correct, but the progress bar jumps rather than climbs, and a quiet stage is not
evidence of a stuck one. Judge liveness from the process and the GPU, not the log.

**6.15 Cancellation cannot be driven by output.** `extract.py` checks the cancel token
inside its line callback, which works only because ffmpeg narrates constantly. OpenMVS
prints *nothing* and COLMAP goes quiet through bundle adjustment, so for those a
per-line check never fires and a forty-minute densify would be uncancellable while
holding the GPU. Everything that shells out therefore goes through
`stages/shell.py:run_tool`, which drives `proc.stream_with_cancel` from
`ctx.proc_cancel` — a `threading.Event` watched on its own thread. `jobs.py` had
created that event all along but never passed it to the stage.

**6.16 Tail a log in binary, not text.** The OpenMVS log tail seeks to a byte offset
each poll. `TextIOWrapper.seek()` only accepts opaque cookies from `tell()`; a plain
integer raises, which killed the tailing thread silently and froze progress for the
rest of the run. Opening the file `"rb"` and decoding per line fixes it. A partial
final line is held back and re-read next pass, so nothing is delivered twice.

**6.17 Reading `project.json` from a second handle can break the writer.** The manifest
is saved by writing a temporary file and `os.replace`-ing it into place, which fails on
Windows with `PermissionError` if anything else holds the target open. The server is
safe because `RunHandle.load()` returns a cached object, but a script that polls
`run.io.load()` in a loop will intermittently break the stage it is watching. Poll
`run.load()`, or the event bus.

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

**Why this run was strategically useful:** it is `camera_orbits`, so masking was not
required at all, and M9 could be skipped entirely — which is how the whole COLMAP and
OpenMVS chain got de-risked before any masking code existed. That worked; §5 has the
numbers. Masking is now skipped on this run rather than pending, which is a state the
manifest records explicitly (see `StageState.SKIPPED`).

Also on disk: `D:\pgh-test\{master,cam_high,cam_eye}.mp4` — the synthetic
ground-truth pair from §5. Keep them; they are the regression test for slot
alignment. `D:\pgh-runs\20260907-extract-test` is their run.

---

## 8. Next steps

M7, M8 and M10 are built (§5). Three milestones remain, in the order they should be
taken.

**M11 — Mesh.** `ReconstructMesh` → `RefineMesh` → `TextureMesh`, following the
pattern in `stages/dense.py`: the OpenMVS argv builders live in `vendor/openmvs.py`
and progress comes from tailing the tool's log (§6.4). Read the flags off the v2.4.0
binaries with `--help` before writing the builders, exactly as was done for the two
tools already wrapped — the source checkout in this repo is months ahead of them.

Show texture-shaded, matte-shaded and wireframe: a surface problem hidden by a
convincing texture is the usual failure. The existing `PointCloudViewer` covers point
clouds only, so this needs a mesh viewer beside it rather than a change to it.

**M12 — Export.** Headless Blender: largest connected component, decimate, scale from
the measured baseline, orient and centre, then GLB/OBJ/STL plus a turntable render.
Orientation matters more than it sounds — the cup model comes out in an arbitrary
frame, so "which way is up" is not recoverable from the reconstruction alone.

**M9 — Mask.** SAM 2.1 large (`sam2.1_hiera_l.yaml` — the pairing is pinned in
`config.py`; a mismatched pair loads *without raising* and produces poor masks).
Click-to-prompt on frame 0 per camera, propagate in timeline order, **chunked at
~150 frames** carrying the last mask forward as the next chunk's seed — the video
predictor's inference state grows with sequence length and will OOM 24 GB otherwise.
Design progress as `chunks × frames` from the start.

Three things are already waiting for it:

- `StageState.SKIPPED` and the skip button exist so a run can proceed without masks;
  un-skipping is what should bring this stage back.
- `SparseStage.external_inputs` already reports whether masks exist, so producing
  them correctly invalidates the alignment rather than leaving it falsely fresh.
- **`DenseStage.preflight` refuses to run if masks are present**, because
  undistorting them (§6.8) is not implemented. That block is deliberate and must be
  lifted as part of this milestone, not before it.

*For the chair spin, deliberately run alignment without masks first.* It will very
likely produce a background-locked model, and seeing that failure in the viewer is
what makes this stage's parameters comprehensible. The align stage already emits a
warning saying so when the capture mode is `subject_rotates` and masking was skipped.

**Also worth doing at some point**

- **The viewer has not been confirmed by eye in a normal browser.** It was verified
  programmatically — canvas sized, cloud and camera count loaded, geometry correct —
  but the embedded browser used during development never fired the ResizeObserver the
  renderer measures itself with, so nothing was ever seen rendered. Open
  `/runs/<id>/stages/dense` in Chrome and look. `scripts/` has no equivalent; the
  offline renders in §5 were made with a throwaway script.
- **Rigs (§4.1).** `sequential_matcher` in COLMAP 4.2 has `expand_rig_images`, and
  `feature_matcher` has `rig_verification` and `skip_image_pairs_in_same_frame`. The
  slot naming that makes this possible is already in place.
- **Sprite sheets.** The contact sheet loads individual thumbnails; fine at 241
  frames, will need batching at 3600.

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
- **Never poll `run.io.load()`** while a stage is running: it reads `project.json`
  from disk, and an open handle breaks the writer's atomic replace on Windows
  (§6.17). Use `run.load()`, which is cached, or the event bus.
- **`openMVS - Source/`** is a source checkout at `develop`, ~7 months ahead of the
  v2.4.0 prebuilt binaries actually in use. Do not mix scene files between versions.
  Only needed if `CreateStructure`/`ExtractKeyframes` are ever wanted.
- **The vocabulary tree in `data/` is unusable** and matching falls back to
  exhaustive. Not a defect — see §6.11. The doctor reports it.
- **COLMAP progress arrives in bursts** because glog block-buffers to a pipe (§6.14),
  so the progress bar jumps rather than climbs. A quiet stage is not a stuck one.
- **The dense stage refuses to run when masks are present** (§6.8), by design, until
  mask undistortion exists. See §8.
- **The 3D viewer has not been seen rendering in a normal browser.** Verified
  programmatically only; see §8.
- **`model_analyzer` contributed nothing on the cup run** —
  `mean_observations_per_image` came back null, so the metrics fall back to the values
  parsed from the text model. Harmless, and the numbers that matter are all present,
  but the parser expects a format the binary may not be emitting.
- **Sprite sheets are not implemented.** The contact sheet currently loads individual
  thumbnails; fine at 241 frames, will need batching at 3600.
- **Blender is optional in the doctor** and only used at M12.

---

## 10. The stills path

A run's source is either video clips or a folder of photographs, chosen at the top of
the Clips page. Both produce the same thing:

```
frames/<camera_group>/<slot:06d>.jpg   +   frames/frames.jsonl
```

and nothing downstream knows which branch ran. It is an ingest branch inside the
extract stage rather than a stage of its own, precisely so that stays true.

**Why it exists.** A folder of stills is the cleanest input this pipeline can be
given: no rolling shutter, no variable frame rate, no display matrix, no inter-frame
compression, and real EXIF. When a video run produces a bad model, a photo run
separates "the reconstruction chain is wrong" from "extraction is feeding it bad
frames" — which is the question you cannot otherwise answer without suspecting
everything at once.

**Photographs are hardlinked, not copied or re-encoded**, whenever nothing needs to
change. That is instant and costs no disk, but the real reason is EXIF: COLMAP seeds
each camera's focal length from it and otherwise falls back to guessing
`1.2 × max(width, height)`. On the cup video, which has no EXIF, that guess was
2784 px and bundle adjustment pulled it down to 1683 px — a 40% correction it had to
find for itself.

Two things force a re-encode, and both are reported in the metrics as `reencoded`:

- an image larger than `max_dimension`;
- an image carrying an **EXIF orientation flag**, which is applied to the pixels and
  then cleared. COLMAP does not honour that flag reliably, and a silently sideways
  subset of a shoot is the kind of failure that costs an afternoon. EXIF is copied
  across on both paths.

**Ordering is natural, not lexical** — `IMG_9` precedes `IMG_10`. Sequential matching
compares frames by index, so index order has to be viewpoint order; plain string sort
scrambles it in a way nothing downstream can detect.

**What photographs do not have is a clock.** `src_pts`, `timeline_time_s` and
`sync_residual_s` are `null` in `frames.jsonl` rather than zero, and the timeline
carries only `total_slots`. Writing 0.0 where there is no timestamp would put a number
that means nothing into the one file every later stage reads. A run therefore cannot
mix photo and video clips, and preflight refuses it: the shared slot clock that pairs
two cameras instant-by-instant comes from video timestamps, and half an invented
timeline is worse than none.

RAW and HEIC need a decoder that is not installed; the scanner names the files it
could not read rather than skipping them quietly. `docs/capture-checklist.md` has the
shooting guidance.
