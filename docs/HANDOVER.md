# Photogrammetry Harness — Handover

**Status:** The whole chain, ingest → export, is **built and verified end to end on
real footage**. Only masking (M9) remains.
**Last verified:** 2026-09-07 on the `cup-1` orbit, plus synthetic ground-truth footage.
**Tests:** 279 passing (`.venv\Scripts\python.exe -m pytest server/tests -q`).

---

## 1. What this is

A local web harness that turns video — or a folder of photographs — into a 3D model,
built around existing reconstruction engines rather than replacing them. The end goal is
a printable 3D bust of the owner's head, captured by rotating in an office chair while
two phone cameras record from different heights.

The harness owns the parts those engines do badly: ingest, frame selection, masking,
quality control, and making failures legible. COLMAP and OpenMVS do the actual
reconstruction.

```
FFmpeg / stills → OpenCV/SAM 2 → COLMAP → OpenMVS → Blender
ingest            select/mask     align    dense/mesh  cleanup
  ✅                ✅ ⬜          ✅       ✅   ✅       ✅
```

**The whole tower has now been proven.** A 60-second handheld orbit of a coffee cup
became a 738,015-point dense cloud in which the mug, its handle, the brushes standing in
it and the patio table are all plainly recognisable. §5 has the numbers, §7 has the
capture. Everything from here is refinement of a chain that works.

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
(gyan full build, on PATH), COLMAP 4.2.0 CUDA (commit be5e291), OpenMVS 2.4.0 CUDA
prebuilt (built Jan 2026), Blender 5.2 LTS, torch 2.14.0+cu130 on an RTX 4090 (24 GB),
all four SAM 2.1 checkpoints. i9-9900K, 64 GB RAM. Python 3.12.9 in `.venv`. Runs live
in `D:\pgh-runs`.

**The 3D viewers work in Firefox.** They do *not* render in some embedded/automation
browsers, where `ResizeObserver` never fires and the canvas stays at its 300×150
default. If a cloud ever looks blank, check it in a real browser before concluding the
reconstruction is empty.

---

## 3. Where things are

```
launch.bat  dev.bat  requirements.txt
scripts/needs-build.ps1        UI freshness check (see §6.9)
scripts/fetch-vocab-tree.ps1   COLMAP vocabulary tree — but read §6.11 first
data/                          vocabulary tree; machine-local, gitignored
docs/capture-checklist.md      read before filming — capture decides most outcomes
docs/HANDOVER.md               this file

server/pgh/
  config.py       tool paths, SAM2 checkpoint↔config pairs, ingest roots
  registry.py     doctor probes: versions, CUDA, COLMAP dialect, vocabulary tree
  manifest.py     project.json models — StageId/State, CaptureMode, Segment, Clip
  fingerprint.py  the staleness engine  ← load-bearing, read this first
  timeline.py     shared slot clock across clips  ← the core of video ingest
  jobs.py         single-worker queue, per-stage logs, lifecycle events
  proc.py         the ONLY module allowed to import subprocess (see §6.4)
  events.py       append-only event log + SSE bus
  store.py        run directories, crash recovery
  sync.py         audio cross-correlation
  files.py        source identity, ingest-root guard
  photos.py       reading a folder of stills: natural order, EXIF  ← §10
  ply.py          point clouds in, browser-sized previews out; face counts  ← §6.13
  orient.py       up axis from the camera plane, scale from the rig baseline
  vendor/
    ffmpeg.py     argv builders and probing
    colmap.py     dialect-aware argv builders (§6.6) + sparse model parsing
    openmvs.py    argv builders, flags read off the v2.4.0 binaries  ← §6.4
    blender.py    headless invocation and the PGH_RESULT protocol
  blender/
    export_mesh.py  runs INSIDE Blender: no pgh, no numpy  ← §6.25
  api/            runs clips stages sync fs artifacts events doctor deps
  stages/
    base.py       Stage contract: params, external_inputs, preflight, run
    shell.py      run_tool, COLMAP counters, OpenMVS log tailing, PeakMemory  ← §6.4
    registry.py   StageResolver the staleness engine runs against
    extract.py    Stage 1 ✅ — video frames, and the stills branch (§10)
    select.py     Stage 2 ✅ — sharpness, duplicates, coverage, manual overrides
    sparse.py     Stage 4 ✅ — COLMAP align
    dense.py      Stage 5 ✅ — undistort, InterfaceCOLMAP, DensifyPointCloud
    mesh.py       Stage 6 ✅ — ReconstructMesh, RefineMesh, TextureMesh  ← §6.19-6.24
    export.py     Stage 7 ✅ — orient, scale, clean and write, via headless Blender
    planned.py    copy for the mask stage, which is the only one left
web/src/          Vite + React + TS. StageShell auto-builds param forms from the
                  pydantic JSON schema — a new stage gets a working UI for free.
  components/viewer3d.tsx           framing, sizing and the flip, shared by both viewers
  components/PointCloudViewer.tsx   points, frusta and trajectory: sparse and dense
  components/MeshViewer.tsx         surfaces: textured, matte and wireframe
  components/ExportPage.tsx         turntable, dimensions and downloads
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
select/selection.jsonl  per-frame keep/reject, with a reason and a sharpness score
sparse/images/<group>/  hardlink farm of just the selected frames
sparse/model/0/         COLMAP model; artifacts["model_best"] names the winner
sparse/poses.json       intrinsics and camera positions, for the frusta overlay
sparse/registration.jsonl  which frames registered, and how many points each holds
sparse/preview.ply      normalised cloud the viewer loads
dense/undistorted/      rectified images plus the pinhole model OpenMVS needs
dense/scene_dense.ply   the full dense cloud
dense/preview.ply       decimated to a cap, so the browser can hold it
mesh/mesh.ply           the raw surface, untextured — count components on THIS (§6.23)
mesh/mesh_textured.glb  the painted mesh, plus mesh_textured_0.png beside it (§6.24)
export/model.glb        the finished object: cleaned, stood up, scaled
export/model.stl        the same, for a slicer -- carries no units, so see §8 on scale
export/turntable/       36 PNGs orbiting what was actually exported
export/report.json      metrics and warnings, as written
logs/                 per-stage run logs
.partial/             stage scratch; committed by atomic rename, wiped on failure
```

`D:\pgh-runs` rather than the project folder is deliberate: the dense stage builds paths
like `dense/undistorted/stereo/depth_maps/cam_high/000042.jpg.geometric.bin`, and COLMAP
and OpenMVS may not honour long-path support even though Windows has it enabled here.

---

## 4. The ideas everything else hangs off

### 4.1 Slot naming is a shared clock

`frames/cam_high/000042.jpg` and `frames/cam_eye/000042.jpg` are **the same instant**.
Not the same source frame number — the same moment in time.

This exists because two cameras on fixed mounts, in the *subject's* reference frame,
orbit the head in lockstep at a constant relative pose. That is a rigid two-sensor rig,
and COLMAP 4.2 can exploit it (`rig_configurator`,
`--Mapper.ba_refine_sensor_from_rig`), which constrains bundle adjustment hard and helps
the orbit close. COLMAP groups rig frames by **matching filename suffix across
per-camera folders**, so the naming *is* the mechanism.

Nothing depends on it yet — rigs are still unbuilt (§8) — but ingest was built so that
adding them later needs no re-extraction. `timeline.py` is where this lives.

Segments carry it: clips recorded simultaneously share a segment and a timeline; a
separate pass (the crown shot) gets its own disjoint block of slot indices.

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

The align stage already emits a warning when the mode is `subject_rotates` and masking
was skipped, because a *confident* model is the failure case there.

### 4.3 Staleness, and why it's pleasant

`fingerprint.py` hashes each stage from its params (minus cosmetic ones), its declared
external inputs, and its upstream stages' fingerprints. Because it is content-derived
rather than a run counter, **changing a param and changing it back leaves downstream
work valid.** Verified on the live cup run, not just in tests.

Mark a param cosmetic with `json_schema_extra={"affects_fingerprint": False}` —
`thumbnail_px` and `preview_point_cap` are the examples.

Two traps, both of which threw away hours of valid work before being fixed, and both now
covered by tests:

- **Params are normalised through their model before hashing.** A stage run before
  anyone opened its page stores `{}`; the moment the UI saves that same form it stores
  every field. Hashing the raw dict meant that merely pressing Save invalidated
  everything downstream.
- **Only add keys to `external_inputs` conditionally.** Adding one unconditionally
  changes the hash of every run ever made. `extract._clip_inputs` emits the photo-set
  keys only for a photo set, for exactly this reason.

`StageState.SKIPPED` is a fourth state alongside done/stale/failed: a deliberate decision
not to run something, which dependents treat as satisfied. Only masking is skippable
(`api/stages.py:SKIPPABLE`).

### 4.4 Adding a stage

Server side is three edits:

1. New `server/pgh/stages/<name>.py`: a `StageParams` subclass, a `Stage` subclass with
   `id`/`label`/`description`/`params_model`, then `external_inputs`, `preflight`,
   `run`. Keep the class short and put the work in module-level free functions —
   `extract.py` and `dense.py` are the shape to copy.
2. Import and `registry.register(YourStage())` in `stages/__init__.py`.
3. Delete its entry from `stages/planned.py`.

Inside `run()`:

- Write into `ctx.scratch/<name>/`, then commit by `rmtree(target)` +
  `scratch.replace(target)`. The framework wipes scratch on failure, so a crashed stage
  leaves nothing half-written — **and leaves nothing to inspect either**, which is why
  `dense.py` logs the PLY header before parsing it.
- Shell out through `stages/shell.py:run_tool`, never `proc` directly. It handles
  cancellation for silent children (§6.15) and raises with the tool named, and that
  message is what the user reads on the stage page.
- Bulk per-item data goes in a `.jsonl` sidecar, never in `metrics` — `project.json` is
  rewritten on every change.
- Warnings are full prose sentences naming the cause *and* the consequence. That house
  style is consistent everywhere; match it.

Client side is two edits: a page component wrapping `StageShell` with a render-prop
results view, and one entry in `BUILT_STAGES` in `App.tsx`. A new SSE event type must
also be added to the `types` array in `useEventStream.ts` or it is dropped silently.

---

## 5. What's verified, and how

**Synthetic ground truth.** Two clips were cut from a single master at frame-exact
offsets (2.0 s and 1.4 s), so correctly-aligned slots must come from the *same master
frame*. Result: all 68 slot pairs differed by exactly −0.6000 s, identical dHash, mean
pixel difference 0.25/255. Audio sync recovered that −0.600 s offset from a clap alone at
confidence 363.

**The full chain, on `cup-1`.**

| Stage | Result | Time |
|---|---|---|
| Extract | 241 frames at 4 fps, correctly oriented from a −90° display matrix | 22.6 s |
| Select | 241 → **183** kept (43 too soft, 15 duplicates) | 3.1 s |
| Align | **183 of 183 registered**, one model, 50,189 points | 11m 52s |
| Dense | **738,015 points**, 4,033 per view, 2.6 GB of depth maps reclaimed | 3m 47s |
| Mesh | **497,926 triangles** over 249,022 vertices, textured to one 4096² atlas | 4m 55s |
| Export | cleaned to **495,490** in one shell, stood up, GLB + STL + 36 turntable frames | 16.3 s |

Align in detail: mean reprojection error **1.162 px**, mean track length **6.04**, a
single submodel — the 1.5-revolution orbit closed, and the camera trajectory is a smooth
unbroken arc. COLMAP started from a guessed focal length of 2784 px (1.2 × the long edge,
because video frames carry no EXIF) and bundle adjustment pulled it to **1683 px**
unaided. That is the strongest argument for the stills path (§10).

Dense peaked at 19.1 GB of system RAM at `resolution_level 1`. Matching was exhaustive
(§6.11) and at 6.3 minutes was over half the total wall clock.

Everything §7 predicted about this capture showed up: the glass table produced a haze of
phantom points beneath the mug, and the brushes came out as streaks rather than
cylinders. Neither is a bug.

Export in detail: the up axis came out of a plane fitted to 183 camera centres at
planarity **0.043** — convincingly planar — and cross-checks against two independent
estimators. It sits **9.3°** from the mean of the cameras' own up vectors, and **7°** from
the normal of a plane fitted to the tabletop geometry, neither of which the fit uses. The
cleanup healed 88,778 seam-split vertices, found 7 connected shells, and dropped 2,436
faces of floating debris to keep 495,490 in one piece. `cup-1` is a single camera, so
there is no rig baseline and the export is in model units — which the stage says in a
warning rather than implying a size it never established.

Mesh in detail: reconstruction took 27.8 s and texturing 264.1 s, of which **4m 2s was
the single "assigning the best view to each face" phase** — which is why the progress
spans weight texturing at roughly ten times reconstruction, and why the heartbeat exists.
Peak system memory 21.2 GB, refinement off. 675 triangles per thousand dense points is
the ratio to expect; the mesh follows the cloud almost exactly, which was confirmed by
projecting both onto the same three planes and comparing. That also means it inherits
everything the glass table did to the cloud: what comes out is a faithful mesh of a
tabletop with a mug on it, and the mug is a small part of it. **`cup-1` is a good
pipeline test and a poor quality test** — judge reconstruction quality on the turnaround
footage when it exists, not on this.

**Cancellation.** Killing a stage that shelled out to a child kills the whole process
tree — verified with `tasklist`, no orphan survived. Now also covered for a child that
produces *no output at all*, which is the OpenMVS case and the one a per-line check
cannot handle (§6.15).

**Crash recovery.** A stage left `RUNNING` by a killed server is marked failed on startup
and its scratch discarded.

**Staleness, on the live run.** Change `resolution_level`, dense goes stale; change it
back, dense returns to done without re-running. Change `target_count` on select and both
align and dense go stale citing "upstream"; revert it and all three come back. The
cosmetic `preview_point_cap` invalidates nothing.

**Stills ingest.** Twelve synthetic photographs with EXIF: natural order preserved
(`IMG_9` before `IMG_10`), eleven hardlinked with EXIF byte-identical, the one carrying a
rotation flag physically rotated 320×240 → 240×320 with the flag cleared and its focal
length preserved.

---

## 6. Expensive lessons — do not re-derive these

**Numbering is stable — code comments cite these by number** (§6.4, §6.5, §6.7 and §6.8
are referenced from `shell.py`, `colmap.py`, `openmvs.py` and `dense.py`). Append, don't
renumber.

### Ingest

**6.1 `fps=...:start_time=` does not phase-shift the sampling grid.** *(the big one)* It
anchors at zero and merely skips earlier frames. Asking for `start_time=0.6` at 6 fps
yields frames at 0.667, 0.833… Two cameras with different offsets would each snap to the
same absolute grid and sample **different instants under the same slot number** — a
silent 67 ms misalignment that survives to reconstruction. Fix: shift the timeline with
`setpts=PTS-<start>/TB` *before* `fps`. See `_build_filters`.

**6.2 `-t` with `-copyts` measures from zero,** not from the first frame, silently
truncating the tail. Bound the range with a `trim` filter instead.

**6.3 The `fps` filter with `round=near` needs half an output period of lookahead** to
commit a frame; at end-of-stream it discards the pending one. `timeline.py` therefore
holds the window back by one source frame + `0.5/fps`. Without it you get one frame fewer
than planned and extraction refuses to map slots.

### Talking to the engines

**6.4 OpenMVS prints *nothing* to stdout or stderr** — not even `--help`. The version
banner and all output go only to `<Tool>-<timestamp>.log` written to the **current
working directory**. Three consequences: every subprocess gets an explicit `cwd`;
`proc.py` is the only module permitted to import `subprocess` (also because the project
path contains spaces and ` - `, so argv must always be a list with `shell=False`); and
progress has to come from tailing that log, which is what `shell.py:OpenMVSLog` does.

**6.5 COLMAP logs to stderr via glog** (`--log_target` defaults to `stderr_and_file`).
Pass `--log_target stdout --log_color 0` from the argv builder, not call sites. Progress
is `[%d/%d]` counters, not percentages.

**6.6 COLMAP 4.2 has *both* option namespaces.** Verified against the binary in `tools/`:
`--FeatureExtraction.max_image_size` and `--SiftExtraction.max_num_features` are *both*
correct at the same time — pipeline and GPU options moved, algorithm tuning did not — and
`filter_stationary_matches` lives on `TwoViewGeometry`. Do not blanket rename.
`registry.colmap_dialect()` records which dialect the binary speaks; read it rather than
guessing. `tests/test_colmap_argv.py` asserts this against both dialects.

**6.7 Masks are three conventions, not one.** COLMAP wants `<name>.<ext>.png` (`.png`
appended to the *full* filename including `.jpg`); OpenMVS wants `<name>.mask.png`. Store
one canonical form and generate per-engine hardlink views. Note that `DensifyPointCloud`
has a native `-m/--mask-path` expecting the OpenMVS form, plus `--ignore-mask-label`.

**6.8 COLMAP does not undistort masks,** but OpenMVS consumes undistorted images. Run
`image_undistorter` a second time over the mask view with **identical** scale/ROI flags,
then threshold at 127. Drifting flags produce subtly wrong masks exactly where ears and
hair live. **Not implemented — `DenseStage.preflight` blocks rather than ignoring masks.**

**6.9 Batch escaping.** A PowerShell pipeline inside `for /f` has to survive two levels of
escaping; cmd hands PowerShell a literal `^|` and the check fails oddly. That logic lives
in `scripts/needs-build.ps1` for this reason. Also: this machine has
`NoDefaultCurrentDirectoryInExePath=1`, so `cmd /c launch.bat` fails from a shell — use
`.\launch.bat`. Double-clicking is unaffected.

**6.10 Thresholds were calibrated, not guessed.** Sync confidence: uncorrelated room tone
scores ~7, real claps score 190–310. Threshold is 25. An initial guess of 4 would have
called pure noise a confident match.

**6.11 The published COLMAP vocabulary trees do not work with this build.** *(cost a full
matching run to find)* COLMAP replaced FLANN with faiss for its visual index in May 2025;
the trees still served from demuc.de are the old format. Handing one to 4.2.0 does not
produce an error — it aborts the process with `STATUS_STACK_BUFFER_OVERRUN`
(`0xC0000409`) partway through matching, *after* feature extraction has been paid for.
Four bytes distinguish them: the faiss-era file opens with a version field of 1 or 2, a
FLANN tree with its word count (32762 for the 32K tree). `sparse.vocab_tree_status()`
checks that before use and the matcher falls back to exhaustive; the doctor reports which
you have. No faiss-format tree has been published yet.

**6.12 Exhaustive matching is fine at this scale, and cannot miss a loop.** 183 frames is
16,653 pairs and took 6.3 minutes on the 4090. It closes the orbit by construction, which
is the thing loop detection exists to do. Reach for sequential matching when frame counts
climb past roughly 500, not before.

**6.13 OpenMVS point clouds have variable-length records.** `scene_dense.ply` carries
*two* per-point lists — `view_indices` and `view_weights` — and puts colour *before* the
normals. Record length therefore differs from point to point and nothing can be
memory-mapped; reading at a fixed stride does not fail, it returns normals and view
indices interpreted as coordinates, which looks like a plausible cloud and is not one.
`ply.py` detects lists in the header and walks records individually in that case, and
works by property *name* throughout. Both stages write a **normalised preview** (`x y z`
float32 + `rgb` uchar, decimated to a cap) beside the full file so three.js's stock
`PLYLoader` never has to guess a layout.

**6.14 COLMAP's output arrives in bursts, not a stream.** glog block-buffers when stdout
is a pipe rather than a console, so a matcher that ran for six minutes delivered its
entire log in one flush at exit. Counters are still parsed and still correct, but the
progress bar jumps rather than climbs, and a quiet stage is not evidence of a stuck one.
Judge liveness from the process and the GPU, not the log.

### The harness itself

**6.15 Cancellation cannot be driven by output.** `extract.py` checks the cancel token
inside its line callback, which works only because ffmpeg narrates constantly. OpenMVS
prints nothing and COLMAP goes quiet through bundle adjustment, so for those a per-line
check never fires and a forty-minute densify would be uncancellable while holding the
GPU. Everything that shells out goes through `stages/shell.py:run_tool`, which drives
`proc.stream_with_cancel` from `ctx.proc_cancel` — a `threading.Event` watched on its own
thread. `jobs.py` had created that event all along but never passed it to the stage.

**6.16 Tail a log in binary, not text.** The OpenMVS log tail seeks to a byte offset each
poll. `TextIOWrapper.seek()` only accepts opaque cookies from `tell()`; a plain integer
raises, which killed the tailing thread silently and froze progress for the rest of the
run. Open `"rb"` and decode per line. A partial final line is held back and re-read next
pass, so nothing is delivered twice.

**6.17 Reading `project.json` from a second handle can break the writer.** The manifest is
saved by writing a temporary file and `os.replace`-ing it into place, which fails on
Windows with `PermissionError` if anything else holds the target open. The server is safe
because `RunHandle.load()` returns a cached object — but a script that polls
`run.io.load()` in a loop will intermittently break the stage it is watching. Poll
`run.load()`, or the event bus. **Corollary:** while the server is running it owns the
cached manifest, so editing `project.json` from a script is silently overwritten by the
server's next write. Stop the server first.

**6.18 `JobState.SUCCEEDED` is set just before the manifest is committed.** A caller
polling job state can therefore read the record before the results land, and
`runner.stop()` signals rather than joining. Wait on the stage record or the
`stage.finished` event instead.

---

**6.19 Image paths inside a `.mvs` are relative to the *working folder*.**
`scene_dense.mvs` names its photographs `undistorted/images/cup/000019.jpg`, and OpenMVS
resolves that against `-w`, not against the directory the scene file sits in — confirmed
by reading the bytes of a real one. `TextureMesh` is where it bites, being the only mesh
tool that reads pixels, and it reports a missing image for a file that is plainly there.
`mesh.py` hardlinks `dense/undistorted` into its scratch and makes *that* the working
folder rather than pointing `-w` at the committed `dense/`, because each tool also drops
its log into the working folder and a failed run would otherwise litter a finished
stage's output.

**6.20 An OpenMVS percentage is far more often a statistic than progress.** `PERCENT_RE`
was `(\d{1,3})%`, which reads `35867 points inside ROI (71.46%)` as **46%** — it matches
the two digits before the sign rather than the number they belong to. Worse, v2.4.0 emits
no marching percentage at all: of the 1,144 lines the cup densify wrote, six carried a
percent sign and every one was a statistic. So the dense stage's bar jumped to ~0.46 in
its first second and sat there for the remaining 3m32s. The regex is fixed, but the real
lesson is the design one: drive progress from a table of phase markers with a monotonic
floor, and beat a heartbeat with elapsed time underneath it, so a tool that says nothing
for four minutes reads as working rather than wedged.

**6.21 The mesh is written to `<-o with its extension stripped> + <export type>`, and the
`.mvs` that `-o` names may never be written at all.** It is skipped when the archive type
is the default and the scene was loaded in interface format — exactly this pipeline:
`-o mesh.ply` produced `mesh.ply` and no `mesh.mvs`. The extension is chosen by a
different flag from the one that names the file, and nothing may chain on `-o`; each tool
gets the original `dense/scene_dense.mvs` plus `-m <mesh>`. Discover the output, as
`dense.py:_find_cloud` and `mesh.py:_find_mesh` both do.

**6.22 `dense/scene_dense.mvs` carries only the *sparse* cloud.** Its own save line reads
`50189 points, 0 vertices, 0 faces`; the 738,015 dense points live beside it in
`scene_dense.ply`, with their per-point `view_indices`. So `ReconstructMesh -p` is
mandatory rather than optional — without it the graph cut runs happily on a fifteenth of
the data and nothing in the log calls it a mistake.

**6.23 A textured GLB's connected components are texture seams, not geometry.**
`TextureMesh` splits vertices at every atlas patch boundary, so the cup mesh reports
**11,015 loose parts** as a GLB — against TextureMesh's own 10,848 patches — with the
largest at 0.5%, and **7 components with 99.5% in one piece** as `mesh.ply`. Blender
imports the GLB with 337,819 vertices against the PLY's 249,022 for the same surface. The
export stage's "largest connected component" step must therefore run on
`mesh_untextured`, or merge by distance first; run on the GLB it keeps 0.5% of the model
and silently throws the subject away.

**6.24 A `.glb` from OpenMVS is not self-contained.** It writes the atlas beside it as
`<stem>_0.png` and references it by relative URI, so the mesh is two files. Both are
recorded as artifacts and both must be served — which works because they land in the same
directory, so `GLTFLoader` resolves the sidecar against the mesh's own URL. Confirmed in
the browser: loading the mesh fetches the `.png` too.

**6.25 Blender's glTF importer converts axes into the vertex data, not the transform.**
glTF is Y-up and Blender is Z-up, so the importer rewrites every vertex as `(x, -z, y)`
on the way in — and leaves `matrix_world` as the identity, so there is nothing at runtime
to read the conversion back off. An up axis computed from the camera poses and applied to
that mesh is therefore 90° out. This cost two wrong fixes before the right one: carrying
the axis through `matrix_world` does nothing (it is identity), and carrying it through
after correcting the object transform double-applies the conversion, because
`matrix_world` maps the mesh's *data* coordinates into the world and the axis was never in
data coordinates. What works is `export_mesh.py:GLTF_TO_RECONSTRUCTION`: undo the
conversion at import so world coordinates *are* reconstruction coordinates, then use the
axis untouched. OBJ and PLY are told `forward_axis="Y", up_axis="Z"` so they convert
nothing in the first place.

**The reason this was worth chasing:** the model exported standing on its edge while
every number in the report — 7 components, 2,436 faces dropped, planarity 0.043 — looked
entirely reasonable. Nothing but looking at it would have caught it, which is what the
orthographic top/front/side check in §5 is for.

**6.26 OpenMVS writes no vertex normals, and a lit material without them renders black.**
The glTF from `TextureMesh` carries `POSITION` and `TEXCOORD_0` and nothing else — it
marks the material `KHR_materials_unlit`, so from the file's point of view normals would
never be read. Feed that geometry to a `MeshStandardMaterial` and it does not fail or
warn: it draws a flat black silhouette, which reads as a catastrophically bad
reconstruction rather than as a missing attribute. Measured on the cup mesh: 30,578
pixels covered, **0 of them lit**, mean brightness 0. After `computeVertexNormals()`,
the same 30,578 pixels, 30,478 lit, mean brightness 158.

`MeshViewer:ensureNormals` computes them on load when they are absent, which is the mesh
stage's file but not the export stage's — Blender writes normals on the way out, which is
why the export page looked shaded while the mesh page did not. That asymmetry is the tell
if this ever comes back.

**Related: an export must not be offered the Flip toggle.** The export stage has already
applied `capture.flip_x` and baked the orientation into the file. Offering it again turns
an upright model upside down, and because `flip_x` is in that stage's fingerprint, merely
looking at it marks the export stale. `MeshViewer` takes `allowFlip`, and `ExportPage`
passes `false`.

**6.27 TextureMesh's seam levelling blacks out the interior of every texture patch.**
Both levelling passes are on by OpenMVS's default, and on the meshes this pipeline
produces they do not smooth the texture — they zero it. The tell is that the atlas still
looks plausible: each patch keeps the photograph in the *padding* dilated around it, and
only the interior, which is the part every face actually samples, goes black. So the
model renders as a black surface webbed with thin coloured lines, which reads as a
texturing failure of some deeper kind.

Measured on the cup mesh, as the fraction of non-empty atlas pixels that are pure black:

| `--global-seam-leveling` / `--local-seam-leveling` | black |
|---|---|
| `1` / `1` — OpenMVS's default | **49.1%** |
| `0` / `1` | 37.3% |
| `0` / `0` | **0.9%** |

Local levelling is the larger offender, but neither is safe here, so `MeshParams` exposes
both and defaults **both off** — against the engine's own default, which is unusual enough
to want the numbers written down. A patchy texture is worth incomparably more than a black
one. Suspicion is that a mesh carrying 10,848 tiny patches is what breaks the solve, so on
a clean subject these may be worth turning back on; the parameters are there, and the
check is the atlas, not the render.

## 7. The smoke test: `cup-1`

**Run:** `D:\pgh-runs\20260907-cup-1-2` · **Source:**
`C:\Users\patru\Downloads\20260907_Coffee-Cup-Orbit.mp4`

Handheld single-camera orbit of a mug on a glass patio table. 60.3 s, ~1.5 circuits, HEVC
2320×1080 portrait (−90° display matrix), 29.92 fps, bt709, has audio. Configured as
`camera_orbits`, `revolutions=1.5`, one `SINGLE` segment. Masking **skipped**.

| Reading | Value | Verdict |
|---|---|---|
| Rotation rate | 9.0 deg/s | Good — well under the 15 deg/s rolling-shutter threshold |
| Sync residual p95 | 0.011 s | Fine (single camera, so not load-bearing) |
| Duplicates | 15 (6%) | Normal; selection dropped them |
| Mean luma range | 17% | **Auto-exposure was hunting** — expect texture seams |
| VFR | suspected | Real phone footage; handled |

**Known weaknesses of this capture, in rough order of severity.** All were predicted
before the run and all showed up; none blocked it.

1. **Glass table.** Transparent and specular. Reflections move with the camera and violate
   the rigid-scene assumption — produced the expected haze of phantom points below the mug.
2. **Exposure hunting** (17%), plus a bright window blowing highlights in part of the
   orbit. Lock AE/AWB next time.
3. **Single elevation ring.** Like the chair spin, this sees one band — the top of the mug
   and the underside of the rim are weak.
4. **Thin objects** (brushes in the mug) reconstructed as streaks, not cylinders.

Also on disk: `D:\pgh-test\{master,cam_high,cam_eye}.mp4` — the synthetic ground-truth
pair from §5. Keep them; they are the regression test for slot alignment.
`D:\pgh-runs\20260907-extract-test` is their run.

---

## 8. Next steps

**One milestone remains.** M11 and M12 are built, so the chain runs from footage to a
printable file. M9 (mask) is the hardest, the only one the face scan strictly requires,
and the one that wants footage that has not been shot yet.

### M11 — Mesh ✅ built

`ReconstructMesh` → `RefineMesh` → `TextureMesh`, over `dense/scene_dense.mvs`.

Follow `stages/dense.py` closely — it is the template, and the awkward parts are already
solved there: argv builders in `vendor/openmvs.py`, `_run_openmvs` for log-tailed
progress, `_PeakMemory` for the RAM reading, scratch-then-commit.

**Do this first, before writing any builder.** Run each tool with `--help` in a scratch
directory and read the log it drops, exactly as was done for the two tools already
wrapped. The `openMVS - Source/` checkout in this repo tracks `develop` and is months
ahead of the v2.4.0 binaries actually being run, so its documented options are not
reliable. Record what you find in the `vendor/openmvs.py` module docstring, which already
lists the flags for `InterfaceCOLMAP` and `DensifyPointCloud`.

Expect `--export-type obj` to matter: an OBJ + MTL + texture PNG is far easier to feed to
Blender at M12 than a textured PLY.

**Viewer.** `PointCloudViewer` handles points only, so this needs a sibling `MeshViewer`.
Three shading modes, per the original design: photographic texture, neutral matte, and
wireframe. That is not decoration — a surface problem hidden by a convincing texture is
the usual failure mode, and matte shading is what exposes it. three.js loads OBJ+MTL via
`OBJLoader`/`MTLLoader` from `three/examples/jsm/`.

Watch: `RefineMesh` is the slow step and the one most likely to exhaust memory. Expose its
resolution/scale knob as prominently as `resolution_level` is exposed on dense, and
consider defaulting refinement **off** until it has been run once successfully.

### M12 — Export ✅ built

Headless Blender: `blender -b -P <script>` (5.2 LTS is installed; note it is
`required=False` in `config.py`, so the export stage must check for it in `preflight`).
Largest connected component, decimate, scale, orient and centre, then GLB/OBJ/STL plus a
turntable render.

**Orientation is recoverable, and worth doing properly.** COLMAP's world frame is
arbitrary — the cup model comes out lying on its side. But the camera centres in
`sparse/poses.json` lie in a plane, and that plane's normal *is* the up axis for both
capture modes: the cameras orbit the subject horizontally in one, and the subject rotates
about a vertical axis in the other. Fit a plane to the camera centres (or take the axis of
least variance, which is the cheap version) and you have "up" without asking the user. The
sign is ambiguous, and that half is already resolved: `capture.flip_x` in `project.json`
records a 180° turn about +X, defaulting on, set from the `Flip` toggle in the cloud
viewer's toolbar. Apply it about the model centre so the export comes out the way up it
was reviewed, and do not ask the user again. It is deliberately absent from every
fingerprint (nothing upstream reads it); add it to the export stage's `external_inputs()`
when that stage is written — at creation, so it cannot rehash anything that exists.

**Scale.** `capture.baseline_mm` for a fixed-mount rig — that measurement is the only part
that cannot be reconstructed afterwards, which is why the checklist nags about it. For
`camera_orbits` there is nothing to scale from, so this needs either a ruler in frame or a
manual "this dimension is N mm" input. Say so plainly in the UI rather than exporting
something confidently mis-scaled.

### M9 — Mask

SAM 2.1 large (`sam2.1_hiera_l.yaml` — the pairing is pinned in `config.py`; a mismatched
pair loads *without raising* and produces poor masks). Click-to-prompt on frame 0 per
camera, propagate in timeline order, **chunked at ~150 frames** carrying the last mask
forward as the next chunk's seed — the video predictor's inference state grows with
sequence length and will OOM 24 GB otherwise. Design progress as `chunks × frames` from
the start.

Four things are already waiting for it:

- `StageState.SKIPPED` and the skip button exist so a run can proceed without masks;
  un-skipping should bring this stage back.
- `SparseStage.external_inputs` already reports whether masks exist, so producing them
  correctly invalidates the alignment rather than leaving it falsely fresh.
- **`DenseStage.preflight` refuses to run when masks are present**, because undistorting
  them (§6.8) is not implemented. Lifting that block is part of this milestone, not before
  it.
- `DensifyPointCloud` takes masks natively via `-m/--mask-path` (§6.7), so the OpenMVS side
  may be less work than expected. COLMAP takes them via `--ImageReader.mask_path`
  (per-image) or `--ImageReader.camera_mask_path` (one for all), and the dialect probe
  already reports whether `mask_path` is supported.

*For the chair spin, deliberately run alignment without masks first.* It will very likely
produce a background-locked model, and seeing that failure in the viewer is what makes
this stage's parameters comprehensible. The align stage already emits a warning saying
exactly this when the mode is `subject_rotates` and masking was skipped.

### Smaller things

- **Rigs (§4.1).** `sequential_matcher` in COLMAP 4.2 has `expand_rig_images`, and the
  matcher has `rig_verification` and `skip_image_pairs_in_same_frame`. The slot naming
  that makes this possible is already in place. This is the single biggest quality lever
  available for the two-camera chair spin.
- **Sprite sheets.** The contact sheet loads individual thumbnails; fine at 241 frames,
  will need batching at 3600.
- **Driving a stage headlessly** is useful for the long OpenMVS runs. The pattern that
  worked: build a `JobRunner(registry)`, `runner.start()`, `runner.submit(run, sid)`, then
  poll `run.load().stages[sid].state` — *not* `run.io.load()` (§6.17), and not `job.state`
  (§6.18). Stop the web server first or it will overwrite you.

---

## 9. Known issues and loose ends

- **`degrees_per_second` needs `capture.revolutions` set by hand.** It stays silent
  otherwise, on purpose — a guessed revolution count produces a confident number derived
  from nothing.
- **dHash is 8×8 and can be fooled by highly regular patterns.** Synthetic `testsrc2`
  colour bars register as 100% duplicates because their 8×8 signature never changes. Real
  footage is unaffected (cup-1: 6%).
- **The vocabulary tree in `data/` is unusable** and matching falls back to exhaustive. Not
  a defect — see §6.11. The doctor reports it.
- **The dense stage refuses to run when masks are present** (§6.8), by design. See §8.
- **`model_analyzer` contributed nothing on the cup run** — `mean_observations_per_image`
  came back null, so the metrics fall back to values parsed from the text model. Harmless,
  and every number that matters is present, but the parser expects a format the binary may
  not be emitting.
- **`opencv-5.0.0-windows.exe`** (186 MB) is still in the project root and unused — OpenCV
  comes from pip. Safe to delete.
- **`openMVS - Source/`** is a source checkout at `develop`, months ahead of the v2.4.0
  prebuilt binaries actually in use. Do not mix scene files between versions, and do not
  trust its option lists (§8, M11).
- **Line endings are mixed** across the tree — some files CRLF, some LF, no
  `.gitattributes`. Match whatever the file you are editing already uses.
- **Blender is optional in the doctor** and only used at M12.

---

## 10. The stills path

A run's source is either video clips or a folder of photographs, chosen at the top of the
Clips page. Both produce the same thing:

```
frames/<camera_group>/<slot:06d>.jpg   +   frames/frames.jsonl
```

and nothing downstream knows which branch ran. It is an ingest branch inside the extract
stage rather than a stage of its own, precisely so that stays true.

**Why it exists.** A folder of stills is the cleanest input this pipeline can be given: no
rolling shutter, no variable frame rate, no display matrix, no inter-frame compression,
and real EXIF. When a video run produces a bad model, a photo run separates "the
reconstruction chain is wrong" from "extraction is feeding it bad frames" — the question
you otherwise cannot answer without suspecting everything at once.

**Photographs are hardlinked, not copied or re-encoded**, whenever nothing needs to
change. That is instant and costs no disk, but the real reason is EXIF: COLMAP seeds each
camera's focal length from it and otherwise falls back to guessing
`1.2 × max(width, height)`. On the cup video, which has no EXIF, that guess was 2784 px
and bundle adjustment had to pull it down to 1683 px on its own.

Two things force a re-encode, both reported in the metrics as `reencoded`:

- an image larger than `max_dimension`;
- an image carrying an **EXIF orientation flag**, which is applied to the pixels and then
  cleared. COLMAP does not honour that flag reliably, and a silently sideways subset of a
  shoot is the kind of failure that costs an afternoon. EXIF is copied across on both
  paths.

**Ordering is natural, not lexical** — `IMG_9` precedes `IMG_10`. Sequential matching
compares frames by index, so index order has to be viewpoint order; plain string sort
scrambles it in a way nothing downstream can detect.

**What photographs do not have is a clock.** `src_pts`, `timeline_time_s` and
`sync_residual_s` are `null` in `frames.jsonl` rather than zero, and the timeline carries
only `total_slots`. Writing 0.0 where there is no timestamp would put a number that means
nothing into the one file every later stage reads. A run therefore cannot mix photo and
video clips, and preflight refuses it: the shared slot clock that pairs two cameras
instant-by-instant comes from video timestamps, and half an invented timeline is worse
than none.

RAW and HEIC need a decoder that is not installed; the scanner names the files it could
not read rather than skipping them quietly. `docs/capture-checklist.md` has the shooting
guidance.
