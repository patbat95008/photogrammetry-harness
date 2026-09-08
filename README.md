# Photogrammetry Harness

A local web harness that turns video — or a folder of photographs — into a 3D model.
It does not implement reconstruction; it drives COLMAP and OpenMVS, and owns the parts
those engines do badly: ingest, frame selection, masking, quality control, and making
failures legible.

Built with agentic tools (Claude Code) by one person, to make a printable 3D scan of the
owner's head — captured by rotating in an office chair while two phone cameras record
from different heights.

```
FFmpeg / stills → OpenCV / SAM 2 → COLMAP → OpenMVS → Blender
    ingest         select / mask     align   dense/mesh   cleanup
```

All seven stages are built and verified end to end on real footage.

## Running it

```
launch.bat     double-click it. Builds the UI if stale, serves the API and UI on
               http://127.0.0.1:8756, opens a browser once it is actually up.
               Closing the window shuts everything down.

dev.bat        two windows: Vite on :5173 with hot reload, auto-reloading API on :8756.
```

Tests: `.venv\Scripts\python.exe -m pytest server/tests -q`

The 3D viewers want a real browser. They do not render in some embedded and automation
browsers, where `ResizeObserver` never fires and the canvas stays at its default size —
so if a point cloud looks empty, check it in Firefox before concluding the
reconstruction failed.

## What it does that a pipeline script would not

**Nothing is recomputed that does not need to be.** Each stage is fingerprinted from its
own parameters, its declared inputs, and its upstream stages' fingerprints — never from
timestamps. So changing a parameter invalidates everything downstream, and changing it
*back* restores the original hash and leaves that work valid. Looking at a knob costs
nothing.

**Two cameras share one clock.** `frames/cam_high/000042.jpg` and
`frames/cam_eye/000042.jpg` are the same instant, not the same source frame number.
Clips are aligned by cross-correlating their audio, and the slot numbering is what lets
COLMAP treat the pair as a rigid rig later.

**How you captured it changes what the pipeline does.** If the subject rotates and the
cameras stay put, the room moves relative to the subject and the solver will lock onto
the room — so masking is mandatory. If the cameras orbit a still subject, the background
is rigid with it, supplies extra features, and masking it out makes alignment *harder*.
The harness knows the difference and says so on the page.

**Failures are meant to be legible.** Warnings name the cause and the consequence, in
sentences. A stage that cannot run explains what would go wrong rather than refusing.
The quiet failures — an unscaled STL, a model that reconstructed the room instead of the
subject, a mask that vanished halfway through the take — are the ones the reporting is
built around, because they all look like success.

## Setup

Run the Doctor page first; it probes every tool and reports what is missing.

Vendored dependencies are not in git — `.gitignore` documents each one with the command
to fetch it again. In short: COLMAP and OpenMVS binaries under `tools/`, the SAM 2
checkout as `sam2-src/` with its checkpoints, Blender installed normally, and
`pip install -r requirements.txt` into `.venv`.

Clone SAM 2 as `sam2-src`, **not** `sam2`: the server's working directory is the project
root, so a directory of that name there shadows the installed package.

Runs live outside the repo (`D:\pgh-runs` by default) because COLMAP and OpenMVS build
deep paths and may not honour long-path support.

## Documentation

- **`docs/capture-checklist.md`** — read before filming. Capture decides most outcomes,
  and almost nothing here is fixable afterwards.
- **`docs/HANDOVER.md`** — the real reference: architecture, what is verified and how,
  and a numbered list of expensive lessons that should not be re-derived.
- `docs/Dependency-Map.md`, `docs/Face-Scan-Plan.md` — the original planning notes.
