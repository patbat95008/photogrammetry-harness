# Capture Checklist

Read this before you film. Most reconstruction failures are decided at capture time
and cannot be fixed in software afterwards.

---

## First: which way round are you shooting?

The harness supports two capture modes, and the choice changes what the later
stages do. Set it on the run's Clips page.

| | **Subject rotates** | **Cameras orbit** |
|---|---|---|
| What moves | You, in a swivel chair | The cameras, around a still subject |
| Background masking | **Mandatory** | **Skip it** |
| Rigid camera rig | Yes, if both cameras are on fixed mounts | No — handheld cameras have no fixed relative pose |
| Absolute scale from baseline | Yes, measure it | No, use a ruler in frame instead |
| Expression drift | A real risk over a 45 s spin | Much worse — an orbit takes longer |

**Why masking inverts.** When you rotate and the cameras stay put, the room is
static in the world but *moves relative to your head*. The reconstruction solves for
one rigid scene, so the room is the part behaving inconsistently: left in, the solver
locks onto it and your head never resolves. When the cameras orbit instead, the room
and the subject are mutually rigid — the background then supplies extra features and
helps the orbit close, and masking it out throws that away.

**A handheld orbit is the better smoke test.** For a first run of the pipeline, orbit
a rigid object on a table with a phone in each hand. It exercises rotation metadata,
HDR handling, frame-rate detection and the frame-count checks against your actual
phones, without needing a chair, a rig, or a still face. Add the two clips as
*simultaneous handheld cameras* so they are paired in time but solved independently.

The rest of this checklist assumes the chair-spin mode, which is the harder one.

---

## Or shoot stills instead

The harness takes a folder of photographs as an alternative to video — pick **Photo
set** at the top of the Clips page. For everything except a moving subject, stills are
the better input, and not marginally:

| | Video frame | Photograph |
|---|---|---|
| Rolling shutter | Skews every moving frame | None on a still |
| Compression | Inter-frame, so detail is borrowed from neighbours | One frame, one encode |
| Frame rate | Has to be detected, and phones lie about it | Not a concept |
| Focal length | Not recorded; COLMAP guesses from image size | In EXIF, and COLMAP uses it |
| Resolution | Capped by the video mode | The sensor's full output |

Photographs are hardlinked into the run untouched wherever possible, precisely so the
EXIF survives. Two exceptions: an image larger than the size cap is resampled, and an
image carrying an EXIF rotation flag is physically rotated and the flag cleared,
because COLMAP does not apply that flag reliably.

**Shooting a photo set**

- **Overlap generously.** Aim for roughly 70% shared content between neighbours —
  in practice a step every 10° or so around the subject. Too little overlap is the
  single most common cause of a reconstruction that will not close.
- **Lock focus, aperture and ISO,** and do not change the zoom. Changing focal length
  mid-shoot means one camera model no longer describes the set; if you must, give each
  focal length its own camera group.
- **Do not crop afterwards.** A crop changes the effective focal length while leaving
  the EXIF claiming otherwise, which is worse than having no EXIF at all.
- **Export at full resolution,** and avoid anything that strips metadata. Messaging
  apps and cloud sync are the usual culprits.
- **Shoot more than one height.** A single ring around the subject leaves the top and
  underside unseen, and the mesher will invent them rather than leave a hole.
- RAW and HEIC are not decodable here — export to JPEG or TIFF first. The harness
  names the files it could not read rather than skipping them quietly.

---

## Before you sit down

**1. Measure the baseline between the two cameras.** *(fixed mounts only)*
Tape-measure the distance between the two phone *lenses*, as precisely as you can,
and write it down in millimetres. Note roughly where each camera sits relative to
your head (e.g. "high, ~30 cm above eye level, angled down ~20°").

> Photogrammetry recovers shape but not size. Two cameras at a known fixed separation
> are the cheapest way to recover absolute scale, and this measurement is the only
> part of it you cannot reconstruct later. If you skip it, you find out at export —
> by which point the tripods have moved.

**2. Lock exposure, focus and white balance on both phones.**
Both iOS and Android allow this (long-press to lock on iOS; pro/manual mode on most
Android cameras).

> Left on auto, the phone re-exposes as you rotate toward a window and hunts for
> focus mid-spin. Exposure drift produces visible seams in the final texture; focus
> hunting produces frames that are silently useless.

**3. Set both phones to the same frame rate and resolution.**
4K60 is ideal. If one phone can only do 4K30, use 30 on both — matched rates make
the two clips line up on a shared timeline much more cleanly.

**4. Lighting: bright, diffuse, and fixed.**
Two soft sources or an overcast-window room. Avoid a single hard lamp, and avoid
positioning yourself so you rotate to face a bright window.

> The lighting stays fixed while *you* turn, so every surface of your face is lit
> differently at each angle. Diffuse light minimises how much that matters. Hard
> shadows that sweep across your face as you rotate are the worst case.

**5. Tie back or wet down flyaway hair.** Fine hair strands cannot be reconstructed
and produce noise that the mesher turns into spikes.

---

## Recording

**6. Start both recordings, then clap once, sharply.**

> The harness aligns the two clips by cross-correlating their audio. A single sharp
> transient gives it something unambiguous to lock onto. Without it, sync falls back
> to manual nudging — and sync accuracy is exactly what makes the two-camera rig
> work, because frames from the two cameras must be paired by identical instant.

**7. Spin slowly — roughly 45 seconds per full revolution.**
Not 15. Push off gently once and let the chair coast, or have someone turn you
smoothly.

> Every phone has a rolling shutter: it reads the sensor row by row rather than all
> at once, so a moving subject is recorded with a slight per-row skew. The faster you
> turn, the worse the skew, and no amount of masking or filtering fixes it. Slow
> rotation is the entire mitigation.

**8. Hold still, and neutral.**
Mouth closed, jaw relaxed, eyes open and looking straight ahead. Breathe shallowly.
Try not to swallow. Keep your head fixed relative to your torso — let the chair do
all the turning.

> The reconstruction assumes you are a rigid object. Every expression change,
> swallow, and neck adjustment violates that. The chin and neck are the usual
> casualties.

**9. One continuous take per pass. Do at least 1.5 revolutions.**

> Never stitch two takes together to make one pass — your expression and posture
> change between them, and the reconstruction will try to average two different
> heads. The extra half-revolution gives the software overlapping views to close
> the loop with.

---

## The third pass: the crown

**10. Record a separate short pass covering the top of your head.**
Lean forward maybe 30° so the crown faces the cameras, and spin again. Or have
someone hold a phone above you and orbit it.

> Two cameras at head height sweep out a horizontal band of views. Neither can see
> the top of your head, and the mesher will not leave a hole — it will confidently
> invent a smooth bald dome. This pass is what prevents that.

This is a *separate* pass, not simultaneous with the other two, so it does not need
to be sync-matched to them. The harness models it as its own capture segment.

---

## Before you tear down

**11. Do a 20-second test spin of any rigid object first** — a mug, a bust, a
backpack on the chair — and push it through the harness end to end. This surfaces
rotation, HDR, and frame-rate problems while the setup is still standing.

**12. Check you have:**

- [ ] Baseline measurement written down, in millimetres
- [ ] Number of revolutions the take covers, so the rotation-rate check has
      something to calibrate against
- [ ] Two clips of the main spin, both cameras, both containing the clap
- [ ] One crown pass
- [ ] Everything transferred off the phones at **full quality** — beware of
      messaging apps and cloud sync that silently re-compress video

---

## What the harness will tell you afterwards

Once extraction runs, the results panel reports several things that are worth
checking before you go further, because each one is only fixable by re-shooting:

| Reading | What it means |
|---|---|
| **Degrees per second** | Above ~15°/s, rolling-shutter skew is degrading the reconstruction. Spin slower. |
| **Mean luma range** | A spread above ~15% means exposure drifted. Lock AE next time. |
| **Sync residual (p95)** | Should be under one source-frame period. Higher means the two clips are not reliably paired. |
| **Duplicate frames** | The phone dropped frames, usually from thermal throttling. Shorter takes, or let it cool. |
| **VFR suspected** | The phone did not hold a constant frame rate. Handled, but worth knowing. |
| **HDR detected** | Footage is HLG/PQ and will be tone-mapped. Fine, just be aware it happened. |
