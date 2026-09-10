"""Turn the rendered turntable frames into two clips the harness can actually ingest.

Three things happen here that a plain ``ffmpeg`` one-liner would get wrong.

**The clips must not start together.** Two phones started by hand never do, and the harness
has a whole module for putting them back in step. Rendering both cameras over the same frame
range and encoding both from frame 1 would hand it a zero offset and quietly skip the part we
want tested. So the raised camera's clip starts a known number of frames late, and that known
number is what ``sync.py`` has to rediscover.

**The clips need audio, and specifically a transient.** ``sync.py`` aligns by
cross-correlating the 300-4000 Hz band and rejects anything scoring under 25; a shared clap
scores 190-310, and silence scores nothing at all. The capture checklist asks for a clap for
exactly this reason, so the synthetic take gets one too -- placed after *both* cameras are
rolling, which is the detail that makes it useful. A clap before the second camera starts is
in only one clip and correlates with nothing.

**Constant frame rate.** ``extract`` reads packet timestamps and will flag a variable rate;
there is no reason for a rendered clip to have one.

Run it with the project venv after the render finishes::

    .venv\\Scripts\\python.exe scripts\\encode_turntable_clips.py
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
#: Placed relative to the *later* camera's start, so it lands in both clips.
CLAP_AFTER_SECOND_START_S = 0.5
ROOM_TONE = 0.004


def find_ffmpeg(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} is not on PATH; the extract stage needs it too")
    return found


def clap_track(duration_s: float, clap_at_s: float, seed: int = 7) -> np.ndarray:
    """Room tone with one sharp transient, which is all the correlator wants.

    A decaying noise burst rather than a tone: a sine has a periodic autocorrelation and
    would give the correlator several equally good answers a fixed distance apart.
    """
    rng = np.random.default_rng(seed)
    samples = int(round(duration_s * SAMPLE_RATE))
    audio = rng.standard_normal(samples).astype(np.float32) * ROOM_TONE

    for offset_s, level in ((0.0, 0.9), (0.035, 0.28)):
        start = int(round((clap_at_s + offset_s) * SAMPLE_RATE))
        length = int(0.09 * SAMPLE_RATE)
        if start < 0 or start + length > samples:
            continue
        envelope = np.exp(-np.linspace(0.0, 1.0, length) * 34.0).astype(np.float32)
        burst = rng.standard_normal(length).astype(np.float32) * envelope * level
        audio[start : start + length] += burst

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.89
    return audio


def write_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def encode(ffmpeg: str, frames_dir: Path, first_frame: int, count: int, fps: int,
           audio: Path, out: Path) -> None:
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "warning",
        "-framerate", str(fps), "-start_number", str(first_frame),
        "-i", str(frames_dir / "%04d.png"),
        "-i", str(audio),
        "-frames:v", str(count),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-pix_fmt", "yuv420p",
        # Tag the colour space rather than leaving it unset: extract's colour handling
        # tone-maps anything it reads as HDR, and an untagged clip is a guess it has to make.
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-fps_mode", "cfr", "-r", str(fps),
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(command, check=True)


def probe(ffprobe: str, path: Path) -> dict:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    return {
        "duration_s": float(data["format"]["duration"]),
        "frames": int(video.get("nb_frames", 0)),
        "size": [video["width"], video["height"]],
        "avg_frame_rate": video["avg_frame_rate"],
        "has_audio": audio is not None,
        "bytes": int(data["format"]["size"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\pgh-test\suzanne")
    args = parser.parse_args()

    root = Path(args.root)
    truth = root / "groundtruth"
    scene = json.loads((truth / "scene.json").read_text(encoding="utf-8"))
    rig = json.loads((truth / "rig.json").read_text(encoding="utf-8"))

    fps = int(scene["fps"])
    total = int(scene["frames"])
    offset_s = float(rig["start_offset_s"])

    # Quantise the offset to whole frames. A fractional one cannot be produced by trimming
    # frames, and claiming an offset the clip does not have would make every later
    # comparison against "ground truth" a comparison against a number we made up.
    offset_frames = int(round(offset_s * fps))
    true_offset_s = offset_frames / fps

    ffmpeg, ffprobe = find_ffmpeg("ffmpeg"), find_ffmpeg("ffprobe")
    master_duration = total / fps
    clap_at_s = true_offset_s + CLAP_AFTER_SECOND_START_S
    master = clap_track(master_duration, clap_at_s)

    plan = {
        "cam-eye": {"first": 1, "count": total, "start_s": 0.0},
        "cam-high": {"first": 1 + offset_frames, "count": total - offset_frames,
                     "start_s": true_offset_s},
    }

    report = {"true_offset_s": true_offset_s, "clap_at_master_s": clap_at_s, "clips": {}}
    for name, spec in plan.items():
        frames_dir = root / f"frames-{name}"
        found = len(list(frames_dir.glob("*.png")))
        if found < total:
            raise SystemExit(f"{frames_dir} holds {found} frames, expected {total}")

        begin = int(round(spec["start_s"] * SAMPLE_RATE))
        wav = root / f"audio-{name}.wav"
        write_wav(wav, master[begin : begin + int(round(spec["count"] / fps * SAMPLE_RATE))])

        out = root / f"{name}.mp4"
        encode(ffmpeg, frames_dir, spec["first"], spec["count"], fps, wav, out)
        info = probe(ffprobe, out)
        info["first_render_frame"] = spec["first"]
        info["clip_local_clap_s"] = clap_at_s - spec["start_s"]
        report["clips"][name] = info
        print(f"{name}: {info['frames']} frames, {info['duration_s']:.3f}s, "
              f"{info['bytes'] / 1e6:.1f} MB, audio={info['has_audio']}")

    rig["start_offset_s"] = true_offset_s
    rig["start_offset_frames"] = offset_frames
    rig["clap_at_master_s"] = clap_at_s
    rig["clips"] = {k: {"first_render_frame": v["first_render_frame"],
                        "clip_local_clap_s": v["clip_local_clap_s"]}
                    for k, v in report["clips"].items()}
    (truth / "rig.json").write_text(json.dumps(rig, indent=2), encoding="utf-8")

    overlap_s = min(p["count"] / fps for p in plan.values())
    print(f"\ntrue offset {true_offset_s:.4f}s ({offset_frames} frames), "
          f"clap at {clap_at_s:.3f}s master")
    print(f"overlapping window {overlap_s:.2f}s = "
          f"{overlap_s * fps * scene['degrees_per_frame'] / 360.0:.2f} revolutions")
    print(f"at 6 fps that is {math.floor(overlap_s * 6) + 1} slots per camera")
    return 0


if __name__ == "__main__":
    sys.exit(main())
