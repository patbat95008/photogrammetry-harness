"""Aligning simultaneously-recorded clips by cross-correlating their audio.

Two phones started by hand are offset by an unknown fraction of a second. That
offset is not a nicety: frames from the two cameras are paired by shared timeline
slot, and COLMAP is told those pairs are one rigid rig observing the same instant.
Get the offset wrong and every rig constraint is a lie.

A single sharp clap at the start of both takes gives the correlator an unambiguous
transient to lock onto, which is why the capture checklist asks for one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .proc import capture_bytes
from .vendor.ffmpeg import ffmpeg_path

logger = logging.getLogger("pgh.sync")

SAMPLE_RATE = 48_000
#: Only the opening seconds are needed -- the clap is there, and correlating whole
#: multi-minute clips costs memory for no extra certainty.
ANALYSIS_SECONDS = 90.0
#: Speech and hand-clap energy live here. High-passing away rumble and low-passing
#: away hiss both sharpen the correlation peak considerably.
BAND_HZ = (300.0, 4000.0)
#: Peak-to-sidelobe ratio below this means the correlator found nothing convincing.
#: Calibrated against measurement, not guessed: two takes sharing a clap score
#: 190-310, while uncorrelated room tone tops out around 7. The maximum of a few
#: hundred thousand noise samples always sits several sigma above their RMS, so a
#: naive low threshold would call pure noise a confident match. 25 sits in the gap.
MIN_CONFIDENCE = 25.0


@dataclass(slots=True)
class SyncResult:
    offset_s: float
    confidence: float
    #: Normalised correlation at the peak, 0-1. Envelopes are unit-norm, so this is
    #: directly interpretable: ~0.85 for a shared clap, ~0.05 for unrelated audio.
    peak: float = 0.0
    method: str = "audio-xcorr"

    @property
    def trustworthy(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE


def decode_audio(source: Path, cwd: Path, seconds: float = ANALYSIS_SECONDS) -> np.ndarray:
    """Decode the opening of a clip to mono float32 PCM at 48 kHz."""
    code, raw, stderr = capture_bytes(
        [
            ffmpeg_path(),
            "-hide_banner", "-nostdin",
            "-loglevel", "error",
            "-t", f"{seconds:.3f}",
            "-i", source,
            "-vn", "-sn", "-dn",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-f", "f32le",
            "-",
        ],
        cwd=cwd,
        timeout=300,
    )
    if code != 0:
        raise RuntimeError(f"audio decode failed: {stderr.strip()[:300]}")

    usable = len(raw) - (len(raw) % 4)
    samples = np.frombuffer(raw[:usable], dtype="<f4")
    if samples.size == 0:
        raise RuntimeError("clip contains no decodable audio")
    return np.nan_to_num(samples.astype(np.float64))


def _bandpass(signal: np.ndarray) -> np.ndarray:
    """Zero-phase band-pass. Zero-phase matters: a filter that shifts the signal
    would shift the very offset we are trying to measure."""
    try:
        from scipy.signal import butter, sosfiltfilt

        nyquist = SAMPLE_RATE / 2
        sos = butter(
            4,
            [BAND_HZ[0] / nyquist, min(BAND_HZ[1] / nyquist, 0.99)],
            btype="band",
            output="sos",
        )
        return sosfiltfilt(sos, signal)
    except Exception:  # pragma: no cover - scipy missing or degenerate input
        logger.warning("scipy band-pass unavailable; correlating unfiltered audio")
        return signal


def _envelope(signal: np.ndarray) -> np.ndarray:
    """Correlate on the energy envelope rather than the waveform.

    Two microphones in different positions record different waveforms of the same
    event -- different room reflections, different phase. The *envelope* of that
    event is far more similar between them, so it correlates much more reliably.
    """
    energy = np.abs(signal)
    window = SAMPLE_RATE // 1000  # 1 ms smoothing
    if window > 1:
        kernel = np.ones(window) / window
        energy = np.convolve(energy, kernel, mode="same")
    energy -= energy.mean()
    norm = np.linalg.norm(energy)
    return energy / norm if norm > 0 else energy


def cross_correlate(reference: np.ndarray, other: np.ndarray) -> SyncResult:
    """Return the offset to add to ``other`` to align it with ``reference``."""
    a = _envelope(_bandpass(reference))
    b = _envelope(_bandpass(other))

    size = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    spectrum = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    correlation = np.fft.irfft(spectrum, size)

    # Lags are stored wrapped: the second half represents negative lags.
    correlation = np.concatenate((correlation[-(len(b) - 1):], correlation[: len(a)]))
    lags = np.arange(-(len(b) - 1), len(a))

    peak_index = int(np.argmax(correlation))
    peak = float(correlation[peak_index])

    # Peak-to-sidelobe ratio, against the RMS of the correlation outside a guard
    # band around the peak. A shared clap gives a tall isolated spike; unrelated
    # room tone gives a broad low hump whose maximum is only a few sigma up.
    guard = SAMPLE_RATE // 20  # 50 ms either side
    masked = correlation.copy()
    masked[max(0, peak_index - guard) : peak_index + guard] = 0
    sidelobe_rms = float(np.sqrt((masked**2).mean())) or 1e-12
    confidence = peak / sidelobe_rms if peak > 0 else 0.0

    return SyncResult(
        offset_s=float(lags[peak_index]) / SAMPLE_RATE,
        confidence=confidence,
        peak=peak,
    )


def align_clips(reference: Path, other: Path, cwd: Path) -> SyncResult:
    """Measure how much later ``other`` started than ``reference``."""
    return cross_correlate(decode_audio(reference, cwd), decode_audio(other, cwd))
