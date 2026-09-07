"""Audio cross-correlation, and the rotation/HDR probe helpers.

The sign convention is the point of the sync tests. The manifest defines
``t_run = pts + time_offset_s``, so a returned offset must be the value that, added
to the *other* clip's timestamps, puts a shared event at the same instant as in the
reference. Getting this backwards would silently double the misalignment rather
than remove it, and the symptom would appear five stages later as a reconstruction
that will not converge.
"""

from __future__ import annotations

import numpy as np
import pytest

from pgh.sync import SAMPLE_RATE, cross_correlate
from pgh.vendor.ffmpeg import (
    _parse_rate,
    detect_vfr,
    effective_rotation,
    extract_rotation,
    rotation_filter,
)


def make_clap_track(
    duration_s: float, clap_at_s: float, *, seed: int = 0, noise: float = 0.02
) -> np.ndarray:
    """Room tone with one sharp transient, like a real take."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SAMPLE_RATE)
    signal = rng.normal(0, noise, n)

    # A clap: a few ms of loud broadband energy with a fast exponential decay.
    start = int(clap_at_s * SAMPLE_RATE)
    length = int(0.03 * SAMPLE_RATE)
    envelope = np.exp(-np.linspace(0, 12, length))
    signal[start : start + length] += rng.normal(0, 1.0, length) * envelope
    return signal


@pytest.mark.parametrize(
    ("ref_at", "other_at"),
    [(2.0, 2.5), (2.5, 2.0), (1.0, 1.0), (0.5, 3.25)],
)
def test_offset_recovers_known_shift(ref_at, other_at):
    reference = make_clap_track(6.0, ref_at, seed=1)
    other = make_clap_track(6.0, other_at, seed=2)

    result = cross_correlate(reference, other)

    expected = ref_at - other_at
    assert result.offset_s == pytest.approx(expected, abs=0.005), (
        f"expected {expected:+.3f}s, got {result.offset_s:+.3f}s"
    )


def test_recovered_offset_actually_aligns_the_event():
    """The property that matters, stated directly in manifest terms."""
    ref_at, other_at = 2.0, 2.62
    result = cross_correlate(
        make_clap_track(6.0, ref_at, seed=3), make_clap_track(6.0, other_at, seed=4)
    )

    # t_run = pts + time_offset_s, with the reference clip at offset 0.
    aligned_other = other_at + result.offset_s
    assert aligned_other == pytest.approx(ref_at, abs=0.005)


def test_confidence_is_high_for_a_real_clap():
    result = cross_correlate(
        make_clap_track(6.0, 2.0, seed=5), make_clap_track(6.0, 2.4, seed=6)
    )
    assert result.trustworthy, f"confidence only {result.confidence:.1f}"


def test_confidence_is_low_for_unrelated_noise():
    """No clap, no shared event: the harness must admit it does not know."""
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 4 * SAMPLE_RATE)
    b = rng.normal(0, 1, 4 * SAMPLE_RATE)

    result = cross_correlate(a, b)

    assert not result.trustworthy, (
        f"pure noise scored {result.confidence:.1f} -- the harness would report a "
        "confident but meaningless offset"
    )


# -- rotation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("probe_rotation", "expected_clockwise"),
    [(0, 0), (-90, 90), (90, 270), (180, 180), (-180, 180), (270, 90)],
)
def test_effective_rotation_mapping(probe_rotation, expected_clockwise):
    """ffprobe reports counter-clockwise degrees needed to display."""
    assert effective_rotation(probe_rotation) == expected_clockwise


@pytest.mark.parametrize(
    ("clockwise", "expected"),
    [
        (0, []),
        (90, ["transpose=1"]),
        (180, ["transpose=1", "transpose=1"]),
        (270, ["transpose=2"]),
    ],
)
def test_rotation_filter_chain(clockwise, expected):
    assert rotation_filter(clockwise) == expected


def test_extract_rotation_prefers_display_matrix():
    stream = {
        "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
        "tags": {"rotate": "180"},
    }
    assert extract_rotation(stream) == -90.0


def test_extract_rotation_falls_back_to_legacy_tag():
    """Older Android and .mov files carry rotation as a clockwise container tag."""
    assert extract_rotation({"tags": {"rotate": "90"}}) == -90.0


def test_extract_rotation_defaults_to_zero():
    assert extract_rotation({"codec_name": "hevc"}) == 0.0


# -- frame rate --------------------------------------------------------------


def test_parse_rational_frame_rates():
    assert _parse_rate("60000/1001") == pytest.approx(59.94, abs=0.01)
    assert _parse_rate("30/1") == 30.0
    assert _parse_rate("0/0") is None
    assert _parse_rate(None) is None


def test_detect_vfr_accepts_constant_rate():
    times = [i / 60.0 for i in range(600)]
    is_vfr, median = detect_vfr(times)
    assert not is_vfr
    assert median == pytest.approx(1 / 60, abs=1e-6)


def test_detect_vfr_flags_dropped_frames():
    """A phone throttling under load drops frames; stride sampling then drifts."""
    times = [i / 60.0 for i in range(600)]
    del times[100:140]  # a run of dropped frames
    is_vfr, _ = detect_vfr(times)
    assert is_vfr
