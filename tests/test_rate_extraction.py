"""Tests for FFT-based breathing-rate extraction and CW cross-validation."""

import numpy as np
import pytest

from src.breathing_models import chest_displacement
from src.cw_radar import cw_iq
from src.demodulation import recover_displacement
from src.rate_extraction import estimate_bpm, estimate_rate

FS = 200.0

# Expected intra-burst rate (carrier during active breathing) per pattern.
EXPECTED = {
    "normal": (15.0, 0.6),
    "tachypnea": (30.0, 1.0),
    "bradypnea": (8.0, 1.0),
    "cheyne_stokes": (15.0, 1.5),
    "obstructive_apnea": (15.0, 1.5),
}


@pytest.mark.parametrize("pattern", list(EXPECTED))
def test_estimate_bpm_per_pattern(pattern):
    expected, tol = EXPECTED[pattern]
    _, d = chest_displacement(pattern, duration=120, fs=FS)
    bpm, _ = estimate_bpm(d, FS)
    assert abs(bpm - expected) < tol


def test_pure_sine_rate_and_quality():
    t = np.arange(0, 120, 1 / FS)
    x = np.sin(2 * np.pi * 0.25 * t)  # 15 br/min
    rate, snr = estimate_rate(x, FS)
    assert abs(rate * 60 - 15) < 0.3
    assert snr > 3.0


def test_cw_pipeline_end_to_end():
    """d(t) -> CW I/Q -> demod -> rate: the full radar chain recovers the rate."""
    _, d = chest_displacement("normal", duration=120, fs=FS, rate_bpm=15)
    i, q = cw_iq(d)
    recovered = recover_displacement(i, q)
    bpm, _ = estimate_bpm(recovered, FS)
    assert abs(bpm - 15) < 0.6
