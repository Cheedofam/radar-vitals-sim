"""Tests for fall-event injection and short-time-energy fall detection."""

import numpy as np

from src.breathing_models import chest_displacement
from src.cw_radar import cw_iq
from src.fall_detection import add_fall_event, detect_fall

FS = 1000.0  # high enough that the fast fall Doppler is not aliased


def test_no_fall_in_normal_breathing():
    _, d = chest_displacement("normal", duration=40, fs=FS)
    i, q = cw_iq(d)
    result = detect_fall(i, q, FS)
    assert result.detected is False


def test_fall_is_detected_near_its_time():
    _, d = chest_displacement("normal", duration=40, fs=FS)
    d = add_fall_event(d, FS, t_start=25.0, magnitude_mm=300, duration_s=0.3)
    i, q = cw_iq(d)

    result = detect_fall(i, q, FS)

    assert result.detected is True
    assert abs(result.t_peak_s - 25.0) < 0.5


def test_fall_raises_energy_ratio():
    _, base = chest_displacement("normal", duration=40, fs=FS)
    fallen = add_fall_event(base, FS, t_start=20.0)

    r_base = detect_fall(*cw_iq(base), FS)
    r_fall = detect_fall(*cw_iq(fallen), FS)

    assert r_fall.energy_ratio > 5 * r_base.energy_ratio
