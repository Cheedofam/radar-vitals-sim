"""Fall-event simulation and detection.

A fall is a large, brief body movement -- very different from the small, slow, periodic
chest motion of breathing. In the radar signal it appears as a short burst of very high
Doppler (phase rate). :func:`add_fall_event` injects such a transient into a displacement
signal, and :func:`detect_fall` flags it using the short-time energy of the demodulated
phase velocity, relative to the breathing baseline.

The detector is intentionally simple: the goal is to demonstrate the detection principle
(energy thresholding on velocity), not to build a sophisticated fall classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from src.demodulation import demod_phase


@dataclass
class FallResult:
    """Outcome of fall detection."""

    detected: bool
    t_peak_s: float          # time of peak short-time energy
    peak_energy: float       # peak short-time energy of phase velocity
    energy_ratio: float      # peak / median energy (dimensionless)


def add_fall_event(d_mm, fs: float, t_start: float,
                   magnitude_mm: float = 300.0, duration_s: float = 0.3) -> np.ndarray:
    """Add a fast, large displacement transient (a fall) to ``d_mm`` at ``t_start``."""
    d = np.array(d_mm, dtype=np.float64).copy()
    t = np.arange(len(d)) / fs
    mask = (t >= t_start) & (t < t_start + duration_s)
    local = (t[mask] - t_start) / duration_s
    d[mask] += magnitude_mm * np.sin(np.pi * local)  # smooth up-and-down lunge
    return d


def detect_fall(i, q, fs: float, win_s: float = 0.3,
                ratio_thresh: float = 20.0) -> FallResult:
    """Detect a fall from I/Q via short-time energy of the demodulated phase velocity.

    The phase velocity (rad/s) is squared and smoothed over a ``win_s`` window. A fall
    produces a peak far above the breathing baseline; detection fires when the peak
    exceeds ``ratio_thresh`` times the median short-time energy.
    """
    phase = demod_phase(i, q)
    velocity = np.diff(phase) * fs
    win = max(1, int(win_s * fs))
    energy = uniform_filter1d(velocity ** 2, size=win)

    peak_idx = int(np.argmax(energy))
    peak = float(energy[peak_idx])
    median = float(np.median(energy)) + 1e-12
    ratio = peak / median
    return FallResult(detected=ratio > ratio_thresh,
                      t_peak_s=peak_idx / fs,
                      peak_energy=peak,
                      energy_ratio=ratio)
