"""Arctangent demodulation: recover chest displacement from radar I/Q.

The instantaneous phase of the baseband signal is ``atan2(Q, I)``, wrapped to
``[-pi, pi]``. Because a chest excursion spans more than one turn of phase at 24 GHz, the
wrapped phase must be **unwrapped** into a continuous curve before it can be scaled back
to displacement via ``d = phase * lambda / (4*pi)``.
"""

from __future__ import annotations

import numpy as np

from src.cw_radar import CARRIER_24GHZ, wavelength


def demod_phase(i, q) -> np.ndarray:
    """Continuous (unwrapped) baseband phase from I/Q, in radians."""
    return np.unwrap(np.arctan2(np.asarray(q, float), np.asarray(i, float)))


def recover_displacement(i, q, carrier_hz: float = CARRIER_24GHZ) -> np.ndarray:
    """Recover chest displacement (mm) from I/Q via unwrapped-phase demodulation."""
    phase = demod_phase(i, q)
    d_m = phase * wavelength(carrier_hz) / (4.0 * np.pi)
    return d_m * 1000.0
