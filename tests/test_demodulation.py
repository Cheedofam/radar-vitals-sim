"""CW radar I/Q + arctangent demodulation round-trip tests.

This is the core physics validation of the whole project: a known chest displacement
d(t), passed through the CW radar model and back through demodulation, must be recovered
to within a tight tolerance.
"""

import numpy as np

from src.cw_radar import CARRIER_24GHZ, cw_iq, wavelength
from src.demodulation import demod_phase, recover_displacement


def test_wavelength_24ghz():
    assert abs(wavelength(CARRIER_24GHZ) - 0.0125) < 1e-4  # ~12.5 mm


def test_cw_roundtrip_recovers_small_displacement():
    fs = 1000.0
    t = np.arange(0, 30, 1 / fs)
    d = 6.0 * np.sin(2 * np.pi * 0.25 * t)  # 6 mm, 15 br/min

    i, q = cw_iq(d)
    rec = recover_displacement(i, q)

    rec -= rec.mean()
    d0 = d - d.mean()
    assert np.sqrt(np.mean((rec - d0) ** 2)) < 0.05  # < 0.05 mm RMS


def test_cw_roundtrip_multiwrap():
    """12 mm swing produces >2*pi of phase, so demod must unwrap correctly."""
    fs = 1000.0
    t = np.arange(0, 20, 1 / fs)
    d = 12.0 * np.sin(2 * np.pi * 0.3 * t)

    i, q = cw_iq(d)
    # The wrapped phase spans the full [-pi, pi], confirming wrapping really happens.
    wrapped = np.arctan2(q, i)
    assert wrapped.max() > 3.0 and wrapped.min() < -3.0

    rec = recover_displacement(i, q)
    rec -= rec.mean()
    d0 = d - d.mean()
    assert np.sqrt(np.mean((rec - d0) ** 2)) < 0.05


def test_demod_phase_is_unwrapped():
    fs = 1000.0
    t = np.arange(0, 10, 1 / fs)
    d = 10.0 * np.sin(2 * np.pi * 0.25 * t)
    i, q = cw_iq(d)

    phase = demod_phase(i, q)
    # Unwrapped phase is smooth: no jumps near 2*pi between adjacent samples.
    assert np.max(np.abs(np.diff(phase))) < 1.0
