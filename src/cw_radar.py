"""Continuous-wave (CW) Doppler radar model.

A CW radar transmits a single-frequency wave; a moving target (the chest wall) phase-
modulates the reflected signal in proportion to its displacement. The baseband I/Q signal
is (de Haan-style phase model)::

    I(t) = A * cos(4*pi*d(t)/lambda)
    Q(t) = A * sin(4*pi*d(t)/lambda)

where ``d(t)`` is the chest displacement and ``lambda = c / f_carrier`` the wavelength.
For 24 GHz, ``lambda`` is about 12.5 mm, so a several-millimetre chest excursion produces
more than one full turn of phase -- which is why demodulation must unwrap.

This is the hand-rolled NumPy phase-modulation model that stands in for ``radarsimpy``
(not installable via pip); it is scientifically equivalent for this scope.
"""

from __future__ import annotations

import numpy as np

C = 299_792_458.0          # speed of light, m/s
CARRIER_24GHZ = 24e9       # Hz


def wavelength(carrier_hz: float = CARRIER_24GHZ) -> float:
    """Radar wavelength (m) for a given carrier frequency."""
    return C / carrier_hz


def cw_iq(d_mm, carrier_hz: float = CARRIER_24GHZ, amplitude: float = 1.0,
          phase0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate baseband ``(I, Q)`` from a chest-displacement signal ``d_mm`` (mm)."""
    d_m = np.asarray(d_mm, dtype=np.float64) / 1000.0
    phase = 4.0 * np.pi * d_m / wavelength(carrier_hz) + phase0
    return amplitude * np.cos(phase), amplitude * np.sin(phase)
