"""Breathing-pathology feature extraction and classifier.

Given a recovered chest-displacement signal, :func:`extract_features` computes a small,
interpretable feature vector, and :class:`PathologyClassifier` (a scikit-learn Random
Forest) labels it as one of the five patterns.

Features (index -> meaning):
    0  rate_bpm            dominant breathing rate (wide band, captures tachypnea)
    1  spectral_entropy    spectral flatness of the in-band signal (regular vs irregular)
    2  envelope_cov        coefficient of variation of the amplitude envelope
                           -> large for Cheyne-Stokes crescendo-decrescendo
    3  apnea_gap_fraction  fraction of time the envelope is near zero
                           -> large for obstructive apnea
    4  mean_amp            mean absolute band-passed amplitude
    5  max_amp             peak band-passed amplitude

The rate-defined patterns (normal / tachypnea / bradypnea) separate mainly on feature 0;
Cheyne-Stokes and obstructive apnea share a normal-ish carrier rate but separate on the
envelope features (2, 3).
"""

from __future__ import annotations

import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import detrend, hilbert
from sklearn.ensemble import RandomForestClassifier

from src.breathing_models import PATTERNS, chest_displacement
from src.cw_radar import cw_iq
from src.demodulation import recover_displacement
from src.noise import add_thermal_noise
from src.rate_extraction import bandpass, estimate_bpm

# Wider than the 0.1-0.5 Hz reporting band so tachypnea (>30 br/min) is still captured.
FEATURE_BAND = (0.1, 0.9)
_EPS = 1e-9


def extract_features(signal, fs: float) -> np.ndarray:
    """Compute the 6-element pathology feature vector from a displacement signal."""
    x = detrend(np.asarray(signal, dtype=np.float64), type="linear")
    xb = bandpass(x, fs, FEATURE_BAND[0], FEATURE_BAND[1])

    rate_bpm, _ = estimate_bpm(x, fs, FEATURE_BAND)

    env = np.abs(hilbert(xb))
    env_cov = float(env.std() / (env.mean() + _EPS))
    median_env = np.median(env)
    apnea_gap_fraction = float(np.mean(env < 0.25 * median_env))

    mean_amp = float(np.mean(np.abs(xb)))
    max_amp = float(np.max(np.abs(xb)))

    spec = np.abs(rfft(xb * np.hanning(len(xb)))) ** 2
    freqs = rfftfreq(len(xb), 1.0 / fs)
    band = (freqs >= FEATURE_BAND[0]) & (freqs <= FEATURE_BAND[1])
    p = spec[band]
    p = p / (p.sum() + _EPS)
    spectral_entropy = float(-np.sum(p * np.log(p + _EPS)) / np.log(len(p)))

    return np.array([rate_bpm, spectral_entropy, env_cov,
                     apnea_gap_fraction, mean_amp, max_amp])


# Per-pattern randomised parameter ranges for dataset generation.
_PARAM_RANGES = {
    "normal": dict(rate_bpm=(12, 20), amplitude_mm=(4, 12)),
    "tachypnea": dict(rate_bpm=(22, 40), amplitude_mm=(2, 5)),
    "bradypnea": dict(rate_bpm=(6, 11), amplitude_mm=(6, 12)),
    "cheyne_stokes": dict(carrier_bpm=(12, 20), amplitude_mm=(5, 10),
                          cycle_s=(45, 90), apnea_frac=(0.2, 0.4)),
    "obstructive_apnea": dict(rate_bpm=(12, 20), amplitude_mm=(4, 10),
                              breath_s=(15, 25), apnea_s=(8, 16)),
}


def build_dataset(n_per_class: int = 60, snr_db=(20.0, 10.0, 0.0),
                  duration: float = 120.0, fs: float = 200.0, seed: int = 0):
    """Generate a labelled feature dataset by simulating noisy radar trials.

    Each trial randomises the pattern's parameters, renders displacement, passes it
    through the CW radar + thermal noise + demodulation chain, and extracts features.

    Returns ``(X, y)`` with ``X`` shape ``(n_per_class*len(PATTERNS), 6)``.
    """
    rng = np.random.default_rng(seed)
    snr_options = np.atleast_1d(snr_db)
    X, y = [], []
    for pattern in PATTERNS:
        ranges = _PARAM_RANGES[pattern]
        for _ in range(n_per_class):
            params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}
            _, d = chest_displacement(pattern, duration=duration, fs=fs, **params)
            i, q = cw_iq(d)
            snr = float(rng.choice(snr_options))
            i, q = add_thermal_noise(i, q, snr, seed=int(rng.integers(1 << 31)))
            rec = recover_displacement(i, q)
            X.append(extract_features(rec, fs))
            y.append(pattern)
    return np.array(X), np.array(y)


class PathologyClassifier:
    """Random-forest classifier over the breathing-pathology feature vector."""

    def __init__(self, n_estimators: int = 200, random_state: int = 0):
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            random_state=random_state)

    def fit(self, X, y) -> "PathologyClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y) -> float:
        return float(self.model.score(X, y))
