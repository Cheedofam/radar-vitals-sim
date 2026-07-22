"""FFT-based breathing-rate extraction from a recovered displacement signal.

The signal is detrended, band-passed to the respiratory band (0.1-0.5 Hz = 6-30
breaths/min), Hann-windowed, zero-padded, and FFT'd; the dominant in-band peak is
located to sub-bin accuracy by parabolic interpolation.

**Definition of "rate" for cyclic patterns.** Cheyne-Stokes and obstructive apnea have no
single breathing rate -- their amplitude waxes, wanes, and pauses. The rate reported here
is the **intra-burst rate**: the carrier frequency of breathing while the patient is
actively breathing. The slow crescendo/apnea envelope (period tens of seconds, i.e. well
below 0.1 Hz) is removed by the band-pass, so the FFT peak is the carrier. The *pattern*
itself is identified separately by the classifier from envelope-shape features.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, detrend, sosfiltfilt

RR_BAND = (0.1, 0.5)  # 6-30 breaths/min


def bandpass(x, fs: float, low: float = RR_BAND[0], high: float = RR_BAND[1],
             order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass (SOS form)."""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def _parabolic_peak(spectrum: np.ndarray, k: int, freqs: np.ndarray) -> float:
    if k <= 0 or k >= len(spectrum) - 1:
        return float(freqs[k])
    a, b, c = spectrum[k - 1], spectrum[k], spectrum[k + 1]
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return float(freqs[k] + delta * (freqs[1] - freqs[0]))


def estimate_rate(signal, fs: float, band: tuple[float, float] = RR_BAND,
                  peak_bw: float = 0.05) -> tuple[float, float]:
    """Estimate the dominant frequency (Hz) within ``band`` and its spectral SNR."""
    x = detrend(np.asarray(signal, dtype=np.float64), type="linear")
    x = bandpass(x, fs, band[0], band[1])
    x = x - x.mean()
    n = len(x)
    if n < 8:
        return 0.0, 0.0
    nfft = int(2 ** np.ceil(np.log2(n * 4)))
    spectrum = np.abs(rfft(x * np.hanning(n), nfft))
    freqs = rfftfreq(nfft, 1.0 / fs)

    in_band = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    if in_band.size == 0:
        return 0.0, 0.0
    k = int(in_band[np.argmax(spectrum[in_band])])
    freq = _parabolic_peak(spectrum, k, freqs)

    power = spectrum ** 2
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    near = (freqs >= freq - peak_bw) & (freqs <= freq + peak_bw)
    near_power = power[near & band_mask].sum()
    rest = power[band_mask].sum() - near_power
    snr = float(near_power / rest) if rest > 1e-12 else float("inf")
    return freq, snr


def estimate_bpm(signal, fs: float, band: tuple[float, float] = RR_BAND) -> tuple[float, float]:
    """Estimate breathing rate in breaths/min (and spectral SNR)."""
    rate, snr = estimate_rate(signal, fs, band)
    return rate * 60.0, snr
