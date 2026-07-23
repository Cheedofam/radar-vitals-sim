"""End-to-end demo of the radar-vitals simulation.

For every breathing pattern it renders chest displacement, passes it through the CW radar
model + thermal noise + demodulation, estimates the breathing rate, and (using a small
trained classifier) predicts the pathology. It also demonstrates fall detection. Results
are printed and a summary figure is written to ``docs/demo.png``.

Usage:
    python run_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.breathing_models import PATTERNS, chest_displacement
from src.classifier import PathologyClassifier, build_dataset, extract_features
from src.cw_radar import cw_iq
from src.demodulation import recover_displacement
from src.fall_detection import add_fall_event, detect_fall
from src.noise import add_thermal_noise
from src.rate_extraction import estimate_bpm

FS = 200.0
DURATION = 120.0
SNR_DB = 20.0
RATE_BAND = (0.1, 0.9)

# Nominal ground-truth rate per pattern, for the printed comparison.
TRUE_BPM = {"normal": 15, "tachypnea": 30, "bradypnea": 8,
            "cheyne_stokes": 15, "obstructive_apnea": 15}


def main() -> int:
    rng = np.random.default_rng(0)

    print("Training pathology classifier on simulated trials ...")
    X, y = build_dataset(n_per_class=30, snr_db=(20.0, 10.0), duration=90, fs=FS, seed=0)
    clf = PathologyClassifier(random_state=0).fit(X, y)

    print(f"\nBreathing-rate + pathology estimates (CW radar, {SNR_DB:.0f} dB SNR):\n")
    print(f"  {'pattern':<20}{'true bpm':>9}{'est bpm':>9}   predicted")
    print("  " + "-" * 55)

    recovered = {}
    for pattern in PATTERNS:
        _, d = chest_displacement(pattern, duration=DURATION, fs=FS,
                                  rate_bpm=TRUE_BPM[pattern], carrier_bpm=TRUE_BPM[pattern])
        i, q = cw_iq(d)
        i, q = add_thermal_noise(i, q, SNR_DB, seed=int(rng.integers(1 << 31)))
        rec = recover_displacement(i, q)
        recovered[pattern] = rec
        bpm, _ = estimate_bpm(rec, FS, RATE_BAND)
        pred = clf.predict(extract_features(rec, FS)[None, :])[0]
        flag = "  <-- misclassified" if pred != pattern else ""
        print(f"  {pattern:<20}{TRUE_BPM[pattern]:>9}{bpm:>9.1f}   {pred}{flag}")

    # Fall-detection demo.
    _, d = chest_displacement("normal", duration=60, fs=1000.0)
    d_fall = add_fall_event(d, 1000.0, t_start=30.0)
    r_norm = detect_fall(*cw_iq(d), 1000.0)
    r_fall = detect_fall(*cw_iq(d_fall), 1000.0)
    print("\nFall detection:")
    print(f"  normal breathing : detected={r_norm.detected} (energy ratio {r_norm.energy_ratio:.1f})")
    print(f"  breathing + fall : detected={r_fall.detected} at t={r_fall.t_peak_s:.1f}s "
          f"(energy ratio {r_fall.energy_ratio:.1f})")

    _make_figure(recovered)
    print("\nSaved figure to docs/demo.png")
    return 0


def _make_figure(recovered: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from numpy.fft import rfft, rfftfreq

    from src.rate_extraction import bandpass

    fig, axes = plt.subplots(len(PATTERNS), 2, figsize=(11, 12))
    win = int(40 * FS)
    for row, pattern in enumerate(PATTERNS):
        rec = recovered[pattern]
        seg = bandpass(rec[:win] - np.mean(rec[:win]), FS, *RATE_BAND)
        t = np.arange(len(seg)) / FS
        axes[row, 0].plot(t, seg, color="#1f4e79", lw=0.9)
        axes[row, 0].set_ylabel(pattern.replace("_", "\n"), fontsize=9)
        if row == 0:
            axes[row, 0].set_title("Recovered displacement (40 s, band-passed)")

        spec = np.abs(rfft(seg * np.hanning(len(seg))))
        freqs = rfftfreq(len(seg), 1 / FS) * 60
        m = (freqs >= 5) & (freqs <= 45)
        bpm, _ = estimate_bpm(rec, FS, RATE_BAND)
        axes[row, 1].plot(freqs[m], spec[m], color="#111")
        axes[row, 1].axvline(bpm, color="#c1121f", ls="--")
        axes[row, 1].set_title(f"spectrum -> {bpm:.0f} br/min", fontsize=9)
        if row == len(PATTERNS) - 1:
            axes[row, 0].set_xlabel("time (s)")
            axes[row, 1].set_xlabel("breaths/min")
    fig.suptitle("radar-vitals-sim: recovered breathing per pathology (CW, 20 dB SNR)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/demo.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
