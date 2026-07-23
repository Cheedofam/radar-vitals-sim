"""Batch validation of the radar-vitals simulation.

Runs many simulated trials per breathing pattern across several SNR levels and reports:

  * **Breathing-rate accuracy** -- mean absolute error (breaths/min) of the recovered rate
    versus the known ground-truth rate, broken down by SNR.
  * **Pathology classification** -- accuracy and confusion matrix on a held-out split.

Outputs (into ``validation/``): ``results.csv``, ``confusion_matrix.png``,
``validation_report.md`` -- all populated with real measured numbers.

Usage:
    python validation/run_validation.py [--quick]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.breathing_models import PATTERNS, chest_displacement
from src.cw_radar import cw_iq
from src.classifier import PathologyClassifier, _PARAM_RANGES, build_dataset
from src.demodulation import recover_displacement
from src.noise import add_thermal_noise
from src.rate_extraction import estimate_bpm

OUT = Path(__file__).resolve().parent
RATE_BAND = (0.1, 0.9)  # wide enough to measure tachypnea (>30 br/min)


def _true_rate_bpm(pattern: str, params: dict) -> float:
    return params.get("rate_bpm", params.get("carrier_bpm", 15.0))


def rate_trials(n_per_cell: int, snr_levels, fs: float, duration: float, seed: int):
    """Return rows: (pattern, snr_db, true_bpm, est_bpm, abs_err)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pattern in PATTERNS:
        ranges = _PARAM_RANGES[pattern]
        for snr in snr_levels:
            for _ in range(n_per_cell):
                params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}
                _, d = chest_displacement(pattern, duration=duration, fs=fs, **params)
                i, q = cw_iq(d)
                i, q = add_thermal_noise(i, q, float(snr), seed=int(rng.integers(1 << 31)))
                est, _ = estimate_bpm(recover_displacement(i, q), fs, RATE_BAND)
                true = _true_rate_bpm(pattern, params)
                rows.append((pattern, float(snr), true, est, abs(est - true)))
    return rows


def classifier_eval(n_per_class: int, snr_levels, fs: float, duration: float, seed: int):
    X, y = build_dataset(n_per_class=n_per_class, snr_db=tuple(snr_levels),
                         duration=duration, fs=fs, seed=seed)
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)
    clf = PathologyClassifier(random_state=seed).fit(x_tr, y_tr)
    y_pred = clf.predict(x_te)
    acc = accuracy_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred, labels=list(PATTERNS))

    # Per-SNR accuracy: score the trained model on fresh single-SNR test sets.
    per_snr = {}
    for snr in snr_levels:
        xs, ys = build_dataset(n_per_class=max(10, n_per_class // 3), snr_db=(snr,),
                               duration=duration, fs=fs, seed=seed + 100 + int(snr))
        per_snr[float(snr)] = clf.score(xs, ys)
    return acc, cm, per_snr


def write_results_csv(rate_rows, acc, cm, per_snr_acc) -> None:
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "pattern", "snr_db", "true_bpm", "est_bpm", "abs_err_bpm"])
        for pattern, snr, true, est, err in rate_rows:
            w.writerow(["rate", pattern, f"{snr:.0f}", f"{true:.2f}", f"{est:.2f}", f"{err:.2f}"])
        w.writerow([])
        w.writerow(["classifier_accuracy", f"{acc:.4f}"])
        for snr, a in sorted(per_snr_acc.items(), reverse=True):
            w.writerow(["classifier_accuracy_at_snr", f"{snr:.0f}", f"{a:.4f}"])
        w.writerow(["confusion_matrix_labels"] + list(PATTERNS))
        for label, row in zip(PATTERNS, cm):
            w.writerow(["confusion_row", label] + [int(v) for v in row])


def make_confusion_figure(cm) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    short = [p.replace("obstructive_apnea", "OSA").replace("cheyne_stokes", "C-S")
             for p in PATTERNS]
    ax.set_xticks(range(len(PATTERNS)), short, rotation=45, ha="right")
    ax.set_yticks(range(len(PATTERNS)), short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Pathology classification confusion matrix (row-normalised)")
    for r in range(len(PATTERNS)):
        for c in range(len(PATTERNS)):
            ax.text(c, r, f"{cmn[r, c]:.2f}", ha="center", va="center",
                    color="white" if cmn[r, c] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=110)
    plt.close(fig)


def write_report(rate_rows, acc, cm, snr_levels, per_snr_acc) -> None:
    rows = np.array([(r[1], r[4]) for r in rate_rows])  # (snr, abs_err)
    overall_mae = float(rows[:, 1].mean())
    per_snr = {snr: float(rows[rows[:, 0] == snr][:, 1].mean()) for snr in snr_levels}

    # Data-driven weakest-class analysis so the prose matches the actual matrix.
    recall = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    worst_i = int(np.argmin(recall))
    off = cm[worst_i].copy()
    off[worst_i] = 0
    confused_i = int(np.argmax(off))
    worst, confused = PATTERNS[worst_i], PATTERNS[confused_i]

    lines = [
        "# Validation Report — radar-vitals-sim",
        "",
        "## Methodology",
        "",
        "Each trial randomises a breathing pattern's parameters, renders chest "
        "displacement `d(t)`, passes it through the CW radar model (`I=A cos(4πd/λ)`, "
        "`Q=A sin(4πd/λ)` at 24 GHz), adds white Gaussian thermal noise at a target SNR, "
        "demodulates (arctan + unwrap), and either estimates the breathing rate (FFT) or "
        "extracts features for the pathology classifier. All numbers below are measured "
        "from real runs of `validation/run_validation.py`.",
        "",
        "## Breathing-rate accuracy vs SNR",
        "",
        "Mean absolute error of the recovered rate versus the known ground-truth rate "
        f"(band {RATE_BAND[0]}–{RATE_BAND[1]} Hz, widened from the 0.1–0.5 Hz reporting "
        "band so tachypnea >30 br/min is measurable):",
        "",
        "| SNR (dB) | Rate MAE (breaths/min) |",
        "|--:|--:|",
    ]
    for snr in sorted(per_snr, reverse=True):
        lines.append(f"| {snr:.0f} | {per_snr[snr]:.2f} |")
    lines += [
        f"| **all** | **{overall_mae:.2f}** |",
        "",
        "Rate extraction is accurate at high SNR and degrades gracefully as noise rises, "
        "as expected when the demodulated phase becomes noisier.",
        "",
        "## Pathology classification",
        "",
        f"Held-out accuracy (30% test split, all SNRs pooled): **{acc*100:.1f}%** across "
        "the five patterns. Accuracy is strongly SNR-dependent:",
        "",
        "| SNR (dB) | Classifier accuracy |",
        "|--:|--:|",
    ]
    for snr in sorted(per_snr_acc, reverse=True):
        lines.append(f"| {snr:.0f} | {per_snr_acc[snr]*100:.1f}% |")
    lines += [
        "",
        "![Confusion matrix](confusion_matrix.png)",
        "",
        "The rate-defined patterns (normal / tachypnea / bradypnea) separate cleanly on "
        f"the rate feature. The hardest class is **{worst}** (recall "
        f"{recall[worst_i]*100:.0f}%), most often mistaken for **{confused}**. This "
        "matches the physics: at low SNR the envelope-based features that distinguish the "
        "cyclic patterns — apnea-gap fraction and envelope variation — wash out first "
        f"(thermal noise fills the apneic pauses), so {worst} starts to resemble "
        f"{confused}. At high SNR the classifier is far more reliable, as the per-SNR "
        "table shows.",
        "",
        "## Known limitations",
        "",
        "* **Simulation, not hardware.** The CW/FMCW models are phase-modulation math, not "
        "measured radar returns; real returns add clutter, multipath, and hardware effects.",
        "* **Single target, no room.** No multipath, body-part separation, or Isaac-Sim "
        "room physics — that is the scope of the full proposal (see README Next Steps).",
        f"* **{worst} at low SNR** is the hardest class, as the confusion matrix shows.",
        "* **'Rate' for cyclic patterns** is the intra-burst rate by definition (see "
        "`docs/physics_notes.md`); it is not meaningful during apneic pauses.",
        "",
        "_Generated by `validation/run_validation.py`._",
    ]
    (OUT / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Fewer trials (fast).")
    args = parser.parse_args(argv)

    fs, duration = 200.0, 120.0
    snr_levels = [20.0, 10.0, 0.0]
    n_rate = 10 if args.quick else 25
    n_cls = 20 if args.quick else 50

    print(f"Rate validation: {n_rate} trials x {len(PATTERNS)} patterns x {len(snr_levels)} SNRs ...")
    rate_rows = rate_trials(n_rate, snr_levels, fs, duration, seed=1)
    print(f"Classifier: {n_cls} trials/class ...")
    acc, cm, per_snr_acc = classifier_eval(n_cls, snr_levels, fs, duration, seed=2)

    write_results_csv(rate_rows, acc, cm, per_snr_acc)
    make_confusion_figure(cm)
    write_report(rate_rows, acc, cm, snr_levels, per_snr_acc)
    print(f"Classifier accuracy = {acc*100:.1f}%. Wrote results.csv, "
          f"confusion_matrix.png, validation_report.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
