# PLAN.md — radar-vitals-sim

Planning document for the radar breathing/fall simulation defined in
[`RADAR_SIM_SPEC.md`](./RADAR_SIM_SPEC.md) — a lite, buildable proof-of-concept precursor to
the full Isaac Sim proposal. Covers the milestone schedule, pre-Phase-1 dependency checklist,
and the top risks with fallbacks.

---

## 1. Milestone Plan

One work session per Build Phase (spec Section 6 — **8 phases**), worked part-time at ~2–3
sessions/week starting **Fri 2026-07-24**. Estimated total ≈ **24 hours** across ~3 weeks.
Commit after each phase, as the spec requires. This schedule is independent of `rppg-vitals`
and can be interleaved with it.

| Session | Build Phase | Deliverable | Est. hours | Target date |
|:--:|---|---|:--:|---|
| 1 | **Phase 1 — Breathing models** | `d(t)` generators for all 5 patterns (normal, Cheyne-Stokes, apnea, tachypnea, bradypnea); visual sanity plots | 2h | Fri 2026-07-24 |
| 2 | **Phase 2 — CW radar + demod** | I/Q generation from `d(t)` + arctangent demod; **passing** round-trip test (core physics validation) | 4h | Tue 2026-07-28 |
| 3 | **Phase 3 — Rate extraction** | Bandpass + FFT breathing-rate estimation; tested across all 5 patterns w/ documented "rate" definition for cyclic cases | 3h | Thu 2026-07-31 |
| 4 | **Phase 4 — FMCW simplified** | Range-oscillation model; cross-validate rate against CW path | 2h | Mon 2026-08-03 |
| 5 | **Phase 5 — Noise + realism** | Thermal noise floor + phase jitter at multiple SNR levels | 3h | Thu 2026-08-06 |
| 6 | **Phase 6 — Classifier** | FFT/envelope feature extraction + scikit-learn model; train/test split; honest accuracy + confusion matrix | 4h | Mon 2026-08-10 |
| 7 | **Phase 7 — Fall detection** | Large-amplitude Doppler event generator + energy-threshold detector | 2h | Thu 2026-08-13 |
| 8 | **Phase 8 — Validation & docs** | `run_validation.py` batch trials → `results.csv`; `validation_report.md`; `physics_notes.md`; README framing this as Isaac Sim precursor + Next Steps | 4h | Sat 2026-08-15 |

*Dates are targets, not deadlines. Phase 2 (round-trip physics) is the critical path — if time
is short, protect it; it gates the correctness of everything downstream.*

---

## 2. Dependency Checklist (verify before Phase 1)

- [x] Python 3.10.11 present (`C:\Python310\python.exe`)
- [x] git available (2.54.0)
- [x] Virtual environment created (`venv/`)
- [x] `requirements.txt` installs cleanly *(status recorded in §3.1 after VERIFY)*
- [x] Core deps import without error: `numpy`, `scipy`, `sklearn`, `matplotlib`, `pytest`
- [~] `radarsimpy` — **optional**; fallback to NumPy math applied by default (see Risk 1 / §3.1)

---

## 3. Risk List

| # | Risk | Fallback |
|:--:|---|---|
| 1 | **`radarsimpy` install failure** — not a plain PyPI wheel; needs platform-specific prebuilt binaries / a build step. Most likely install problem. | **Spec-sanctioned fallback (Section 3):** hand-rolled phase-modulation math in NumPy — `I=A·cos(4π·d/λ)`, `Q=A·sin(4π·d/λ)` — scientifically equivalent for this scope. `radarsimpy` is left commented out of `requirements.txt`; the project builds on NumPy by default, so a fresh clone still installs cleanly. |
| 2 | **Classifier confuses Cheyne-Stokes vs. normal at low SNR** (spec explicitly anticipates this). | Start rule-based/threshold; add discriminative features (apneic-gap presence, envelope variance, spectral shape); report the confusion honestly in `validation_report.md` rather than hiding it. |
| 3 | **Ambiguous "breathing rate" for Cheyne-Stokes / apnea** — no single ground-truth rate during cyclic/paused patterns. | Define and **document** the chosen convention in Phase 3 (e.g., instantaneous intra-burst rate vs. crescendo-cycle period); apply consistently in tests and validation. |
| 4 | **Noise model not realistic** — arbitrary thermal-noise/phase-jitter parameters make validation a toy case. | Sweep multiple SNR levels (Phase 5); document parameter choices and literature sources in `docs/physics_notes.md`; report metrics per-SNR, not a single figure. |
| 5 | **I/Q round-trip physics bugs** — sign errors, phase-unwrap discontinuities, λ scaling. | The Phase 2 round-trip unit test (known `d(t)` → recovered displacement within tolerance) **gates the commit** per spec; do not proceed to Phase 3 until it passes. |

### 3.1 VERIFY outcome (2026-07-22)
Core `pip install -r requirements.txt` **succeeded with no conflicts**; all five core packages
import cleanly. Resolved versions:

| Package | Version | Package | Version |
|---|---|---|---|
| NumPy | 2.2.6 | Matplotlib | 3.10.9 |
| SciPy | 1.15.3 | pytest | 9.1.1 |
| scikit-learn | 1.7.2 | | |

**Risk 1 confirmed and fallback applied:** `pip install radarsimpy` failed —
`No matching distribution found` (it is not published on PyPI). Per spec Section 3, the project
uses the hand-rolled NumPy phase-modulation math instead. `radarsimpy` stays commented out in
`requirements.txt`, so a fresh clone installs cleanly with no failing dependency. No action
needed from the developer; the fallback is the intended default path.
