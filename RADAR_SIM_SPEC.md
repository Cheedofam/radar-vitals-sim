# RADAR_SIM_SPEC.md — Radar-Based Breathing & Fall Simulation (Lite, No Isaac Sim)

## 1. Objective

Build a physics-grounded **simulation** of FMCW and CW radar sensing of a person's chest
motion and gross body movement, without requiring Isaac Sim or any paid/licensed software.
The system must: (1) generate synthetic radar returns from a simulated moving chest wall
exhibiting multiple clinically realistic breathing patterns, (2) recover breathing rate from
those returns via I/Q demodulation and FFT, and (3) classify which breathing pathology
produced a given signal using a simple trained classifier. This is a scaled-down, buildable
version of the Isaac Sim proposal previously discussed with Professor Bolic, intended as a
working proof of concept to bring to that conversation — not a replacement for the full
simulation platform.

## 2. Scope

**In scope**
- Synthetic chest-wall displacement generator for 5 breathing patterns: normal, Cheyne-Stokes
  (heart failure), obstructive apnea, tachypnea, bradypnea
- CW Doppler radar model: generate I/Q baseband signal from chest displacement via phase
  modulation physics
- FMCW radar model (simplified): generate a range-time signal showing chest wall range bin
  oscillation, extract range-Doppler style output
- Signal processing: arctangent demodulation, phase unwrapping, bandpass filtering, FFT
  breathing-rate extraction
- A classifier (start with simple threshold/rule-based or a small scikit-learn model) that
  labels which of the 5 pathologies a given signal segment shows
- A basic "fall" scenario: simulate a sudden large-amplitude Doppler event distinct from
  breathing, and a simple energy-threshold detector to flag it
- Validation report: accuracy of breathing-rate extraction (Hz error) and pathology
  classification (confusion matrix) across many simulated trials with noise added

**Out of scope (do not build)**
- Isaac Sim / USD / PhysX room environment — this comes later if Bolic approves the full project
- Real hardware radar integration
- ROS2 bridge (mention in README as future work, don't build)
- Multi-person or multi-room scenarios

## 3. Tech Stack (100% free/open-source)

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Physics/signal math | NumPy, SciPy |
| Radar-specific simulation | `radarsimpy` (free/open Python radar simulator) — if install issues arise, fall back to hand-rolled phase-modulation math in NumPy, which is scientifically equivalent for this scope |
| Classifier | scikit-learn (start simple: SVM or Random Forest on FFT-derived features) |
| Visualization | Matplotlib (range-Doppler maps, I/Q constellation, breathing waveforms) |
| Testing | pytest with known-frequency synthetic inputs |

## 4. Physics Background (for reference while building)

- **CW Doppler radar**: transmits a continuous single-frequency wave; a moving target
  (chest wall) phase-modulates the reflected signal proportional to displacement. Baseband
  I/Q signal: `I(t) = A*cos(4*pi*d(t)/lambda)`, `Q(t) = A*sin(4*pi*d(t)/lambda)`, where
  `d(t)` is chest displacement and `lambda` is the radar wavelength (for 24GHz, lambda ≈
  12.5mm). Breathing rate is recovered via `arctan2(Q,I)`, phase unwrapping, then FFT.
- **FMCW radar**: transmits a frequency-swept chirp; beat frequency between transmit and
  receive encodes range. For this simplified version, simulate range bin oscillation
  directly from `d(t)` rather than full chirp generation — this captures the same
  breathing-detection principle at far lower implementation complexity.
- **Breathing waveform models** (chest displacement `d(t)` in mm, use literature-typical
  parameters, document sources used):
  - Normal: ~12-20 breaths/min, roughly sinusoidal, amplitude ~4-12mm
  - Tachypnea: >20 breaths/min, same shape, faster
  - Bradypnea: <12 breaths/min
  - Cheyne-Stokes: cyclic crescendo-decrescendo amplitude pattern with apneic pauses,
    period ~45-90s
  - Obstructive apnea: normal breathing interrupted by periods of drastically reduced/absent
    chest motion despite continued effort (paradoxical motion is a stretch goal, note as such)

## 5. Repository Structure

```
radar-vitals-sim/
├── README.md
├── requirements.txt
├── run_demo.py                  # generates a signal, runs full pipeline, plots results
├── src/
│   ├── breathing_models.py       # chest displacement generators for all 5 patterns
│   ├── cw_radar.py               # I/Q signal generation from displacement
│   ├── fmcw_radar.py             # simplified range-oscillation model
│   ├── demodulation.py           # arctangent demod, phase unwrap, filtering
│   ├── rate_extraction.py        # FFT-based breathing rate estimation
│   ├── fall_detection.py         # energy-threshold fall event detector
│   └── classifier.py             # feature extraction + pathology classifier
├── tests/
│   ├── test_breathing_models.py
│   ├── test_demodulation.py      # known displacement -> verify recovered rate
│   └── test_classifier.py
├── validation/
│   ├── run_validation.py         # batch-generates N noisy trials per pathology
│   ├── results.csv
│   └── validation_report.md      # rate-extraction error + confusion matrix + discussion
└── docs/
    └── physics_notes.md          # the radar physics + parameter sources, written up cleanly
```

## 6. Build Phases (do these in order, commit after each)

**Phase 1 — Breathing models**
Implement `breathing_models.py` generating `d(t)` for all 5 patterns given a duration and
sample rate. Plot each to visually sanity-check shape. Commit.

**Phase 2 — CW radar + demodulation**
Implement I/Q generation from `d(t)`, then arctangent demodulation to recover displacement.
Unit test: feed a known sinusoidal `d(t)`, verify demodulated output matches within
tolerance. Commit only once this round-trip test passes — this is the core physics
validation of the whole project.

**Phase 3 — Breathing rate extraction**
FFT-based rate extraction from demodulated signal, with bandpass filtering to isolate the
breathing frequency band. Test against all 5 pathology types with known ground-truth rates
(for Cheyne-Stokes/apnea, define what "rate" means during the cyclic pattern and document
the choice).

**Phase 4 — FMCW simplified model**
Implement the simplified range-oscillation FMCW model, show it recovers equivalent
breathing rate to the CW path (cross-validation between the two radar types is a nice
result to have).

**Phase 5 — Noise + realism**
Add realistic noise (thermal noise floor, random phase jitter) to all signals at multiple
SNR levels, so the validation isn't testing a noiseless toy case.

**Phase 6 — Classifier**
Extract features (dominant frequency, spectral shape, envelope variance, presence of
apneic gaps) from noisy signals and train a classifier (scikit-learn) to label pathology
type. Use a proper train/test split. Report accuracy and confusion matrix honestly, including
where it struggles (e.g., Cheyne-Stokes vs. normal at low SNR).

**Phase 7 — Fall detection**
Add a large-amplitude, short-duration Doppler event generator (simulating a fall) distinct
from breathing-frequency content, and a simple energy-threshold or short-time-energy
detector to flag it. This does not need to be sophisticated — the point is demonstrating
you understand the detection principle.

**Phase 8 — Validation & documentation**
Run `validation/run_validation.py` across many trials per pathology at multiple SNR levels.
Produce `results.csv` and `validation_report.md` with rate-extraction error (Hz/bpm) and
classifier confusion matrix. Write `docs/physics_notes.md` explaining the radar physics
used, citing where the breathing-parameter values came from. Write the main README framing
this explicitly as a precursor/proof-of-concept for the full Isaac Sim proposal, with a
clear "Next Steps" section listing what the full simulation platform would add.

## 7. Definition of Done

- `python run_demo.py` runs end-to-end from a fresh clone and produces plots + printed
  breathing rate estimates for all 5 pathologies
- All pytest tests pass, including the CW round-trip physics validation test
- `validation_report.md` contains real numbers from real runs, not placeholder text
- README clearly explains this is a lite/proof-of-concept version of the Isaac Sim proposal
  and what the full platform would add
- Code committed incrementally with descriptive messages

## 8. Resume bullet (for reference, do not put in README as-is)

"Built a radar physics simulation modeling CW and FMCW sensing of clinically realistic
breathing patterns (normal, Cheyne-Stokes, apnea, tachypnea, bradypnea); implemented I/Q
demodulation and FFT-based rate extraction, achieving [X]% classification accuracy across
noise conditions, as a proof-of-concept for a contactless elderly monitoring research
platform."
