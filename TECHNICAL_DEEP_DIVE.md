# radar-vitals-sim — Technical Deep Dive

A plain-language explanation of what this project does, the ideas behind it, and how it is
built. It assumes **no prior knowledge** — every technical term is defined the first time it
appears.

---

## 1. What this project does

It is a **simulation** — a computer model — of how a **radar** could measure a person's
**breathing**, and tell apart different *kinds* of breathing linked to illnesses. It also detects
a **fall**.

Two clarifications up front:

- A **radar** is a device that sends out a radio wave and listens for the echo. It's the same
  idea as the speed-camera radar that clocks passing cars, or the weather radar that tracks rain.
- This project has **no real radar hardware.** It *simulates* the radio signals with math (in
  Python), because building the signal-processing and machine-learning skills doesn't require a
  physical device — and because it's a stepping-stone toward a bigger simulation project.

So the whole thing runs on a laptop, start to finish, and anyone can reproduce it exactly.

---

## 2. The big idea, explained simply

### 2.1 How radar can sense breathing
When you breathe, your chest moves in and out by a few millimetres. If a radar points at your
chest, the radio wave has to travel a tiny bit farther when your chest is out and a tiny bit
nearer when it's in. That change of distance shifts the **phase** of the returning wave.

- **Phase** = *where you are in a wave's cycle*, like the position of a second-hand on a clock.
  Two identical waves that are slightly out of step are "out of phase." When the chest moves, the
  echo's phase shifts back and forth in time with the breathing.

So: **measure the phase of the echo over time, and you've measured the chest movement — without
touching the person, and even in the dark or through a blanket.** That's the appeal of radar for
health monitoring.

### 2.2 The core formula (don't worry, it's friendly)
When the radar's echo is processed, it comes out as **two numbers per instant**, called **I** and
**Q** (for *in-phase* and *quadrature* — just the traditional names for the two parts that
together pin down the phase). They follow:

```
I = cos(4π · d / λ)
Q = sin(4π · d / λ)
```

- `d` is the chest displacement (how far the chest has moved).
- `λ` (the Greek letter "lambda") is the **wavelength** — the length of one full radio wave.
  Shorter wavelength = more sensitive to small movements.
- The `cos` and `sin` are just the two coordinates of a point going around a circle. As the chest
  moves, the point `(I, Q)` **rotates**. The **angle** of that point *is* the phase.

The radar here uses a **24 GHz** wave (24 billion cycles per second), which has a wavelength of
about **12.5 millimetres**. That's small enough that even a few-millimetre breath makes the
`(I, Q)` point swing around by **more than a full circle** — a fact that matters in the next step.

### 2.3 Getting the breathing back out ("demodulation")
**Demodulation** just means *undoing* the encoding to recover the original movement. Three steps:

1. **Find the angle** of the `(I, Q)` point at each instant (basic trigonometry).
2. **Unwrap** it. Here's the catch: an angle naturally resets every full circle (after 360° it
   reads 0° again). Since a breath swings *more* than a full circle, the raw angle "wraps around."
   **Unwrapping** stitches those resets back together into one smooth, continuous line. Without
   this, the recovered movement would be wrong.
3. **Scale** the smooth angle back into millimetres.

The result is the chest movement over time — from which the breathing rate falls out easily.
Checking that this round trip (movement → I/Q → movement) recovers the original almost perfectly
is the project's **most important test**, because everything else is built on it.

### 2.4 A second radar type (FMCW), simplified
There are two common radar styles:

- **CW (continuous-wave):** sends a steady tone. Great for detecting tiny movements (like
  breathing) via phase — that's the one described above.
- **FMCW (frequency-modulated continuous-wave):** sweeps its tone up and down (a "chirp"). By
  comparing sent vs received, it can measure **distance** (range) to a target.

Building a full FMCW radar is complex, so this project models just the useful part: the target's
**distance wobbling** with each breath, `distance = baseline + chest movement`. It then shows this
gives the **same breathing rate** as the CW method — a reassuring cross-check that both approaches
agree.

---

## 3. A guided tour of how it works

```
make a breathing movement d(t) → turn it into radar I/Q → add realistic noise
   → demodulate back to movement → find the repeating rate → BREATHING RATE
   → measure signal shape features → machine-learning model → WHICH BREATHING DISORDER
   → detect a sudden big movement → FALL ALERT
```

Each stage in plain terms (file in brackets):

### 3.1 Make realistic breathing movements  ·  `breathing_models.py`
The software generates the chest movement for **five kinds of breathing**:

- **Normal** — steady, 12–20 breaths per minute.
- **Tachypnea** — abnormally *fast* breathing.
- **Bradypnea** — abnormally *slow* breathing.
- **Cheyne-Stokes** — a pattern seen in **heart failure**: breathing gradually gets deeper, then
  shallower, then **pauses**, over and over (a "waxing and waning" cycle).
- **Obstructive apnea** — a **sleep-apnea** pattern: normal breathing that is repeatedly
  **interrupted by pauses** where the chest barely moves.

Knowing what these look like medically is the domain knowledge that makes this a health project.

### 3.2 Turn movement into a radar signal, then dirty it up  ·  `cw_radar.py`, `noise.py`
`cw_radar.py` applies the `I = cos(...)`, `Q = sin(...)` formula from §2.2. Then `noise.py` adds
**realistic imperfections**, because a perfect signal would be cheating:

- **Thermal noise** — random fuzz, like the hiss on an old radio. Its strength is set by a chosen
  **SNR** (signal-to-noise ratio: how much stronger the real signal is than the fuzz), measured in
  **decibels (dB)** — a standard ratio scale where bigger = cleaner.
- **Phase jitter** — a slow, random wander in the timing (real electronics aren't perfectly
  steady).

### 3.3 Recover the movement and find the rate  ·  `demodulation.py`, `rate_extraction.py`
`demodulation.py` does the angle → unwrap → scale recovery from §2.3. Then `rate_extraction.py`
finds the breathing rate with an **FFT**:

- **FFT** = *Fast Fourier Transform*, a math tool that takes a wiggly line and reports **which
  repeating rhythms it contains.** The strongest rhythm in the breathing range (0.1–0.5 cycles per
  second = 6–30 breaths per minute) is the breathing rate.

### 3.4 Decide which breathing disorder it is  ·  `classifier.py`
This is the machine-learning part. Instead of feeding raw numbers to a black box, the software
first measures **six meaningful descriptors** ("**features**") of the signal, for example:

- the **breathing rate** (separates fast vs slow vs normal),
- how much the breathing depth **varies** (large for Cheyne-Stokes' waxing/waning),
- what **fraction of the time the chest is nearly still** (large for apnea's pauses).

It then feeds these to a **Random Forest** — a popular, reliable machine-learning method that is
essentially a committee of simple yes/no decision trees that vote on the answer. The software
simulates hundreds of noisy examples to train it. Choosing features that *mean something*
(rather than raw data) is a deliberate design choice that makes the model both accurate and
explainable.

### 3.5 Detect a fall  ·  `fall_detection.py`
A fall is a **big, fast, brief** movement — completely unlike slow, gentle breathing. The
software watches how *fast* the radar phase is changing and raises an alert when that speed
briefly spikes far above the breathing level. In testing, a fall produces an energy spike roughly
**100,000×** larger than breathing — trivially easy to tell apart.

---

## 4. The key ideas, defined simply

### 4.1 Phase, and why higher frequency means more sensitivity
**Phase** is position within a wave's cycle. The formula's `4π/λ` says: the shorter the wavelength
`λ`, the more the phase changes per millimetre of movement — so a higher-frequency radar (like 24
GHz) is **more sensitive** to tiny motions. This same phase idea underlies GPS, Wi-Fi, and lidar.

### 4.2 Phase wrapping (and why the sampling rate must be high enough)
An angle resets every 360°, so a large movement makes the measured angle "wrap around." Unwrapping
fixes this — **but only if the angle doesn't change by more than half a turn between two samples.**
That sets a **minimum sampling rate** (how many measurements per second). Because a *fall* is fast,
the simulation samples at **1000 times per second** to be safe. (The rule that you must sample fast
enough to capture a rhythm is called the **Nyquist** limit.)

### 4.3 I/Q signals
Describing a wave by its two parts, **I** and **Q**, lets you recover both *how much* and *in which
direction* the phase moved. This "two-number" representation is the foundation of essentially all
modern radio, radar, and wireless systems.

### 4.4 FFT and cleaning up the spectrum
The FFT finds the rhythms in a signal. To make its answer precise, the software first **detrends**
(removes slow drift), applies a **window** (a gentle fade at the edges that prevents the FFT from
smearing energy — the smearing is called **spectral leakage**), then **zero-pads and curve-fits**
the peak to pinpoint the rate finely. The breathing search is deliberately limited to the human
breathing range, which encodes the physics into the math.

### 4.5 Signal-to-noise ratio in decibels
**SNR** compares wanted signal to unwanted noise. In **decibels**, 20 dB is a clean signal, 0 dB
means the noise is as strong as the signal (very harsh). The project deliberately tests at 20, 10,
and 0 dB to see **where the method breaks**, rather than only showing the easy case.

### 4.6 Features, Random Forest, and the confusion matrix
- **Feature** — a single meaningful number describing the signal (rate, variability, etc.).
- **Random Forest** — a classifier (a labeller) made of many small decision trees that vote.
- **Confusion matrix** — a results table showing, for each true class, what the model *guessed*.
  Reading it reveals *which* disorders get mixed up and *why*. Here it showed that when the noise
  is worst (0 dB), the model most often confuses **obstructive apnea** with **normal** breathing —
  which makes sense, because heavy noise fills in the apnea pauses so they no longer look like
  pauses.

---

## 5. How the project was built

### 5.1 Test-first development (TDD)
**TDD** = *Test-Driven Development*: write an automatic check before the code, then code until it
passes. The pivotal check is the **round-trip physics test**: make a known chest movement, turn it
into a radar signal, demodulate it back, and confirm you recover the original to within a hair
(under 0.05 mm). Only after that passed was anything else built on top. There are 27 automatic
checks, all passing.

### 5.2 Built in eight phases, saved step by step
The work went: breathing models → radar + demodulation (the physics gate) → rate extraction →
FMCW cross-check → noise → classifier → fall detection → validation + documentation. Each phase
ended with passing tests and a saved snapshot (a **git commit**). The history reads like the story
of the build.

### 5.3 Deterministic and self-contained
Because it's a simulation with fixed random seeds, **every result is exactly reproducible** — no
hardware, no external data, no flaky downloads. Anyone can clone it and get the same numbers.

### 5.4 A real constraint that was handled
The project brief suggested a radar library called `radarsimpy`, but it **can't be installed the
normal way** (it isn't on the standard Python package index). The brief anticipated this and named
a fallback: **write the radar physics directly in NumPy** (the standard Python math library), which
is scientifically equivalent for this scope. That's exactly what was done, and the unavailable
library was kept out of the install list so a fresh copy still installs cleanly.

### 5.5 Honest reporting
The classifier is **100% accurate at good signal levels but only ~39% at the harshest noise**,
which drags the pooled average to 76%. The project reports the **breakdown by noise level** and
names the class that fails, instead of quoting only the flattering number — the kind of honesty
that makes results trustworthy.

---

## 6. How well it works (and how to read the results)

**Breathing-rate error** (recovered rate vs the known true rate):

| Signal cleanliness | Average error |
|---|--:|
| Clean (20 dB) | 0.0 breaths/min |
| Moderate (10 dB) | 0.0 breaths/min |
| Very noisy (0 dB) | ~5 breaths/min |

**Telling the five breathing types apart:**

| Signal cleanliness | Accuracy |
|---|--:|
| Clean (20 dB) | 100% |
| Moderate (10 dB) | 100% |
| Very noisy (0 dB) | ~39% |

**Fall detection:** the fall's energy spike is about 100,000× the breathing level — easily
separated.

The takeaway: **at realistic signal levels the method is essentially perfect, and it degrades
gradually and predictably as noise rises**, with a failure mode (apnea looking like normal under
extreme noise) that has a clear physical explanation. That's a complete, honest result.

---

## 7. Why certain choices were made

| Choice | Reason in one line |
|---|---|
| Write the radar physics in NumPy | the suggested library couldn't be installed; the math is equivalent |
| Unwrap the phase | a breath swings more than a full circle, so the raw angle must be stitched together |
| Sample 1000×/second | needed so a fast fall isn't missed and the unwrapping stays valid |
| Simplify FMCW to a distance wobble | captures breathing detection without heavy complexity; cross-checks CW |
| Meaningful features + Random Forest | the features map to real physiology; the model stays accurate and explainable |
| Test at 20, 10, and 0 dB noise | show where and how the method breaks, not just the easy case |
| Round-trip test as a gate | prove the physics before building anything on it |

---

## 8. Glossary (plain definitions)

- **Radar** — a device that sends a radio wave and measures its echo.
- **CW / FMCW radar** — steady-tone radar (good for tiny motion) / frequency-sweeping radar (good
  for distance).
- **Phase** — position within a wave's cycle; it shifts when the target moves.
- **I/Q** — the two numbers that together describe a wave's phase.
- **Wavelength (λ)** — the length of one wave; shorter = more sensitive to small motions.
- **Demodulation** — recovering the original movement from the radar signal.
- **Phase unwrapping** — stitching a repeatedly-resetting angle into one smooth line.
- **FFT** — a tool that finds the repeating rhythms in a signal.
- **Sampling rate / Nyquist** — measurements per second / the fastest rhythm you can capture.
- **SNR / decibel (dB)** — how much stronger the signal is than the noise / the ratio scale for it.
- **Feature** — a meaningful number summarising the signal.
- **Random Forest** — a classifier made of many voting decision trees.
- **Confusion matrix** — a table of true-vs-guessed classes.
- **Cheyne-Stokes / obstructive apnea** — a heart-failure breathing pattern / a sleep-apnea
  pattern with pauses.
- **git commit** — a saved, labelled snapshot of the code.

---

## 9. How to run it

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest                                   # run the automatic checks (incl. the round-trip physics test)
python run_demo.py                       # prints all 5 rates + the disorder + a fall; writes docs/demo.png
python validation/run_validation.py      # reproduces the accuracy numbers and the confusion matrix
```

To read the code, start with `src/cw_radar.py` and `src/demodulation.py` (the physics), then
`src/breathing_models.py`, `src/rate_extraction.py`, `src/classifier.py`, and
`src/fall_detection.py`. The physics is also written up in `docs/physics_notes.md`. Every idea in
this document lives in that code, with comments explaining each step.
