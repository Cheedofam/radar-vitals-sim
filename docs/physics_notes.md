# Physics notes

Background for the radar models and the parameter choices in this simulation. Values
follow the ranges in the project brief, consistent with the microwave Doppler-radar
vital-signs literature (e.g. Droitcour, *Non-contact measurement of heart and respiration
rates with a single-chip microwave Doppler radar*, Stanford, 2006; Li & Lin, *Recent
advances in Doppler radar sensors for pervasive healthcare*, IEEE MTT, 2010).

## CW Doppler radar

A CW radar transmits a single-frequency wave. A target moving by `d(t)` changes the
round-trip path by `2·d(t)`, which phase-modulates the reflected signal. After quadrature
downconversion the complex baseband signal is:

```
I(t) = A·cos(4π·d(t)/λ + φ0)
Q(t) = A·sin(4π·d(t)/λ + φ0)
```

The `4π/λ` factor is the round-trip (`2·d`) times the wavenumber (`2π/λ`). At a **24 GHz**
carrier, `λ = c/f = 3e8 / 24e9 ≈ 12.5 mm`, so `4π/λ ≈ 1005 rad/m`. A chest-wall excursion
of a few millimetres therefore produces **several radians** of phase — more than one full
turn — which is why demodulation must unwrap.

**Demodulation.** The instantaneous phase is recovered as `atan2(Q, I)`, wrapped to
`[-π, π]`. Because the true phase exceeds `2π`, it is unwrapped into a continuous curve and
scaled back to displacement:

```
d(t) = unwrap(atan2(Q, I)) · λ / (4π)
```

Breathing rate is then the dominant spectral peak of `d(t)` in 0.1–0.5 Hz. The round-trip
`d → I/Q → d` is verified in `tests/test_demodulation.py` to < 0.05 mm RMS.

## FMCW radar (simplified)

A full FMCW radar transmits a frequency-swept chirp; the beat frequency between transmit
and receive is proportional to target range, `f_beat = 2·R·S/c` for sweep slope `S`. The
chest sits in one range bin, and that bin's position oscillates with breathing.

For this proof-of-concept the informative quantity is the *range oscillation*, so the model
represents range directly as `R(t) = R0 + d(t)` rather than synthesizing and mixing chirps.
This captures the same breathing-detection principle at far lower complexity, and
`tests/test_rate_extraction.py` confirms it recovers the same rate as the CW path.

## Breathing waveform models

Chest-wall antero-posterior displacement `d(t)` in millimetres (see `src/breathing_models.py`):

| Pattern | Rate | Amplitude | Shape |
|---|---|---|---|
| Normal | 12–20 br/min | 4–12 mm | quasi-sinusoidal |
| Tachypnea | > 20 br/min | shallower (~3 mm) | same shape, faster |
| Bradypnea | < 12 br/min | ~8 mm | same shape, slower |
| Cheyne-Stokes | ~15 br/min carrier | waxing/waning | crescendo-decrescendo envelope, ~45–90 s cycle, with an apneic pause |
| Obstructive apnea | ~15 br/min carrier | normal then ~0 | normal breathing interrupted by 10–20 s of near-absent chest motion |

A small second harmonic is added to each carrier so the waveform is not a pure sinusoid
(real breathing has a faster inspiration than expiration); the fundamental still dominates.
Paradoxical chest/abdomen motion during obstructive events is a documented stretch goal and
is **not** modelled here.

## Definition of "rate" for cyclic patterns

Cheyne-Stokes and obstructive apnea have no single breathing rate — amplitude waxes, wanes,
and pauses. The rate reported by `rate_extraction` is the **intra-burst rate**: the carrier
frequency while the subject is actively breathing. The slow crescendo/apnea envelope has a
period of tens of seconds (well below the 0.1 Hz band edge) and is removed by the
band-pass, so the FFT peak is the carrier. The *pattern* itself is identified separately by
the classifier from envelope-shape features (envelope variation, apnea-gap fraction).

## Fall detection

A fall is a large, brief body movement — very different from small, slow, periodic
breathing. In the radar signal it appears as a short burst of very high Doppler (phase
rate). The detector computes the short-time energy of the demodulated phase velocity and
flags a fall when the peak exceeds a multiple of the breathing-baseline median energy. This
is deliberately simple: the aim is to demonstrate the detection principle, not to build a
sophisticated fall classifier. Note that the fast fall Doppler (~160–320 Hz at 1–2 m/s)
sets the sampling-rate floor (fs = 1000 Hz) to avoid aliasing.

## Noise model

Two effects move validation off the noiseless case (`src/noise.py`): additive white
Gaussian **thermal noise** on I and Q at a target SNR (dB, total-power definition), and a
random-walk **phase jitter** (oscillator instability) that perturbs the phase while leaving
the envelope amplitude unchanged. Validation sweeps SNR = {20, 10, 0} dB.
