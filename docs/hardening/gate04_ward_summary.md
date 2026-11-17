# Gate 4 (+5) — Ward-identity metric crosswalk and normalized metric

**Verdict: PASS WITH REFORMULATION.**

The architecture discrimination the Ward identities are asked to support does
hold, and holds more cleanly than the manuscript currently states it. But three
labels attached to the quoted numbers are wrong, and one estimator class is
being cited for something it cannot measure. None of these is repaired by
changing a number: each needs a wording change that the computation below
specifies exactly.

Runner `scripts/run_gate04_ward.py` (post-processing only — no estimator was
re-run; every quantity is reconstructed exactly from the frozen per-point rows).
Figure `gate04_ward_absolute_vs_normalized.pdf`.

## 1. The metric, pre-specified and versioned

`metric_version = ward_norm_v1`, carried on every row:

    E_norm = sqrt( Σ_g w_g |T_est(z_g) − T_tgt(z_g)|² / Σ_g w_g |T_tgt(z_g)|² )

over the probe set of a configuration, with **uniform** geometry weight
`w_g = 1`. Uniform is a recorded choice, not a default: no principled
pre-specified weighting existed, so inventing one at analysis time would be a
post-hoc window (mission rule 12). The runner refuses to run if the config
asks for any other weighting.

Pooled in L2 rather than pointwise because several probes sit near a zero of
`T_tgt`, where a pointwise ratio is unstable. `|T_tgt|` is reconstructed as
`stress_absolute_error / stress_relative_error`, which is exact — the stored
`stress_estimate` and `stress_target` are magnitudes of complex quantities, so
their difference does *not* reproduce the stored absolute error.

## 2. Defect 1 — the quoted numbers are absolute, not normalized

All four values reproduce exactly as the **maximum absolute** stress-Ward error
in `deterministic_feature_ward_points.csv`, tapered regulator:

| quoted | architecture | N | reproduces as max-absolute | `E_norm` at the same setting | ratio |
|---|---|---|---|---|---|
| 0.129304 | selected | 200000 | ✔ exact | **0.919** | 7.1× |
| 12.191757 | angular scramble | 200000 | ✔ exact | 82.06 | 6.7× |
| 2.152960 | RBF Gaussian | 200000 | ✔ exact | 31.15 | 14.5× |
| 7.574309 | selected | 500 | ✔ exact | 82.85 | 10.9× |

This is not a quibble about a factor. Calling `0.129` *normalized* asserts the
stress Ward identity is satisfied to ~13% of the target. The normalized error at
that setting is **0.92** — the RMS violation is about the same size as the RMS
target. Every normalization a reader might reasonably have meant lands in the
same place, so the discrepancy is not an artifact of the L2 choice:

| normalization (selected, N=200000, tapered) | value |
|---|---|
| max absolute *(the quoted number)* | 0.129 |
| **`E_norm`, L2 pooled (principal)** | **0.919** |
| mean pointwise relative | 0.800 |
| median pointwise relative | 0.682 |
| max pointwise relative | 3.330 |

**Smallest remediation.** Either relabel the four numbers "maximum absolute
stress-Ward error", or quote the normalized values (0.92 / 82.1 / 31.1 / 82.9).
Both are defensible; mixing them is not. `ward_metric_crosswalk.csv` carries
both columns for every quoted value so the choice can be made once and applied
consistently.

## 3. Defect 2 — "largest N and M" is undefined for this estimator class

The four values come from the `deterministic_finite_feature` campaign, whose
file **has no `n_networks` column at all**. `M` is a property of the
`sampled_finite_network` class. Attributing a deterministic number to "the
largest N and M" merges two estimator classes, which mission rule 7 forbids.

**Smallest remediation.** "at the largest width, N = 2×10⁵" — and delete the
reference to M from that sentence.

## 4. Defect 3 — two campaigns share one column name (rule 11)

`ward_stress_max_absolute_error` is published by two different campaigns whose
values are **not comparable**:

| campaign | file | geometry | probes | N range | selected value at max N |
|---|---|---|---|---|---|
| WP-A controls grid | `architecture_controls_seed_rows.csv` | 2 vertices at ±0.9, charges ±0.3 | 3 | 10⁴ – 10⁶ | 0.0148 |
| WP-C Ward campaign | `deterministic_feature_ward_points.csv` | 3 configurations; two-vertex at ±0.85 | 4 | 500 – 2×10⁵ | 0.129 |

Different probe sets, different vertex positions, different widths. The frozen
ledger's `wpA_control_*` rows (selected 0.0128 tapered at N = 10⁶) and the
quoted 0.129 are **different measurements**, and the smaller number is not a
better version of the larger one. Both rows are in
`ward_metric_crosswalk.csv` under `kind = shared_label_collision`.

## 5. Defect 4 — the sampled Ward rows cannot carry architecture evidence

The sampled campaign's normalized error scales as a pure Monte Carlo variance:

    E_norm ∝ M^b,   b = −0.482 ± 0.060 over 18 cells,   16/18 within 0.12 of −1/2

with per-cell R² of 0.94 – 0.9999, and it is **flat in N** at fixed M
(four-vertex: 1349, 1127, 1297 at N = 500, 1000, 2000). There is no
architecture-dependent floor visible at these M: the campaign is measuring its
own sampling noise.

The consequence is stark. At the observed rate, reaching `E_norm < 1` — the
precision the *deterministic* campaign already has — would need

    M ≈ 5.8×10⁸ to 5.1×10¹⁰   (against the 3.2×10⁴ actually run)

i.e. four to six orders of magnitude more sampling. This is recorded in
`sampled_ward_monte_carlo_scaling.csv`.

This also explains, and disposes of, an apparent contradiction. In the sampled
class the paired ratio against the selected architecture *reverses*: RBF scores
0.042 (≈24× "better" than selected, better in 100% of pairs) and the angular
scramble ties at 1.05. That is not evidence about chiral structure. `T_tgt` is
analytic and identical across architectures, so the ratio is a ratio of RMS
errors, and at these M those errors are sampling noise whose amplitude tracks
each architecture's field variance. The selected log kernel has the largest
`σ_φ² = 4L_w`, hence the largest noise.

**Smallest remediation.** Present the sampled Ward rows as a finite-M variance
diagnostic. Do not cite them for or against any architecture. Preserve them —
they are an informative negative control, and mission rule 8 requires keeping
them.

## 6. What the Ward identities *do* establish

Restricted to the estimator class that can measure it, the result is stronger
than the quoted quartet suggests, because the evidence is a **convergence rate**
across four widths rather than one number:

| architecture | `E_norm` vs N exponent | N=500 | N=2000 | N=2×10⁴ | N=2×10⁵ | converges? |
|---|---|---|---|---|---|---|
| selected log kernel | **−0.783** | 79.98 | 22.16 | 3.01 | **0.76** | yes |
| local exponent direction | **−0.911** | 457.14 | 124.78 | 12.28 | 2.06 | yes |
| RBF Gaussian prior | +0.004 | 24.47 | 25.25 | 25.31 | 25.14 | **no** |
| angular mode scramble | −0.085 | 116.09 | 62.39 | 51.38 | 64.82 | **no** |

Paired against the selected architecture at identical configuration, width and
seed, at the quoted width N = 2×10⁵, tapered regulator.

**The uncertainty is aggregated at the independent replicate level.** The raw
pairs behind a cell are configurations × seeds; configurations are repeated
measurements on the *same* draw, so only the **seed** is an independent
replicate. Treating "worse in 24/24 pairs" as 24 Bernoulli trials would
overstate the effective sample size roughly threefold. The log-ratio is
therefore averaged within each seed first, and the spread taken across seeds
(`ward_paired_block_level.csv`):

| control | geometric-mean ratio | 95% CI | independent blocks | raw pairs | p |
|---|---|---|---|---|---|
| RBF Gaussian prior | **31.74** | [28.09, 35.87] | 8 | 24 | 1.6×10⁻¹⁰ |
| angular mode scramble | 62.75 | [44.76, 87.98] | 8 | 24 | 5.5×10⁻⁸ |
| local exponent direction | 2.41 | [1.89, 3.07] | 8 | 24 | 2.0×10⁻⁴ |

All three are worse than the selected architecture in **8 of 8 independent
blocks**. The conclusion survives the correct weighting; it is the *significance*
that would have been inflated by counting pairs. The per-pair figures remain in
`ward_paired_ratio.csv` as a descriptive breakdown, not as a test.

Discrimination **emerges with width** and is absent at the smallest: at N=500
RBF scores 0.29 (better than selected) and the scramble 1.63 (worse in only 52%
of pairs). Quoting a single width therefore understates the evidence; the
convergence exponent is the claim that the data actually supports.

Leave-one-block-out over all 1164 drops moves no cell by more than **0.98 block
standard errors**, so no single seed drives any of this
(`sampled_ward_leave_one_block_out.csv`).

## 7. The control the quoted set omits

The quoted quartet reports the selected architecture and the *two worst*
controls, plus the selected architecture at its worst width. It omits
**`local_exponent_direction`**, which is the nearest control on this axis —
2.65× at N = 2×10⁵, and converging with an exponent (−0.911) at least as steep
as the selected architecture's (−0.783). The Ward identities separate the
selected architecture from RBF and from the angular scramble decisively; they do
**not** separate it from `local_exponent_direction` qualitatively, only by a
factor of about 2.6 in amplitude.

This is consistent with Gate 1, where `local_exponent_direction` was also the
control that intrinsic admissibility failed to exclude cleanly. Excluding it is
not a Ward-identity result and must not be claimed as one.

**Smallest remediation.** Quote all four architectures, or state explicitly that
the two reported controls are the two that fail most strongly and name the
nearest surviving control.

## 8. Ledger status

| claim | status |
|---|---|
| Ward numbers as **absolute** max errors, largest width, deterministic class | `accepted_with_reformulation` (relabel; drop "M") |
| Ward-based exclusion of RBF and angular scramble | `accepted` (paired, 48/48 and 47/48 at N=2×10⁵; non-convergent in N) |
| Ward-based exclusion of `local_exponent_direction` | `fails_current_claim` (converges at −0.911; only 2.65× behind) |
| Sampled Ward rows as architecture evidence | `diagnostic_only` (pure M^(−1/2) noise; needs 10⁴–10⁶× more M) |
| Cross-campaign comparison of `ward_stress_max_absolute_error` | `needs_new_computation` if a common-geometry comparison is wanted; otherwise crosswalk only |
