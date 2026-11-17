# Gate 3 (+5) — TT estimator provenance, direct/Wick separation, and the c = 1 claim

**Verdict: PASS WITH REFORMULATION.**

The claim `c = 1` survives, and there is now an assumption-free measurement
supporting it that did not exist before. But the *frozen headline number does
not support it for the reason the manuscript gives*, and the reformulation is
not cosmetic: at 133× the statistics the same estimator moves 23σ away from 1.
The resolution is a fit-weighting definition mismatch plus a finite-regulator
offset, both of which are computable, and once both are accounted for
everything agrees.

Runners `scripts/run_gate03_tt.py` and `scripts/run_gate03_direct_extension.py`.
Figure `gate03_tt_estimator_comparison.pdf`.

## 1. Provenance of the headline — answered

`c_sampled = 1.016 ± 0.011` resolves **uniquely** in the frozen aggregate
(`tt_estimator_provenance.csv`):

| property | value |
|---|---|
| estimator variant | **`wick_factorized` — Wick only.** Not direct, not combined |
| regulator class | `tapered_log_window` (taper 0.15) |
| width | **N = 250 — the smaller of the two audited widths {250, 750}** |
| blocks / realizations | 12 named seeds × M = 60000 |
| geometry | 48 points on \|z\|=0.6; chord separations, fit window [0.30, 0.55] |
| normal ordering | `c_N = (L_w/2N) Σ_j conj(k_j)²`, conditional phase-averaged |

The Wick estimator forms `c = (mean_blocks S)²` from the two-point function. It
therefore **presupposes the Gaussian factorization** — the property an
independent `⟨TT⟩` determination would be testing. The frozen ledger discloses
this honestly in its notes; the manuscript's CLAIM-GATE FM-02/FM-07 wording
("the independent sampled ⟨TT⟩ determination") does not.

## 2. The direct/Wick cumulant identity — verified

`G_TT^dir = G_TT^Wick + ¼κ_JJJJ(z,z,w,w)`, computed by different expressions
from the same draws (the left uses the per-realization subtraction `c_N`, the
right uses ensemble second moments), at a benign cutoff where the direct
estimator is well conditioned (Λ ∈ [0.3, 3], taper 0, N = 200):

| M | relative residual |
|---|---|
| 50000 | 4.09e-03, 3.53e-04 |
| 200000 | 5.72e-04, 9.70e-04 |

The residual falls **2.88× for a 4× increase in M** (M^(−1/2) predicts 2.00×) —
Monte Carlo convergence, so the identity is exact and the naive form is correct.
A candidate `−¼Var(c_N)` correction was derived and tested; it made agreement
slightly *worse*, so `Var(c_N)` is negligible here and the correction is not
part of the identity.

**Why it is not checked at the production cutoff.** There, `G_dir ≈ 1800` where
the true `⟨TT⟩ ≈ 0.24`: the direct four-field estimator has per-sample variance
`~Var(∂φ)² ~ 10⁶`, so every term is noise-dominated and the identity is
unverifiable, not violated. That variance limitation is a Gate-3 finding in its
own right and is recorded in the config.

## 3. The direct campaign was extended — the mission's condition was met

The mission permits extending direct Monte Carlo only if the existing direct
result cannot support the manuscript wording. It could not: the frozen direct
value at the headline setting is `2.06 ± 0.58`, which neither establishes `c=1`
nor excludes it. Extended to **M = 7,980,993 per block × 12 blocks
(9.6×10⁷ realizations, 97.9 min on 12 cores)**, geometry, regulator and fit
window unchanged, on an RNG stream distinct from the frozen run so the result is
an independent draw:

| estimator | frozen (M=6×10⁴) | extended (M=8×10⁶) | SE shrink (expected 11.5×) |
|---|---|---|---|
| **direct** (no analytic input) | 2.057 ± 0.577 | **1.0508 ± 0.0386** | 14.9× |
| **Wick** (assumes factorization) | 1.016 ± 0.011 | **1.0341 ± 0.0014** | 7.8× |

The two agree with each other: `direct − Wick = +0.0167 ± 0.0386` (0.43σ).

**This is the new result.** There is now an assumption-free determination of the
central charge to 3.7%, consistent with `c = 1` at **1.3σ**. That sentence could
not have been written from the frozen release.

## 4. The contradiction, stated precisely

At M = 8×10⁶ the **Wick** estimator gives `1.0341 ± 0.0014`, which is
**23.6σ from c = 1**. The frozen headline's apparent agreement with 1 was a
consequence of its error bar, not evidence about the central charge.

This does **not** mean the measured central charge differs from 1. Two effects,
both computable, account for the offset exactly.

### 4a. A shared label with two different fit weightings (mission rule 11)

The column `c_exp` is written by two kinds of row that summarize the **same
c(r) curve** with **different weights**:

- `analytic_closed_kernel_reference` rows carry zero per-separation stderrs, so
  `fit_central_charge` falls back to **uniform** weights;
- every sampled row (`wick_factorized`, `direct_TT`) uses **inverse-variance**
  weights.

Inside the fit window the analytic curve is far from flat, and the sampled
weights are far from uniform:

| separation r | analytic c(r) | sampled weight |
|---|---|---|
| 0.3106 | 1.0577 | **0.569** |
| 0.3857 | 1.0382 | 0.233 |
| 0.4592 | 0.9526 | 0.134 |
| 0.5307 | 1.0043 | 0.065 |

One curve, two summaries:

    analytic c_exp, uniform weights          = 1.013202   <- the frozen reference row
    analytic c_exp, inverse-variance weights = 1.035886   <- what the sampled rows target
    weighting shift                          = +0.022685

### 4b. Everything agrees once the weighting is matched

| quantity | value | vs matched-weight target 1.0359 |
|---|---|---|
| Wick, M = 8×10⁶ | 1.0341 ± 0.0014 | **−1.23σ** |
| direct, M = 8×10⁶ | 1.0508 ± 0.0386 | **+0.39σ** |

So the sampled estimators reproduce their own analytic regulator target to
about one standard error. There is **no finite-N anomaly** and no unexplained
23σ effect — the 23σ was against the wrong reference value.

### 4c. The target tends to 1 when the regulator is removed

Purely analytic, at the campaign geometry and fit window
(`gate03_regulator_convergence.csv`):

| window | c_exp (uniform) | max\|c(r) − 1\| |
|---|---|---|
| [10⁻², 10²] *(production)* | 1.01320 | 5.77e-02 |
| [10⁻³, 10³] | 1.00043 | 2.36e-03 |
| [10⁻⁴, 10⁴] | 1.00001 | 8.7e-05 |
| [10⁻⁶, 10⁶] | 1.00000 | 6.3e-07 |

Convergence is uniform in r, so it is weighting-independent: one extra decade of
cutoff already brings the target to 1.0004. The 3.6% offset at the production
regulator is a **finite-cutoff artifact of a deliberately narrow window**, not a
statement about the theory.

## 4d. The estimator-independent ratio — the decisive comparison

The comparisons above still route through a window summary. The clean test
removes the fit entirely by forming the ratio **separation by separation**:

    q_g = c_direct,g / c_analytic,g   at each geometry g

At a given separation the sampled estimator and the analytic regulated target
are two numbers for the same quantity, so `q_g = 1` whatever weighting anyone
would have chosen. Uncertainties are taken at the level of the independent
replicate — the block — and the four separations inside a block share
realizations, so the test is covariance-aware: Hotelling's `T²` on the
block-level 4-vector plus a GLS fit of the single common ratio
(`gate03_q_ratio_test.csv`, 12 blocks × 4 separations).

| estimator | GLS common ratio | p vs 1 | Hotelling `T²` p |
|---|---|---|---|
| **direct** | **1.00830 ± 0.03377** | 0.810 | 0.246 |
| **Wick** | **0.99941 ± 0.00133** | 0.666 | 0.319 |

Per separation, the Wick estimator reproduces the analytic regulated target to
under 1% everywhere: `q = 1.0014, 0.9975, 0.9923, 0.9941`.

**This closes the question.** With the fit weighting removed, both sampled
estimators agree with the analytic regulated target. The 23σ gap against `c = 1`
in §4 was the weighting mismatch plus the finite regulator — there is no
estimator defect and no finite-N anomaly.

One wrinkle, reported rather than smoothed: the direct estimator's common-ratio
fit has a marginal goodness of fit (`χ² = 9.19` on 3 dof, `p = 0.027`). It is
driven entirely by the largest separation, `r = 0.531`, where `q = 1.92 ± 0.56`
with a per-block spread of 1.95 (range −1.90 to 4.39) — the direct estimator is
variance-limited there. The other three separations give `0.992, 0.979, 1.003`.
The joint Hotelling test, which accounts for this, does not reject (`p = 0.246`).

## 5. What the evidence for c = 1 actually is

The manuscript currently rests the claim on a single number agreeing with 1.
After this gate, the supportable chain is stronger and has three independent
links:

1. the analytic regulated target → 1 uniformly as the cutoffs are removed (§4c);
2. the sampled estimators reproduce that target **separation by separation and
   with no fit weighting at all** — `q = 1.008 ± 0.034` (direct), `0.99941 ±
   0.00133` (Wick) (§4d);
3. the **direct** estimator, which assumes no factorization, is independently
   consistent with `c = 1` at 1.3σ (§3).

Link 3 is the one that answers the "independent determination" wording; links 1
and 2 are what make the production-regulator number interpretable.

## 6. Regulator dependence is a discovered control, not a blemish

At the sharp annulus the Wick estimator at N = 750 gives `1.381 ± 0.116`,
**3.28σ above 1** and incompatible with it. That row carries the status
`failed_gate` in the frozen ledger — a label the hardening schema does not
allow; it must be remapped to **`fails_current_claim`**. The tapered window is
what makes the estimator well behaved, and the sharp-cutoff failure should be
presented as the regulator theorem it is.

## 7. Ledger status

| claim | status |
|---|---|
| `c = 1.016 ± 0.011` as an *independent* determination | `fails_current_claim` — Wick only, presupposes factorization; agreement with 1 is a low-statistics artifact |
| `c = 1.016 ± 0.011` as a *Wick-factorized* determination at N=250, M=6×10⁴ | `accepted_with_reformulation` — relabel; its target is 1.0359, not 1 |
| direct finite-regulator TT normalization `1.051 ± 0.039`, M = 8×10⁶ | `accepted` — new, assumption-free, 1.3σ from c=1. **Not** to be published alone as "the central charge" |
| estimator-independent ratio `q_g = 1` | `accepted` — GLS 1.008 ± 0.034 (direct), 0.99941 ± 0.00133 (Wick) |
| Wick determination `c = 1.0341 ± 0.0014`, M = 8×10⁶ | `accepted` — 1.2σ from its matched-weight analytic target |
| `c_exp` compared across analytic and sampled rows | `accepted_with_reformulation` — requires the §4a weighting crosswalk |
| analytic target → 1 as Λ widens | `accepted` |
| sharp-annulus Wick at N=750 | `fails_current_claim` (relabel from the disallowed `failed_gate`) |

## 8. Smallest remediations

1. Say **"Wick-factorized"** wherever `1.016` appears, and drop "independent".
2. Quote the **direct** extended value `1.051 ± 0.039` as the *direct
   finite-regulator TT normalization* — not as "the central charge" — and
   compare it to the matched analytic regulated target, never to 1.
3. When comparing any sampled `c_exp` to an analytic reference, use the
   **matched-weight** analytic value `1.0359`, never the uniform-weight
   `1.0132`. Either that, or refit both with uniform weights — but versioned,
   applied to every affected row, and documented (mission rule 10).
4. State that the production regulator carries a computed +3.6% offset that
   vanishes as the cutoffs widen; cite `gate03_regulator_convergence.csv`.
5. Remap the sharp-annulus N=750 row from `failed_gate` to `fails_current_claim`.
