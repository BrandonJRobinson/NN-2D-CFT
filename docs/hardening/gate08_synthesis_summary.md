# Gate 8 — Intrinsic symmetry audit and the architecture synthesis

**Verdict: PASS. The selected architecture is uniquely isolated by the joint
structural suite.**

Runner `scripts/run_gate08_synthesis.py`; new module
`src/nnft_virasoro/translation_invariance.py`. Outputs
`intrinsic_stationarity_seed_rows.csv` (160 rows), `convergence_joint_fit.csv`,
`architecture_synthesis.csv`.

## 1. The gap this gate closed

Before this gate, `phase_correlated_features` **survived every axis in the
campaign** — and not marginally. It has the correct `|k|^{-2}` radial spectrum
and the correct isotropic angular law; it differs from the selected architecture
only by pairing features to share a phase. On the existing axes it was as good
as, or better than, the selected architecture:

| axis | selected | phase-correlated |
|---|---|---|
| δ_iso at max width | 0.0541 | 0.0598 |
| δ_iso exponent | −0.488 | **−0.591** (steeper) |
| λ_min | 0.918 | **0.953** (larger) |
| Ward at N=10⁶ | 0.0128 | **0.00986** (smaller) |
| Ward exponent | −0.581 | −0.579 |

The reason is structural, not accidental. Every one of those diagnostics is a
functional of the kernel built from contour integrals or from the pairing
matrix, and each is **blind to loss of translation invariance**. The suite could
not have excluded this control no matter how much more width was spent on it.

## 2. The stationarity diagnostic

On the **true two-argument kernel** — this matters, because the one-argument
helpers in `kernels.py` take only `ξ = z − w` and are translation invariant *by
construction*; routing a phase-correlated architecture through them returns a
defect of exactly zero and hides the effect:

    δ_TI = [ Σ_{g,a} |K(x_g + a, y_g + a) − K(x_g, y_g)|²
             / Σ_{g,a} |K(x_g, y_g)|² ]^{1/2}

with the sum over the architecture's own `idx_a`/`idx_b` pairing, 6 separations
× 16 rigid translations, on a **deterministic** geometry (no RNG, so a defect
can never be confused with sampling scatter). A test asserts that the general
form reproduces the one-argument helpers exactly whenever `idx_a == idx_b`, and
that it does *not* for the phase-correlated control.

**Result — scalar channel, mean over 4 seeds:**

| architecture | N=10⁴ | N=3×10⁴ | N=10⁵ | N=3×10⁵ | N=10⁶ |
|---|---|---|---|---|---|
| selected_log_kernel | 8.5e-17 | 8.0e-17 | 7.6e-17 | 8.3e-17 | 9.1e-17 |
| *(all six other independent-phase controls)* | ~1e-16 | ~1e-16 | ~1e-16 | ~1e-16 | ~1e-16 |
| **phase_correlated_features** | **0.2282** | **0.2279** | **0.2266** | **0.2277** | **0.2279** |

Conditional phase averaging makes stationarity **exact up to floating point**
for every independent-phase architecture — the largest defect over all seven, at
every width, is `1.73e-16`. The phase-correlated defect is `0.228` and is **flat
across a 100× width range**: structural, not statistical. The separation is
fifteen orders of magnitude.

The current channel agrees but reads differently: `0.325 → 0.0625` from N=10⁴ to
10⁶ against `~1e-15` for everything else. The cross terms carry a
`conj(k_a)conj(k_b)` prefactor with random relative phase, so their contribution
is suppressed as the diagonal grows — the defect shrinks but stays fourteen
orders above the independent-phase floor. The **scalar** channel is the sharp
structural statement; the current channel corroborates it.

## 3. A second symmetry, added for the same reason

Two controls (`angular_mode_scramble`, `anisotropic_angular_prior`) are designed
to break **rotational** invariance, and were being judged on a Ward proxy. That
proxy was not decisive: `anisotropic_angular_prior` has a Ward exponent of
−0.270, converging, and would have passed. Measuring the symmetry directly, with
the channel's spin weight (`s = 0` scalar, `s = 2` current — the same spin-two
weight Gate 6 pins in check C7):

    δ_RI = [ Σ_{g,φ} |e^{isφ} K(e^{iφ}x_g, e^{iφ}y_g) − K(x_g,y_g)|² / Σ |K|² ]^{1/2}

Isotropy differs from stationarity in kind, and the diagnostic is read
accordingly: an isotropic *prior* is only **statistically** isotropic at finite
N, since the empirical angular distribution of N draws is anisotropic at order
`N^(−1/2)`. So a rotationally invariant architecture has a defect that
**shrinks**, not one that vanishes. The verdict is convergence, not zero:

| architecture | δ_RI at N=10⁴ | at N=10⁶ | exponent | verdict |
|---|---|---|---|---|
| selected_log_kernel | 0.00846 | 0.00091 | **−0.496** | converging |
| rbf_gaussian_prior | 0.00139 | 0.00016 | −0.459 | converging |
| local_exponent_direction | 0.01441 | 0.00161 | −0.490 | converging |
| non_symplectic_dense_mixing | 0.00710 | 0.00099 | −0.443 | converging |
| sign_breaking_mode_reassignment | 0.00779 | 0.00087 | −0.494 | converging |
| phase_correlated_features | 0.00749 | 0.00065 | −0.534 | converging |
| **anisotropic_angular_prior** | 0.06371 | 0.06181 | **−0.005** | **pinned** |
| **angular_mode_scramble** | 0.10067 | 0.09687 | **−0.018** | **pinned** |

Exactly the two architectures with anisotropic priors are pinned; the other six
converge at `N^(−1/2)`.

## 4. The joint structural suite

Five axes, each with a pre-specified threshold recorded in the config:

| axis | criterion |
|---|---|
| translation invariance | `δ_TI ≤ 1e-8` |
| rotational invariance | `δ_RI` exponent `≤ −0.25` (converging) |
| pairing fidelity | `δ_iso` exponent `≤ −0.25` (converging) |
| physical admissibility | `λ_min > 0` |
| raw Virasoro fidelity | `b_Ω` null `−1/2` not rejected (where Gate 2 ran it) |

Following the instruction that each control need only fail **the property it was
designed to violate**:

| architecture | designed to violate | fails that axis? | all axes failed |
|---|---|---|---|
| **selected_log_kernel** | — | — | **none** |
| rbf_gaussian_prior | radial spectrum | ✔ | pairing fidelity; raw Virasoro; Ward |
| local_exponent_direction | radial exponent / positivity | ✔ | pairing fidelity; **physical admissibility** (λ_min = −0.644) |
| sign_breaking_mode_reassignment | symplectic sign structure | ✔ | pairing fidelity; **physical admissibility** (−1.158); raw Virasoro |
| non_symplectic_dense_mixing | symplectic structure | ✔ | pairing fidelity; **physical admissibility** (−11.44) |
| angular_mode_scramble | rotational invariance | ✔ | pairing fidelity; **rotational invariance**; Ward |
| anisotropic_angular_prior | rotational invariance | ✔ | pairing fidelity; **rotational invariance** |
| phase_correlated_features | translation invariance | ✔ | **translation invariance** — and nothing else |

**Every control fails its designed axis. The selected architecture fails none.
Exactly one architecture in the table has a clean sheet.**

The last row is the gate's point. `phase_correlated_features` passes physical
admissibility, pairing fidelity, Ward behaviour *and* rotational invariance, and
is isolated solely by the axis this gate added.

## 5. Pooled convergence

`convergence_joint_fit.csv` pools exponents **only where an `N^(−1/2)` law is
actually predicted** — random-feature kernel defects. Eight rows: Gate 1 `δ_iso`,
Gate 2 `b_Ω`, and Gate 7's three quantities under both NS constructions.

    pooled exponent = −0.4991 ± 0.0047   (8 rows)
    vs −1/2:  p = 0.853
    Cochran Q = 0.775 on 7 dof, p = 0.998,  I² = 0%

Zero measured heterogeneity across two gates, two NS constructions and four
distinct observables. Three families are **reported but never pooled**, each
with its reason recorded in the row:

- **Ward error** — a different functional of the kernel, with no `N^(−1/2)`
  prediction (the selected architecture's is −0.581);
- **stationarity defect** — structural; for a phase-correlated architecture it
  is *not* predicted to converge at all, and does not;
- **isotropy defect** — `N^(−1/2)` only for an isotropic prior and pinned
  otherwise, so convergence is the discriminator rather than a common law.

Pooling any of these into the `N^(−1/2)` estimate would have manufactured
heterogeneity or, worse, hidden a structural defect inside an average.

## 6. Ledger status

| claim | status |
|---|---|
| selected architecture is uniquely isolated by the joint structural suite | `accepted` |
| translation invariance is exact for independent-phase architectures | `accepted` — ≤ 1.73e-16 over 7 architectures × 5 widths × 4 seeds |
| `phase_correlated_features` is excluded | `accepted` — δ_TI = 0.228, flat over a 100× width range |
| `phase_correlated_features` excluded by any *pre-existing* axis | `fails_current_claim` — it passes all four; only the new axis isolates it |
| `anisotropic_angular_prior` excluded by Ward behaviour | `fails_current_claim` — Ward exponent −0.270 is converging; it is excluded by the isotropy defect instead |
| `angular_mode_scramble`, RBF, and the three λ_min<0 controls excluded | `accepted` |
| `N^(−1/2)` random-feature law | `accepted` — pooled −0.4991 ± 0.0047, I² = 0% |
| Ward / stationarity / isotropy exponents as instances of that law | `diagnostic_only` — reported, never pooled |
