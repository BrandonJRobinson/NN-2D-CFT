<!-- generated 2026-09-16T20:42:36Z from docs/hardening/gate01_intrinsic_admissibility_summary.md | commit 2900ae4aa10f | config f51ad38719b59543 -->

# Gate 1 — intrinsic physical admissibility

**Verdict: PASS WITH REFORMULATION.**

The intrinsic pre-Darboux diagnostics reproduce the manuscript's *qualitative*
architecture-selection conclusion, but they do so through a different quantity
than the manuscript names, and they overturn the specific numbers it quotes.

## What was computed

320 seed rows = 8 architectures x 2 spectral regulator classes x 5 widths
(1e4..1e6) x 4 named seeds, at the frozen controls geometry
(`lambda_ir=1e-2`, `lambda_uv=1e2`, `mode_cutoff=4`, `r0=1`, `rho=2`).
Omega was not persisted by the August release, so the pairing was recomputed
and all 320 matrices are now saved under `omega/` for reuse by Gate 2.

Mode ordering `(alpha_1..alpha_L, alpha_-1..alpha_-L)`; `C=[[0,I],[I,0]]`;
`Omega=[[A,G],[-G^T,B]]`. All five intrinsic defects are computed from Omega
alone — no canonicalizer is chosen — so they are architecture diagnostics in
the sense of mission rule 4.

## Headline numbers (tapered, N=1e6, mean over 4 seeds)

| architecture | delta_star | delta_A | delta_H | delta_iso | lambda_min | principal S_0 |
|---|---|---|---|---|---|---|
| `selected_log_kernel` | 0.009021 | 0.006029 | 0.007601 | **0.05406** | **+0.9184** | exists |
| `phase_correlated_features` | 0.04281 | 0.00667 | 0.04142 | **0.05975** | **+0.9525** | exists |
| `angular_mode_scramble` | 0.4305 | 0.05688 | 0.4088 | **0.3198** | **+0.6927** | exists |
| `anisotropic_angular_prior` | 0.4545 | 0.005987 | 0.4545 | **0.3488** | **+0.6356** | exists |
| `local_exponent_direction` | 0.0415 | 0.03198 | 0.03719 | **1.066** | **-0.6438** | **does not exist** |
| `rbf_gaussian_prior` | 0.001126 | 0.0002695 | 0.0005574 | **1.373** | **+0.1751** | exists |
| `sign_breaking_mode_reassignment` | 1.405 | 0.008117 | 1.405 | **1.537** | **-1.158** | **does not exist** |
| `non_symplectic_dense_mixing` | 1.291 | 0.3914 | 1.214 | **13.28** | **-11.44** | **does not exist** |

## The central finding: delta_star alone does NOT discriminate

The RBF control has the **smallest** intrinsic involution defect of all eight
architectures (0.001126), a factor 8 *smaller* than the selected
Log-Kernel (0.009021), and its width exponent
(-0.571 +/- 0.053) is statistically
indistinguishable from the selected architecture's
(-0.573 +/- 0.053).

An architecture-selection argument resting on the involution defect converging
would therefore **admit the RBF control**. That is a real weakness in the
current framing and it is what Gate 1 was designed to expose.

What separates them is the pair (`delta_iso`, `lambda_min`), i.e. whether the
normalized cross block approaches the identity:

| N | selected delta_iso | RBF delta_iso | selected lambda_min | RBF lambda_min |
|---|---|---|---|---|
| 10,000 | 0.48 | 1.378 | +0.3719 | +0.1734 |
| 30,000 | 0.4402 | 1.379 | +0.4676 | +0.1806 |
| 100,000 | 0.2538 | 1.377 | +0.6691 | +0.1746 |
| 300,000 | 0.1279 | 1.375 | +0.8985 | +0.1739 |
| 1,000,000 | 0.05406 | 1.373 | +0.9184 | +0.1751 |

The selected architecture's isotropy defect falls 0.48 -> 0.05406 (exponent -0.488, R^2 0.941)
with `lambda_min` rising 0.372 -> 0.9184, i.e. towards the
canonical value 1. The RBF control is **pinned** at delta_iso ~ 1.37 across the
whole two-decade range; the power-law fit was *declined* rather than forced,
with recorded status: "not fitted: series is constant to within 5% (no power law to estimate)".
Its `lambda_min` is likewise pinned at ~0.175. RBF respects the involution
structure (it is isotropic) but has permanently wrong radial normalization.

## Crosswalk: this metric is NOT the frozen one (mission rule 11)

The manuscript's `epsilon_*` is `||conj(S) C - C S||_F/||S||_F` for an
*arbitrary* Darboux `S`. The Gate-1 `delta_star` is
`||C conj(Omega) C^T + Omega||_F/||Omega||_F`, computed from Omega alone.
Same conceptual label, different definition, and they disagree — in one case
qualitatively:

| architecture | frozen epsilon_* (arbitrary S) | intrinsic delta_star |
|---|---|---|
| `angular_mode_scramble` | 0.2842 | 0.4305 |
| `anisotropic_angular_prior` | 0.2266 | 0.4545 |
| `local_exponent_direction` | 1.314 | 0.0415 | **<- qualitative disagreement**
| `non_symplectic_dense_mixing` | 1.376 | 1.291 |
| `phase_correlated_features` | 0.04086 | 0.04281 |
| `rbf_gaussian_prior` | 1.088 | 0.001126 | **<- qualitative disagreement**
| `selected_log_kernel` | 0.03159 | 0.009021 |
| `sign_breaking_mode_reassignment` | 1.414 | 1.405 |

`local_exponent_direction` reads 1.314 under the frozen metric (apparently a
gross failure) but 0.0415 intrinsically (apparently fine). The intrinsic
diagnostics place its failure elsewhere — `lambda_min = -0.644`, so no
principal canonicalizer exists at all. The frozen metric conflated the two.

## Manuscript implications (exact)

1. The redraft states the selected architecture "approaches the involution
   condition as `N^-0.720`". Under the intrinsic metric the exponent is
   `-0.573 +/- 0.053` (R^2 0.975), consistent with the
   random-feature rate -1/2 at 1.4 sigma. The number -0.720 belongs to the
   arbitrary-canonicalizer metric and must be relabelled or replaced.
2. The redraft quotes `lambda_min -> 0.994`. That is the frozen metric. The
   intrinsic normalized eigenvalue reaches `0.918` at N=1e6.
3. The redraft says sign-breaking "remains at `epsilon_* = sqrt2` for every
   width". Intrinsically it sits at `1.405` with a
   non-converging exponent, so the qualitative claim survives but the exact
   value sqrt2 is specific to the frozen metric.
4. **The architecture-selection sentence must be re-grounded on `delta_iso`
   and `lambda_min`, not on the involution defect**, because RBF beats the
   selected architecture on the latter.

## Caveat on the boolean flag

`physically_admissible` is False in all 320 rows (0 True). The
criterion `A=B=0, G=G^dagger, G>0` is applied at an **absolute** tolerance
1e-9, while finite-N defects are O(N^-1/2) ~ 1e-3. It is an asymptotic
statement and must be read together with the width fits and the sign of
`lambda_min` — never alone. (The same miscalibration affected
`vacuum_normalizable` in the August release.)

## Tests

`tests/test_gate01_intrinsic.py`, 19 passing: the six-case synthetic matrix
(canonical / star-breaking / non-isotropic positive / Hermiticity-breaking /
indefinite / positive physical) and the iff theorem verified numerically in
both directions over 12 randomized cases.
