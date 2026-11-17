# Gate 2 — pre-Darboux Virasoro stability

**Verdict: PASS.**

The raw Virasoro residual is controlled by the pre-Darboux pairing error at
the rate the theorem predicts, for the selected architecture under the
tapered regulator, and the controls fail in the expected way.

## Definition actually used, and one necessary correction

The residual is expanded in a **fixed** degree-<=2 PBW basis
`{1} u {alpha_i alpha_j : i <= j}` over modes `-L..L`. The basis depends on
the mode window only, never on `Omega` (mission requirement), and it
**includes the zero mode**: the truncated Sugawara sum runs over `p` with
`|p| <= L` and `|m-p| <= L`, so `alpha_0` genuinely appears even though the
measured pairing is supported on `M^x = {-L..-1, 1..L}`. Omitting it made the
expansion incomplete (the first implementation raised on the word
`alpha_-3 alpha_0`).

A second correction is load-bearing. The residual as literally defined does
**not** vanish at `Omega = Omega_can`: it leaves the truncation **edge
remainder**, here `14.0134` in combined PBW norm (an
`Omega`-independent floor, the same object enumerated by the August WP-G
audit, which annihilates the filtered Fock subspace but is a nonzero abstract
element). Forming `R/(eps + eps^2)` from the unsubtracted residual would
diverge as `eps -> 0` purely because of that floor and would misattribute a
known cutoff artifact to the architecture. Both quantities are therefore
stored: `*_raw` as defined, and `*_excess` with the `Omega_can` baseline
removed in the same fixed basis. **The theorem is tested on the excess**,
which vanishes identically at `eps = 0` (verified to 1e-12).

Scalar coefficient, quadratic coefficient vector and combined norm are stored
separately, as required.

## Result: selected architecture, tapered regulator

| N | eps_Omega | R_excess | R_raw | R/(eps+eps^2) |
|---|---|---|---|---|
| 10,000 | 4.049 | 6.785 | 15.8001 | 0.368 |
| 30,000 | 3.702 | 6.164 | 15.0611 | 0.382 |
| 100,000 | 2.417 | 3.875 | 14.3917 | 0.517 |
| 300,000 | 1.138 | 1.852 | 14.2393 | 0.809 |
| 1,000,000 | 0.4552 | 0.753 | 14.0134 | 1.175 |

    eps_Omega ~ N^(-0.483 +/- 0.083)   p(b=-1/2) = 0.85  -> null NOT rejected
    R_excess  ~ N^(-0.487 +/- 0.078)   R^2 = 0.928

The two exponents agree to well within their errors, i.e. the residual is
**linear** in the pairing error over this range, which is the `O(eps)` branch
of the theorem.

## Two five-decade sweeps of the same theorem — not to be conflated

The ratio `R_excess/(eps + eps^2)` is bounded in both, but they are different
experiments and they sit in different places, so quoting one range against the
other would be a rule-11 error:

| sweep | what spans five decades | ratio | reading |
|---|---|---|---|
| **epsilon sweep** (`gate02_stability_epsilon_sweep.csv`) | `eps`, dialled directly by perturbing the canonical pairing at fixed mode cutoff | **1.6502 – 1.8151** (reference direction) | sits **on** the plateau |
| **architecture grid** (`gate02_predarboux_virasoro_fits.csv`) | the **width** `N`; `eps` follows | 0.206 – 1.311 (per-width mean 0.368 → 1.175) | still **climbing toward** it |

The epsilon sweep is the cleaner statement of the theorem, because it isolates
the perturbation: over **12 independent perturbation directions** the band is
1.5583 – 2.1106, and within every single direction the ratio varies by a factor
of only **1.100** across the five decades. The plateau is therefore a property of
the theorem, not of one lucky direction.

The width sweep is the physically relevant one, and it has not yet reached the
plateau: at `N = 10^6` the ratio is still rising (mean 1.175). That is expected —
`eps` at the largest audited width is far larger than the smallest `eps` in the
controlled sweep — and it is the honest way to report it. Boundedness, which is
what the theorem asserts, holds in both.

## Controls behave as they must

| regulator | architecture | b_Omega | p(b=-1/2) | b_R | ratio max |
|---|---|---|---|---|---|
| sharp_annulus | `rbf_gaussian_prior` | +0.0004 +/- 0.0006 | 3.2e-09 (**rejected**) | +0.0010 | 0.307 |
| sharp_annulus | `selected_log_kernel` | -0.1462 +/- 0.0387 | 0.0028 (**rejected**) | -0.1550 | 0.273 |
| sharp_annulus | `sign_breaking_mode_reassignment` | -0.0727 +/- 0.0194 | 0.0002 (**rejected**) | -0.0708 | 0.188 |
| tapered_log_window | `rbf_gaussian_prior` | -0.0003 +/- 0.0005 | 2.1e-09 (**rejected**) | -0.0007 | 0.310 |
| tapered_log_window | `selected_log_kernel` | -0.4830 +/- 0.0825 | 0.85 (not rejected) | -0.4873 | 1.175 |
| tapered_log_window | `sign_breaking_mode_reassignment` | +0.0016 +/- 0.0114 | 2.6e-05 (**rejected**) | -0.0021 | 0.151 |

RBF and sign-breaking have `b_Omega` statistically indistinguishable from
**zero** — their pairings do not converge at all, and the `-1/2` null is
rejected at `p ~ 1e-9` and `p ~ 3e-5`. Their residual ratios are bounded only
because numerator and denominator are both stuck; a bounded ratio is
therefore **necessary but not sufficient**, and must be read together with
`b_Omega`.

Under the sharp annulus even the selected architecture manages only
`b_Omega = -0.146 +/- 0.039` and the `-1/2` null **is** rejected
(p = 0.0028), consistent with the sharp regulator being a negative
control rather than a viable regulator.

## Manuscript implication

Supports the redraft's pre-Darboux convergence claim. Two wording
requirements follow:

1. the residual quoted must be identified as the **edge-baseline-subtracted**
   PBW excess, not the bare residual, otherwise the stated bound is not the
   quantity being bounded;
2. `eps_Omega` here is the **absolute** Frobenius norm `||Omega - Omega_can||_F`,
   not the relative error used elsewhere (at N=1e6, 0.4552 absolute vs
   0.0588 relative, since ||Omega_can||_F = sqrt(60)).

## Tests

`tests/test_gate02_pbw.py`, 10 passing: fixed-basis independence of `Omega`,
degree <= 2, separate scalar/quadratic storage, the edge-remainder identity,
linearity in `eps`, and bounded `R/(eps+eps^2)` over five decades for four
random perturbation directions.
