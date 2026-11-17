<!-- generated 2026-09-16T20:42:36Z from docs/hardening/gate07_ns_summary.md | commit 2900ae4aa10f | config f51ad38719b59543 -->

# Gate 7 — NS sector: an independent Grassmann architecture

**Verdict: PASS.**

The NS claim no longer rests on a construction whose width and randomness are
the boson's. An explicit randomized Cauchy architecture was added, and it
reproduces the CAR and the super-Virasoro relations with `N^(−1/2)` convergence
and a central charge of `3/2`. The native derived construction is retained and
agrees.

Runner `scripts/run_gate07_ns.py`; architecture `src/nnft_virasoro/ns_cauchy.py`;
design record `NS_IMPLEMENTATION_MAP.md`. Grid: 3 bosonic architectures × 7
widths (10³ … 10⁶) × 8 seeds = 336 pairing rows and 336 algebra rows.

## 1. Why the gate was additive

The native NS code *is* a real architecture — real momentum draws, exact ordered
pairing, no perturbation of the target CAR — but its kernel is derived from the
bosonic seed as `K_ψψ = −∂_z K_φφ`, so it has no Grassmann degrees of freedom of
its own. Neither branch of the mission's Gate-7 instruction applied verbatim, so
the native construction was **kept** and an independent one **added**. This is
not a redefinition of the experiment, so stop condition 4 does not apply.

## 2. The added architecture, and its declared proposal

`1/(z−w) = Σ_{n≥0} w^n z^{−n−1}` for `|z|>|w|`, sampled by importance:

    psi_N(z) = (1/√N) Σ_j a_j z^{−n_j−1/2} / √P(n_j),   n_j ~ P,  {a_j,a_l} = δ_jl

Width = number of drawn modes; randomness = the mode draw. Neither is shared
with the bosonic sample.

**Unbiasedness** holds at every proposal tested (18 rows, all
`unbiased_within_3sem`), including the ones with divergent variance — as it
must, since unbiasedness does not depend on the proposal.

**The finite-variance condition is real, and demonstrated rather than asserted.**
For geometric `P(n) = (1−q)q^n` the second moment is a geometric series of ratio
`(|w|/|z|)²/q`, so the variance is finite iff `q > (r_in/r_out)² = 1/ρ² = 0.25`.
The *exact* variance, evaluated against the truncation `n_max`:

| q | n_max=50 | n_max=100 | n_max=200 | n_max=400 | flat? |
|---|---|---|---|---|---|
| 0.10 | 7.30e+19 | 5.76e+39 | 3.59e+79 | **inf** | no |
| 0.20 | 2.19e+05 | 1.53e+10 | 7.53e+19 | 1.81e+39 | no |
| 0.30 | 3.2547 | 3.2551 | 3.2551 | 3.2551 | **yes** |
| 0.50 | 0.9694 | 0.9694 | 0.9694 | 0.9694 | **yes** |
| **0.75** *(used)* | 1.9694 | 1.9694 | 1.9694 | 1.9694 | **yes** |
| 0.90 | 5.8604 | 5.8923 | 5.8925 | 5.8925 | **yes** |

The flat/divergent split falls exactly at the predicted threshold. The runner
refuses to start if the configured proposal violates it.

This had to be computed exactly: a **sample** standard deviation cannot show the
divergence. With finitely many draws the rare heavy-weight modes are essentially
never drawn, so a divergent estimator still returns a modest, merely *unstable*,
number — at `q = 0.10` the sample s.d. over five truncations wandered over
0.073–0.268 with no trend, which would have been easy to misread as noise. The
exact second moment is what makes the condition a demonstration.

Where the variance is finite, theory matches measurement: at `q = 0.75`,
predicted `9.85e-05` against empirical `1.04e-04`; at `q = 0.30`, `1.63e-04`
against `1.36e-04`.

`q = 0.75` was chosen for the **pairing**, not for the pointwise kernel. The
binding constraint on the CAR error is the rarest mode in the window, weight
`P(n) = (1−q)q^n` at `n = 4`, maximized at `q = n/(n+1) = 0.8`. The pointwise
kernel variance is instead minimized near `q = 0.5`. The two criteria differ and
`q = 0.75` serves the one this gate measures.

## 3. Two evaluation paths agree to machine precision

The ordered pairing of `K^(N)` has a closed form — the contour integrals select
a single Laurent mode, leaving the empirical importance ratio on the canonical
antidiagonal. Because that makes part of the structure exact by construction, it
is **not** used as the sole path: `cauchy_pairing_quadrature` recomputes the same
object by the same torus quadrature used for the ideal kernel.

    max |exact − quadrature| = 5.58e-16   over 72 rows

The quadrature path is capped at `N ≤ 10⁴`; its cost is `O(q²N)` and it is an
evaluation check, not a statistical one.

## 4. Convergence: both constructions, all three quantities

Power-law fits over the seven widths, selected architecture, pre-Darboux
(`ns_super_virasoro_fits.csv`):

| quantity | native derived | randomized Cauchy | vs −1/2 |
|---|---|---|---|
| CAR relative error | −0.506 ± 0.019 | −0.515 ± 0.022 | p = 0.75 / 0.54 |
| `[L_m, G_r]` residual | −0.497 ± 0.009 | −0.497 ± 0.009 | p = 0.78 / 0.78 |
| `{G_r, G_s}` residual | −0.499 ± 0.012 | −0.499 ± 0.012 | p = 0.93 / 0.94 |

The null `−1/2` is not rejected in any of the six fits. This is the
`N^(−1/2)` random-feature law the architecture-to-operator argument requires,
now established for an NS architecture that does not borrow the boson's width.

**Central charge → 3/2** (selected architecture, mean over 8 seeds):

| N | native | randomized Cauchy |
|---|---|---|
| 10³ | 1.394 | 1.263 |
| 10⁴ | 1.513 | 1.514 |
| 10⁵ | 1.472 | 1.473 |
| 10⁶ | **1.5017** | **1.5019** |

with `[L,G]` residuals falling 13.05 → 0.437 and `{G,G}` 4.47 → 0.143 over the
same range.

## 5. Two honest limitations

**(a) The super-Virasoro residual is bosonic-dominated.** The native and Cauchy
constructions differ by a factor of ~10 in CAR relative error (at N=10⁶:
0.0250 vs 0.0024), yet their `[L,G]` residuals differ by only **0.26–0.41%**:

| N | native `[L,G]` | Cauchy `[L,G]` | relative difference |
|---|---|---|---|
| 10³ | 13.0538 | 13.0003 | 0.41% |
| 10⁵ | 1.3499 | 1.3457 | 0.31% |
| 10⁶ | 0.43737 | 0.43622 | 0.26% |

The residual is set almost entirely by the bosonic `Ω`. It is therefore a weak
test of the fermionic sector, and the agreement between the two constructions in
§4 should not be over-read: on `[L,G]` and `{G,G}` they would have agreed
almost regardless. The **CAR relative error** is the fermion-specific
observable, and that is where the two constructions genuinely differ.

**(b) CAR leakage is not an observable for this architecture.** The closed form
puts the entire estimator on the canonical antidiagonal, so
`fermionic_pairing_leakage` is **identically 0.0** in all 168 randomized-Cauchy
rows, against 0.0099–0.859 for the native construction. This is structural, not
a result: the Cauchy architecture cannot produce off-antidiagonal leakage. It
must not be quoted as evidence that this architecture is better behaved. This
limitation was predicted in `NS_IMPLEMENTATION_MAP.md` §5 before the run, not
discovered afterwards.

Consequently the CAR error magnitudes in §5(a) are **not** an apples-to-apples
comparison: the Cauchy construction has fewer error channels by construction.
What is comparable, and what the gate rests on, is the **exponent** — both
converge at `N^(−1/2)`.

## 6. Ledger status

| claim | status |
|---|---|
| NS sector has an architecture with its own width and randomness | `accepted` — explicit randomized Cauchy factorization |
| CAR and super-Virasoro converge at `N^(−1/2)` | `accepted` — six fits, null −1/2 not rejected (p ≥ 0.54) |
| central charge `c = 3/2` | `accepted` — 1.5017 (native), 1.5019 (Cauchy) at N=10⁶ |
| finite-variance proposal condition `q > 1/ρ²` | `accepted` — exact variance flat above, divergent below |
| native derived construction agrees with the independent one | `accepted_with_reformulation` — true, but the residual is bosonic-dominated (§5a) |
| CAR leakage as NS architecture discrimination | `diagnostic_only` — identically zero by construction (§5b) |

All Gate-7 super-Virasoro rows are **pre-Darboux** (`pre_darboux = True`); no
canonicalized residual is used as architecture evidence (mission rule 3).
