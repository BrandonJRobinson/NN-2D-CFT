<!-- generated 2026-09-16T20:42:36Z from docs/hardening/gate06_normalization_crosswalk.md | commit 2900ae4aa10f | config f51ad38719b59543 -->

# Gate 6 — Kernel normalization and convention crosswalk

**Verdict: PASS.**

Runner `scripts/run_gate06_normalization.py`; table
`gate06_kernel_normalization_checks.csv` (128 rows: **125 graded, all pass**;
3 recorded as diagnostic). Purely analytic — no sampling, no seeds, estimator
class `analytic_closed_kernel` throughout.

This gate exists so that every later gate inherits a *pinned* convention rather
than an implicit one. It was run before any expensive job, as the mission
requires.

## 1. Why this gate is graded in three kinds

A convention check that can only ever pass is worthless. Every row carries a
`check_kind`:

| `check_kind` | meaning | rows |
|---|---|---|
| `must_match` | measured must land within `tolerance` of the convention target | 125 |
| `must_exceed` | a **control that must fail**: measured must differ from the target by at least `must_exceed` | 2 of the 125 |
| `diagnostic` | recorded as evidence, never graded | 3 |

The two `must_exceed` rows are the ones that give the gate its teeth: they
confirm that the spin-zero radial moment and the single-ordering contour
integral genuinely *do not* reproduce the convention. Without them, C8 and C10
would pass for reasons unrelated to the conventions they claim to fix.

## 2. The crosswalk

| id | quantity | convention | measured | tol / bound |
|---|---|---|---|---|
| C1 | `L_w` at zero taper | `= L = log(Λ_UV/Λ_IR)` | exact (0 error, 3 windows) | 1e-12 |
| C2 | `L_w / L` at 15% taper | `= 1 − taper = 0.85` | 0.85 (max err 1.3e-16) | 1e-6 |
| C3 | `σ_φ²` | `= 4 L_w` (κ = 1 calibration) | exact, 6 rows | 1e-12 |
| C4 | `2Z/π` with `Z = 2π L_w` | `= σ_φ²` | exact, 6 rows | 1e-12 |
| C5 | `K_Λ(0)` | `= σ_φ²/2 = 2L` | exact, 3 windows | 1e-12 |
| C6 | `dK_Λ/dlog r` inside the window | `= −σ_φ²/(2L) = −2` | −1.99749, −1.99759, −1.99759 | 5e-3 |
| C7 | `K_JJ(θ)/K_JJ(0)` | `= e^{−2iθ}` (**spin two**) | max err 8.5e-14 over 72 rows | 1e-9 |
| C8 | closed-form `K_JJ(r)` vs `J_2` spectral quadrature | equal | max err 8.0e-14, 12 rows | 1e-6 |
| C8′ | same against the **`J_0`** quadrature | must differ | relative gap 0.371 – 1.621 | ≥ 0.1 |
| C9b | UV tail envelope of `K_JJ^Λ` | `= √Λ_UV · √(2/πr) / (2r)` | err 0.0058 – 0.0150 | 5e-2 |
| C10 | ordered pairing of the ideal kernel | `= m δ_{m+n,0}` | 1.04e-15 | 1e-8 |
| C10′ | **one** ordering `I^>` alone | must differ | 4.0 | ≥ 0.5 |

`Z_{Λ,τ}` is carried explicitly on the C4 rows (e.g. 57.870 at
`Λ ∈ [10⁻², 10²]`, taper 0; 49.190 at taper 0.15).

## 3. The one substantive finding: the ideal limit is *not* pointwise

C9 was originally written as the obvious check — that the regulated current
kernel approaches the ideal `1/(z−w)²` at fixed separation as the window widens.
**It does not**, and the failure is not numerical:

| window | `K_JJ^Λ(1, 0)` | ideal |
|---|---|---|
| `[10⁻², 10²]` | +4.837 | 1 |
| `[10⁻³, 10³]` | −1.389 | 1 |
| `[10⁻⁴, 10⁴]` | −17.230 | 1 |

The values do not converge; they oscillate with a *growing* envelope. The cause
is structural. With `J_2(kr) ~ √(2/πkr) cos(kr − 5π/4)`, the moment

    K_JJ^Λ(r) = ½ ∫^Λ dk k J_2(kr)

has a boundary term of amplitude `√Λ_UV · √(2/πr) / (2r)`. C9b confirms this law
to between 0.6% and 1.5% at `Λ_UV = 10⁵`, across three separations — so the
non-convergence is established quantitatively, not merely observed.

**Consequence, carried into every later gate.** A pointwise comparison of the
regulated kernel to the ideal one is not a meaningful test of the construction,
and no gate uses one. The ideal limit is only meaningful **after ordered contour
pairing**, which integrates the UV oscillation away — and there it is exact
(C10: 1.04e-15). This is also why the architecture diagnostics in Gates 1 and 2
are built from `Ω`, never from kernel values at a point.

The three pointwise rows are retained in the CSV as `diagnostic`, because they
are the evidence for the convention rather than a failure of it.

## 4. Sign convention, recorded to prevent recurrence

`dK_Λ/dlog r = −2`. The finite-difference form

    (K(r₁) − K(r₂)) / log(r₂/r₁)

is **minus** that derivative. An earlier version of this check reported a slope
of `+2.000` against a target of `−2` and was briefly read as a code failure; the
error was in the check expression, not in `kernels.scalar_kernel`. The runner
now computes `(K(r₂) − K(r₁))/log(r₂/r₁)` with the comment stating the trap.

## 5. Provenance

All rows carry `git_commit` and `config_hash`. The values in this document were
produced at config hash `f072a08b64847099`, which post-dates the Gate-1/2 run
(`7ff5ed3b0095b8f3`) because the Gate-3 and Gate-6 config sections were extended
afterwards. **This is expected and is not the release state**: the mission
requires one commit and one config hash across the final release, which the
single clean run at the end of the campaign produces. No gate conclusion in this
document depends on a value that changed between those hashes — Gate 6 is
analytic and its inputs (`regulators`, `tapers`, `mode_cutoff`, `rho`, `r0`)
were not edited, only added to.
