# Final scientific gate matrix — JHEP submission hardening

State at commit `81b5757` (plus the tree-hygiene commit that follows it).
Test suite: **215 passing**. Claim ledger: **39 adjudicated claims**, no
ambiguous `failed_gate` label.

## 1. Gate matrix

| gate | subject | verdict | what closed it |
|---|---|---|---|
| **0** | provenance map, repo disposition | **CLOSED** | `IMPLEMENTATION_MAP.md`; the August release (`b3fd73d`, config `278e438b94465564`) is untouched |
| **1** | intrinsic pre-Darboux physical admissibility | **PASS** | δ_iso and λ_min discriminate; δ_star does not. RBF isotropy defect pinned at 1.373–1.379 |
| **2** | PBW stability theorem | **PASS** | Ratio bounded in both sweeps; the ε-sweep plateau 1.6502–1.8151 is now a release artifact |
| **3** | TT provenance and the c = 1 claim | **PASS WITH REFORMULATION** | `q_g = 1` estimator-independent: GLS 1.00830 ± 0.03377 (direct), 0.99941 ± 0.00133 (Wick) |
| **4** | Ward metric crosswalk | **PASS WITH REFORMULATION** | Numbers are absolute not normalized; sampled campaign is M^(−1/2) noise; block-level paired ratio 31.74 [28.09, 35.87], p = 1.6e-10 |
| **5** | estimator-class separation | **PASS** | Folded into Gates 3 and 4; no artifact pools estimator classes |
| **6** | kernel normalization conventions | **PASS** | 125/125 graded checks, including two controls that must fail |
| **7** | NS sector | **PASS** | Native construction retained as primary; randomized Cauchy as independent cross-check; six exponents all ≈ −1/2 |
| **8** | architecture synthesis | **PASS** | Selected architecture fails no axis; every control fails its designed axis; exactly one clean sheet |

**No scientific gate remains open.** Zero claims carry `needs_new_computation`.

## 2. What Gate 8 added, and why it mattered

`phase_correlated_features` survived every pre-existing axis — and beat the
selected architecture on three of them (λ_min 0.953 vs 0.918; δ_iso exponent
−0.591 vs −0.488; Ward 0.0099 vs 0.0128 at N = 10⁶). The reason was structural:
every existing diagnostic is a functional of the kernel built from contour
integrals or the pairing matrix, and all of them are blind to loss of
translation invariance. More width would never have excluded it.

Two intrinsic symmetry diagnostics were added, on the **true two-argument
kernel** (the one-argument helpers take only `z − w` and are translation
invariant by construction — routing a phase-correlated architecture through them
returns exactly zero and hides the effect):

| | separates by | selected | the control it catches |
|---|---|---|---|
| `δ_TI` stationarity | **exact vs broken** | ≤ 1.7e-16 at every width | `phase_correlated_features`: 0.228, flat over a 100× width range |
| `δ_RI` isotropy | **converging vs pinned** | exponent −0.496 | `anisotropic_angular_prior` (−0.005), `angular_mode_scramble` (−0.018) |

The two are read differently on purpose. Stationarity is exact for an
independent-phase architecture, so its verdict is *vanishing*. Isotropy is only
statistical at finite N — the empirical angular distribution of N draws is
anisotropic at order N^(−1/2) — so its verdict is *converging*, not zero.

## 3. Joint structural suite

| architecture | designed to violate | fails it? | all axes failed |
|---|---|---|---|
| **selected_log_kernel** | — | — | **none** |
| rbf_gaussian_prior | radial spectrum | ✔ | pairing fidelity; raw Virasoro; Ward |
| local_exponent_direction | radial exponent / positivity | ✔ | pairing fidelity; physical admissibility (λ_min = −0.644) |
| sign_breaking_mode_reassignment | symplectic sign structure | ✔ | pairing fidelity; physical admissibility (−1.158); raw Virasoro |
| non_symplectic_dense_mixing | symplectic structure | ✔ | pairing fidelity; physical admissibility (−11.44) |
| angular_mode_scramble | rotational invariance | ✔ | pairing fidelity; rotational invariance; Ward |
| anisotropic_angular_prior | rotational invariance | ✔ | pairing fidelity; rotational invariance |
| phase_correlated_features | translation invariance | ✔ | **translation invariance — and nothing else** |

Pooled convergence exponent over the eight quantities where an `N^(−1/2)` law is
predicted: **−0.4991 ± 0.0047**, Cochran `Q = 0.78` on 7 dof (`p = 0.998`),
`I² = 0%`. Ward, stationarity and isotropy exponents are reported but never
pooled, each with its reason recorded in the row.

## 4. Claims the manuscript currently makes that the computation contradicts

Eight, each with an identified remediation — none requiring new computation.

| claim | gate | remediation |
|---|---|---|
| `c = 1.016 ± 0.011` as an *independent* determination | 3 | say "Wick-factorized"; quote the direct value for independence |
| sharp-annulus Wick at N = 750 labelled `failed_gate` | 3 | remap to `fails_current_claim`; present as the regulator theorem |
| Ward exclusion of `local_exponent_direction` | 4 | it converges at −0.911; exclude it by λ_min < 0 instead |
| δ_star as an architecture discriminant | 1 | RBF has the *smallest* δ_star; use δ_iso and λ_min |
| bounded PBW ratio as sufficient | 2 | every control is bounded too; necessary, not sufficient |
| pointwise ideal limit of `K_JJ` | 6 | it does not converge; the limit is meaningful only after ordered pairing |
| Ward exclusion of `anisotropic_angular_prior` | 8 | Ward exponent −0.270 converges; exclude it by the isotropy defect |
| `phase_correlated_features` excluded by a pre-existing axis | 8 | it passes all four; only the stationarity audit isolates it |

## 5. Remaining work before the final clean hardening release

**Engineering only — no science is outstanding.**

1. **One commit, one config hash, clean tree.** The hardening namespace currently
   carries 6 config hashes and 7 commits with `git_dirty = True`, because gates
   were developed incrementally. The final release requires emptying
   `results/jhep_submission_hardening/` and running the whole suite once from a
   clean tree. `test-output.xml` has been untracked so the tree stays clean
   after the test suite runs.
2. **Master runner** `scripts/run_jhep_submission_hardening.py` with
   `--skip-existing`, `--only-gate`, `--quick`.
3. **Release bookkeeping**: `all_metrics.csv`, `manifest.json`,
   `environment.txt`, `run_commands.sh`, `summary.md`.
4. **Manuscript regeneration from the ledger** (explicitly deferred).

Estimated cost of the clean run: ~3.5 h wall, dominated by the Gate-3 direct
extension (112 min on 12 cores) and Gate 7 (~10 min); everything else is
minutes.

## 6. The August release is intact

`git commit b3fd73dd887b`, `config hash 278e438b94465564`. Nothing in
`results/jhep_frozen/` or `figures_jhep_frozen/` was modified at any point; the
hardening campaign writes only to its own namespace.
