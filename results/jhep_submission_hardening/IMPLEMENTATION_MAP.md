<!-- generated 2026-09-16T20:42:36Z from docs/hardening/IMPLEMENTATION_MAP.md | commit 2900ae4aa10f | config f51ad38719b59543 -->

# Implementation / provenance map — JHEP submission hardening (2026-09-15)

Phase 0 deliverable. Written **before** any expensive computation, as mandated.

August frozen release (immutable checkpoint, verified intact this session):

    code commit   b3fd73dd887b92de783ee8d825075d66641c1436
    config hash   278e438b94465564
    ledger rows   235

---

## 0. Finding of fact: the specification documents are absent

The mission directs me to read ten files "in full" before touching code. **None
of them exists.** Searched: the repository, `~/Downloads` to depth 4, and `$HOME`
to depth 6.

| file | status |
|---|---|
| `GATE_01_INTRINSIC_PHYSICAL_DARBOUX.md` … `GATE_08_EDITORIAL_NOTATION_STATISTICS.md` | **absent** (all eight) |
| `JHEP_SUBMISSION_HARDENING_CHECKPOINT_2026-09-14.md` | **absent** |
| `Draft-JHEP-Final-Scientific-Redraft.tex` | **not in this repo**; found at `../JHEP_final_scientific_redraft_bundle/` |

**Disposition.** The mission prompt itself carries a complete, self-consistent
inline specification for every gate: mode ordering and all five Gate-1 intrinsic
defects plus the principal canonicalizer; the Gate-2 PBW residual and its
`O(eps + eps^2)` target; the Gate-3 provenance schema, Wick/direct estimators and
the direct–Wick cumulant identity; the Gate-4 crosswalk schema, exact Ward
targets and the normalized complex-RMS metric; the Gate-6 `J_2` channel and
`sigma_phi` normalization; the Gate-7 randomized Cauchy factorization with a
geometric proposal; and the Gate-8 regression and synthesis schemas.

I therefore treat **the prompt as the operative specification**. Where the prompt
is silent I make a choice and record it here and in the gate summary rather than
guess at absent text. Two places where absent fine print could still matter are
flagged inline below (Gate 1 "iff" statement, Gate 7 acceptance criteria).

The redraft manuscript is read from the sibling bundle and treated as the
authoritative statement of current claims. It is **not** modified by this
campaign.

---

## 1. Artifact -> native generator map

Every frozen artifact resolves to a generator. No name remapping was needed.

| frozen artifact | native module(s) | runner |
|---|---|---|
| `ordered_pairing_grid.csv` | `ordered_pairing`, `kernels`, `nonholomorphy` | `scripts/run_ordered_pairing_audit.py` (stage a3) |
| `ordered_pairing_radius_plateau.csv` | `nonholomorphy` | same, stage a3 |
| `predarboux_convergence_seed_rows.csv` / `_fits.csv` | `ordered_pairing`, `algebra`, `fits` | same, stage a5 |
| `architecture_controls_seed_rows.csv` | `architectures`, `ordered_pairing`, `algebra`, `darboux`, `ward` | same, stage a6 |
| `darboux_physical_admissibility.csv` | `darboux.admissibility_report` | same, stage a6 |
| `darboux_admissibility_vs_width.csv` | `darboux` + `fits.power_law_fit` | same, stage a6 |
| `sampled_tt_seed_rows.csv`, `sampled_tt_central_charge.csv` | `stress_tensor`, `sampling` | `scripts/run_sampled_tt.py` |
| `analytic_ward_points.csv` | `ward.analytic_ward` | `scripts/run_ward_audit.py` |
| `deterministic_feature_ward_points.csv` | `ward.deterministic_ward` | same |
| `sampled_ward_seed_rows.csv`, `sampled_ward_summary.csv` | `ward.sampled_ward`, `sampling` | same |
| `ns_pairing_seed_rows.csv`, `ns_super_virasoro_seed_rows.csv`, `ns_frozen_metrics.csv` | `ns_fermion`, `algebra`, `darboux` | `scripts/run_ns_audit.py` |
| `finite_width_cumulant_seed_rows.csv`, `finite_width_cumulant.csv` | `vertex.sampled_cumulant4` | `scripts/run_vertex_and_cumulant.py` |

## 2. Decisive structural finding: Omega was never persisted

`find` over the whole repo returns **no** `.npy/.npz/.h5/.pkl`. The pre-Darboux
pairing matrices were computed in memory and only scalar summaries were written.

Consequence: Gates 1 and 2 **cannot** be done as pure post-processing. They need
one shared recomputation of the pairing that **persists Omega**, after which both
gates post-process the same matrices. Measured cost of that recomputation at the
frozen controls geometry (`lambda_ir=1e-2`, `lambda_uv=1e2`, `mode_cutoff=4`,
`r0=1`, `rho=2`; widths `1e4,3e4,1e5,3e5,1e6`):

    per (architecture, regulator class, seed) over the 5-width grid   7.3 s
    full 8 x 2 x 4 = 64 combinations                                  ~7.8 min

This is cheap, so Gate 1 and Gate 2 are run over the **exact** frozen grid — same
eight architectures, both regulator classes, same widths, same four named seeds.

## 3. Gate-by-gate plan

| gate | native module | runner (new) | config section | input | output | estimator class | runtime | recompute or post-process |
|---|---|---|---|---|---|---|---|---|
| 6 | `kernels`, `ordered_pairing` | `run_gate06_normalization.py` | `gate06` | none (analytic) | `gate06_kernel_normalization_checks.csv`, `gate06_normalization_crosswalk.md` | `analytic_closed_kernel` | < 1 min | analytic only |
| 1 | new `intrinsic_admissibility`; `architectures`, `ordered_pairing` | `run_gate01_intrinsic.py` | `gate01` | config | `gate01_intrinsic_admissibility_{seed_rows,fits}.csv`, `_summary.md`, persisted Omega | `deterministic_finite_feature` | ~8 min | **recompute** (Omega absent), then post-process |
| 2 | new `pbw_residual`; `superalgebra`, `algebra` | `run_gate02_pbw.py` | `gate02` | Omega from Gate 1 | `gate02_predarboux_virasoro_{seed_rows,fits}.csv`, `_summary.md` | `deterministic_finite_feature` | ~10 min | post-process Gate-1 Omega |
| 3+5 | `stress_tensor`, `sampling` | `run_gate03_tt.py` | `gate03` | frozen `sampled_tt_seed_rows.csv` + new blocks | `tt_estimator_provenance.csv`, `sampled_tt_{wick,direct}_blocks.csv`, `sampled_tt_estimator_comparison.csv` | `sampled_finite_network` | audit < 1 min; extension only if needed | **audit first**, extend only if the existing direct result cannot support the wording |
| 4+5 | `ward`, `sampling` | `run_gate04_ward.py` | `gate04` | frozen ward rows + new normalized metric | `ward_metric_crosswalk.csv`, `sampled_ward_normalized_{blocks,summary}.csv`, `sampled_ward_leave_one_block_out.csv` | all three, kept separate | ~20 min | recompute normalized metric from block-level paired ratios |
| 7 | `ns_fermion` + new Cauchy architecture | `run_gate07_ns.py` | `gate07` | config | `NS_IMPLEMENTATION_MAP.md`, `ns_kernel_seed_rows.csv`, `ns_{pairing,super_virasoro}_seed_rows.csv`, `ns_super_virasoro_fits.csv` | `deterministic_finite_feature` | ~15 min | **additive**: retain native, add explicit Cauchy |
| 8 | `fits` | `run_gate08_synthesis.py` | `gate08` | all above | `convergence_joint_fit.csv`, `architecture_synthesis.csv` | mixed, labelled per row | ~2 min | post-process |

## 4. Provenance answers already established (before any run)

These three were resolved during Phase 0 by reading the frozen CSVs directly.

**Gate 3 — origin of `c_sampled = 1.016 +/- 0.011`.** Uniquely located:
`regulator_class=tapered_log_window`, `architecture=selected_log_kernel`,
`estimator_variant=wick_factorized`, `n_features=250`, `fit_window_index=0`,
`n_blocks=12`, `n_networks=60000`. It is **Wick only**, **not** combined, and it
is at the **smallest** audited width (the grid is `{250, 750}`). The direct
estimator at the same setting gives `2.057 +/- 0.577`; at `N=750`, Wick gives
`1.023 +/- 0.017` and direct `0.751 +/- 0.617`.

**Gate 4 — origin of `0.129 / 12.2 / 2.15`.** All three come from
`deterministic_feature_ward_points.csv`, as the **maximum absolute** stress-Ward
error at `n_features=200000` under the tapered regulator (selected `0.1293`,
angular scramble `12.19`, RBF `2.153`; the `7.57` is the same metric at
`n_features=500`). Two discrepancies with the redraft follow, recorded now and
adjudicated in the Gate-4 summary:
  1. the manuscript calls this quantity **normalized**; no normalized Ward metric
     exists anywhere in the frozen release (`grep` for `norm` returns only
     `normal_ordering_subtraction`). The quantity is **absolute**.
  2. the manuscript attributes it to "the largest `N` **and `M`**"; the
     deterministic finite-feature file has **no `n_networks` column at all**, so
     `M` is undefined for it. This conflates the deterministic and sampled
     estimator classes, which mission rule 7 forbids.

**Gate 7 — is the NS code a real architecture?** Yes. `run_ns_audit.py` draws
genuine random momenta via `architectures.sample_architecture` and
`ns_fermion.fermionic_pairing_empirical_exact` builds
`K_psipsi^(N,Lambda)(z,w) = (L_w/N) sum_j conj(k_j) sin(Re(conj(k_j)(z-w)))`,
then evaluates the ordered contour pairing exactly. Width = `n_features`;
randomness across seeds = the momentum draw. It does **not** perturb the target
CAR matrix. It is, however, *derived* from the bosonic scalar seed
(`K_psipsi = -d_z K_phiphi`) rather than an independent Grassmann field.

Therefore neither branch of the mission's Gate-7 instruction applies verbatim.
Decision: **additive**. Retain the native derived construction as a cross-check
and *also* implement the explicit randomized Cauchy factorization the gate
specifies, giving the NS claim an independent Grassmann architecture with its own
width and its own randomness. This is not a redefinition of the experiment, so
stop condition 4 does not apply.

## 5. Rules carried into every runner

* covariance is never identified with a commutator;
* `analytic_closed_kernel` / `deterministic_finite_feature` /
  `sampled_finite_network` never share an uncertainty;
* post-Darboux closure is reported as non-discriminating;
* no collaborator alpha-sweep artifact is read (verified absent from the repo);
* every row carries `config_hash`, `git_commit`, `seed`, `estimator_class`;
* `lambda_ir`, `lambda_uv`, `spectral_regulator_class` and `mode_cutoff` are four
  distinct columns — no column named `Lambda` carries two meanings;
* the August release is read-only.
