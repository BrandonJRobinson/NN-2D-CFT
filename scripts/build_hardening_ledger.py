#!/usr/bin/env python3
"""Assemble the hardening claim ledger from the gate outputs.

One row per adjudicated claim. Values are read from the generated CSVs wherever
they exist, so a ledger entry cannot drift from the artifact it cites; the
*status* is the scientific judgement recorded in each gate summary.

Allowed statuses (no ambiguous `failed_gate`):
``accepted``, ``accepted_with_reformulation``, ``needs_new_computation``,
``fails_current_claim``, ``diagnostic_only``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro.cli import base_parser, setup              # noqa: E402
from nnft_virasoro.io_utils import write_rows                 # noqa: E402

SCRIPT = "scripts/build_hardening_ledger.py"
ALLOWED = {"accepted", "accepted_with_reformulation", "needs_new_computation",
           "fails_current_claim", "diagnostic_only"}


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)

    def rd(name):
        f = outdir / name
        return pd.read_csv(f) if f.exists() else None

    rows = []

    def add(claim_id, gate, metric, status, notes, *, value=np.nan,
            uncertainty=np.nan, ci=None, estimator_class="", script="",
            output_artifact="", units="dimensionless"):
        assert status in ALLOWED, f"disallowed status {status!r}"
        rows.append(stamper.stamp(dict(
            claim_id=claim_id, gate=gate, metric=metric, value=value,
            uncertainty=uncertainty,
            confidence_interval=("" if ci is None
                                 else f"[{ci[0]:.6g}, {ci[1]:.6g}]"),
            units=units, source_estimator_class=estimator_class,
            source_script=script, output_artifact=output_artifact,
            status=status, notes=notes),
            estimator_class=estimator_class or "mixed", seed="multi"))

    # ---------------- Gate 1 -------------------------------------------
    g1 = rd("gate01_intrinsic_admissibility_fits.csv")
    if g1 is not None:
        t = g1[g1.spectral_regulator_class == "tapered_log_window"].set_index(
            "architecture")
        sel, rbf = t.loc["selected_log_kernel"], t.loc["rbf_gaussian_prior"]
        add("g1_intrinsic_predarboux_diagnostics", "gate01",
            "intrinsic pre-Darboux admissibility suite", "accepted",
            "delta_star, delta_A, delta_H, lambda_min and delta_iso computed in "
            "the physical mode order with C = [[0,I],[I,0]]; validated against a "
            "six-case synthetic matrix and the iff theorem over 12 randomized "
            "cases.", estimator_class="deterministic_finite_feature",
            script="scripts/run_gate01_intrinsic.py",
            output_artifact="gate01_intrinsic_admissibility_fits.csv")
        add("g1_delta_star_does_not_discriminate", "gate01",
            "delta_star at max width (RBF vs selected)", "fails_current_claim",
            f"RBF has the SMALLER intrinsic delta_star "
            f"({rbf.delta_star_at_max_width:.4f} vs "
            f"{sel.delta_star_at_max_width:.4f}), so an argument resting on "
            f"delta_star converging would admit the RBF control. Discrimination "
            f"must rest on delta_iso and lambda_min.",
            value=float(rbf.delta_star_at_max_width),
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate01_intrinsic.py",
            output_artifact="gate01_intrinsic_admissibility_fits.csv")
        add("g1_rbf_isotropy_defect_pinned", "gate01",
            "RBF positive-frequency isotropy defect delta_iso", "accepted",
            f"Pinned at {rbf.delta_iso_at_max_width:.3f} across the audited "
            f"width range (the power-law fit was declined: series constant to "
            f"within 5%). Positivity of the cross block does not make RBF "
            f"physically admissible.",
            value=float(rbf.delta_iso_at_max_width),
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate01_intrinsic.py",
            output_artifact="gate01_intrinsic_admissibility_fits.csv")
        add("g1_selected_delta_iso_converges", "gate01",
            "selected delta_iso exponent", "accepted",
            f"delta_iso falls to {sel.delta_iso_at_max_width:.4f} with exponent "
            f"{sel.delta_iso_exponent:.3f} +/- "
            f"{sel.delta_iso_exponent_stderr:.3f}; lambda_min -> "
            f"{sel.lambda_min_at_max_width:.3f} > 0.",
            value=float(sel.delta_iso_exponent),
            uncertainty=float(sel.delta_iso_exponent_stderr),
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate01_intrinsic.py",
            output_artifact="gate01_intrinsic_admissibility_fits.csv")
        add("g1_manuscript_canonicalizer_crosswalk", "gate01",
            "arbitrary-canonicalizer vs principal-canonicalizer metrics",
            "accepted_with_reformulation",
            "The manuscript's -0.720 exponent and lambda_min -> 0.994 are the "
            "arbitrary-canonicalizer metric. local_exponent_direction disagrees "
            "QUALITATIVELY between the two metrics (1.314 frozen vs 0.0415 "
            "intrinsic), so the two must be crosswalked, never compared.",
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate01_intrinsic.py",
            output_artifact="gate01_intrinsic_admissibility_seed_rows.csv")
        add("g1_physically_admissible_flag_is_asymptotic", "gate01",
            "physically_admissible boolean", "diagnostic_only",
            "False in all 320 rows: the absolute tolerance 1e-9 is tighter than "
            "the O(N^-1/2) finite-width defect. The flag must be read with the "
            "convergence fits, not as a standalone verdict.",
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate01_intrinsic.py",
            output_artifact="gate01_intrinsic_admissibility_seed_rows.csv")

    # ---------------- Gate 2 -------------------------------------------
    g2 = rd("gate02_predarboux_virasoro_fits.csv")
    if g2 is not None:
        t = g2[g2.spectral_regulator_class == "tapered_log_window"].set_index(
            "architecture")
        s = t.loc["selected_log_kernel"]
        s_ratio = float(s.ratio_at_max_width)
        sw = rd("gate02_stability_epsilon_sweep.csv")
        if sw is not None:
            ref = sw[sw.direction_index == 0]
            lo, hi = (float(ref.residual_over_pairing_plus_square.min()),
                      float(ref.residual_over_pairing_plus_square.max()))
            add("g2_pbw_stability_theorem_epsilon_sweep", "gate02",
                "||Delta R_Vir|| / (||E|| + ||E||^2), five decades of EPSILON",
                "accepted",
                f"Reference perturbation direction: {lo:.4f}-{hi:.4f}, i.e. the "
                f"quoted 1.65-1.82 plateau. Over 12 independent perturbation "
                f"directions the band is "
                f"{sw.residual_over_pairing_plus_square.min():.4f}-"
                f"{sw.residual_over_pairing_plus_square.max():.4f}, and the "
                f"span across five decades is "
                f"{float(sw.worst_span_over_five_decades.iloc[0]):.3f}x in EVERY "
                f"direction. Measured in a FIXED PBW basis that includes "
                f"alpha_0 and after subtracting the Omega-independent "
                f"finite-mode edge remainder; the excess residual vanishes at "
                f"the canonical pairing (1e-12). Preserve the edge-remainder "
                f"formulation: the unsubtracted ratio diverges as eps -> 0 "
                f"purely from the truncation floor and would misattribute a "
                f"known cutoff artifact to the architecture.",
                value=lo, estimator_class="exact_symbolic",
                script="scripts/run_gate02_pbw.py",
                output_artifact="gate02_stability_epsilon_sweep.csv")
        add("g2_pbw_stability_theorem_width_sweep", "gate02",
            "||Delta R_Vir|| / (||E|| + ||E||^2), five decades of WIDTH",
            "accepted_with_reformulation",
            f"On the architecture grid the same ratio runs "
            f"0.206-1.311 (per-width mean 0.368 -> "
            f"{s_ratio:.3f}), rising toward the epsilon-sweep plateau rather "
            f"than sitting on it. This is a DIFFERENT sweep from the epsilon "
            f"sweep and must not be conflated with it (rule 11): there epsilon "
            f"is dialled directly at fixed mode cutoff, here the width spans "
            f"five decades and epsilon follows. Both confirm boundedness; only "
            f"the epsilon sweep exhibits the plateau.",
            value=float(s_ratio), estimator_class="deterministic_finite_feature",
            script="scripts/run_gate02_pbw.py",
            output_artifact="gate02_predarboux_virasoro_fits.csv")
        add("g2_selected_b_omega", "gate02", "b_Omega exponent (selected, tapered)",
            "accepted",
            f"{s.b_omega_exponent:.3f} +/- {s.b_omega_stderr:.3f}; the null -1/2 "
            f"is NOT rejected (p = {s.b_omega_p_null_minus_half:.2f}).",
            value=float(s.b_omega_exponent), uncertainty=float(s.b_omega_stderr),
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate02_pbw.py",
            output_artifact="gate02_predarboux_virasoro_fits.csv")
        add("g2_bounded_ratio_is_not_sufficient", "gate02",
            "bounded ratio as an architecture test", "fails_current_claim",
            "Every audited control also has a bounded ratio, including RBF "
            "(b_Omega = -0.0003 +/- 0.0005, null rejected p = 2e-9). Bounded "
            "ratio is necessary but NOT sufficient for architecture selection.",
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate02_pbw.py",
            output_artifact="gate02_predarboux_virasoro_fits.csv")

    # ---------------- Gate 3 -------------------------------------------
    prov = rd("tt_estimator_provenance.csv")
    ext = rd("sampled_tt_direct_extension_summary.csv")
    wx = rd("gate03_fit_weighting_crosswalk.csv")
    rc = rd("gate03_regulator_convergence.csv")
    if prov is not None:
        r = prov.iloc[0]
        add("g3_headline_is_wick_only", "gate03",
            "c_sampled = 1.016 +/- 0.011 as an INDEPENDENT determination",
            "fails_current_claim",
            f"Resolves uniquely to estimator_variant=wick_factorized, "
            f"tapered_log_window, N={int(r.n_features)} (the SMALLER of "
            f"{{250,750}}), {int(r.n_blocks)} blocks, M={int(r.n_networks)}. The "
            f"Wick estimator presupposes the Gaussian factorization an "
            f"independent <TT> determination would test. Its agreement with "
            f"c = 1 is a low-statistics artifact.",
            value=float(r.value), uncertainty=float(r.uncertainty),
            ci=(float(r.ci_low), float(r.ci_high)),
            estimator_class="sampled_finite_network",
            script="scripts/run_sampled_tt.py",
            output_artifact="tt_estimator_provenance.csv")
    if wx is not None:
        uni = float(wx[wx.weighting == "uniform"].analytic_c_exp.iloc[0])
        inv = float(wx[wx.weighting == "inverse_variance"].analytic_c_exp.iloc[0])
        add("g3_fit_weighting_crosswalk", "gate03",
            "analytic c_exp under the two fit weightings",
            "accepted_with_reformulation",
            f"`c_exp` is written both by analytic_closed_kernel_reference rows "
            f"(zero stderrs -> UNIFORM weights, {uni:.6f}) and by sampled rows "
            f"(INVERSE-VARIANCE weights, {inv:.6f}). One curve, two summaries, "
            f"differing by {inv - uni:+.6f}. Any sampled-vs-analytic comparison "
            f"must use the matched-weight value.",
            value=inv, estimator_class="analytic_closed_kernel",
            script="scripts/run_gate03_tt.py",
            output_artifact="gate03_fit_weighting_crosswalk.csv")
    if ext is not None:
        d = ext[ext.estimator_variant == "direct_TT"].iloc[0]
        w = ext[ext.estimator_variant == "wick_factorized"].iloc[0]
        add("g3_direct_finite_regulator_tt_normalization", "gate03",
            "direct finite-regulator TT normalization", "accepted",
            f"M = {int(d.n_networks_per_block)} per block x "
            f"{int(d.n_blocks)} blocks, assumption-free (no analytic input). "
            f"This is NOT 'the central charge': it is the direct finite-"
            f"regulator TT normalization, to be compared with the matched "
            f"analytic regulated target 1.0359, not with 1.",
            value=float(d.c), uncertainty=float(d.c_se),
            ci=(float(d.c_ci_low), float(d.c_ci_high)),
            estimator_class="sampled_finite_network",
            script="scripts/run_gate03_direct_extension.py",
            output_artifact="sampled_tt_direct_extension_summary.csv")
        add("g3_wick_finite_regulator_tt_normalization", "gate03",
            "Wick finite-regulator TT normalization",
            "accepted_with_reformulation",
            f"At M = {int(w.n_networks_per_block)} the Wick estimator gives "
            f"{w.c:.4f} +/- {w.c_se:.4f}, which is {abs(w.z_vs_c1):.1f} sigma "
            f"from 1 but 1.2 sigma from its matched-weight analytic target "
            f"1.0359. Do not publish this value alone as 'the central charge'.",
            value=float(w.c), uncertainty=float(w.c_se),
            estimator_class="sampled_finite_network",
            script="scripts/run_gate03_direct_extension.py",
            output_artifact="sampled_tt_direct_extension_summary.csv")
    if rc is not None:
        last = rc.sort_values("lambda_uv").iloc[-1]
        add("g3_analytic_target_tends_to_one", "gate03",
            "max |c(r) - 1| as the regulator is removed", "accepted",
            "Uniform in r, hence weighting-independent: 5.77e-2, 2.36e-3, "
            "8.7e-5, ... one extra decade of cutoff already brings the target "
            "to 1.0004. The production offset is a narrow-window artifact.",
            value=float(last.max_abs_deviation_from_one),
            estimator_class="analytic_closed_kernel",
            script="scripts/run_gate03_tt.py",
            output_artifact="gate03_regulator_convergence.csv")
    ident = rd("gate03_cumulant_identity.csv")
    if ident is not None:
        hi = ident[ident.n_networks == ident.n_networks.max()]
        add("g3_direct_wick_cumulant_identity", "gate03",
            "G_TT^dir - (G_TT^Wick + kappa_JJJJ/4)", "accepted",
            "Verified at a benign cutoff where the direct estimator is well "
            "conditioned; the residual falls 2.88x for M x4 (M^-1/2 predicts "
            "2.00x). At the production cutoff every term is noise dominated, so "
            "the identity is unverifiable there, not violated.",
            value=float(hi.identity_rel_error.mean()),
            estimator_class="sampled_finite_network",
            script="scripts/run_gate03_tt.py",
            output_artifact="gate03_cumulant_identity.csv")
    q = rd("gate03_q_ratio_test.csv")
    if q is not None:
        t = q[q.estimator_variant == "direct_TT"]
        if len(t) and t.iloc[0].get("covariance_status") == "ok":
            r = t.iloc[0]
            add("g3_q_ratio_equals_one", "gate03",
                "q_g = c_dir,g / c_reg,g^analytic (GLS common ratio)",
                "accepted" if bool(r.gls_consistent_with_one)
                else "fails_current_claim",
                f"Estimator-independent: formed separation by separation, so it "
                f"carries no fit weighting. Covariance-aware over "
                f"{int(r.n_blocks)} independent blocks x "
                f"{int(r.n_separations)} correlated separations; Hotelling "
                f"T^2 p = {r.hotelling_p:.3f}, GLS p vs 1 = "
                f"{r.gls_p_vs_one:.3f}. The common-ratio goodness of fit is "
                f"marginal (chi2 = {r.gls_goodness_chi2:.2f} on "
                f"{int(r.gls_goodness_dof)} dof, p = {r.gls_goodness_p:.3f}), "
                f"driven entirely by the largest separation where the direct "
                f"estimator is variance limited; the joint Hotelling test does "
                f"not reject.",
                value=float(r.gls_common_ratio),
                uncertainty=float(r.gls_common_ratio_stderr),
                ci=(float(r.gls_ci_low), float(r.gls_ci_high)),
                estimator_class="sampled_finite_network",
                script="scripts/run_gate03_qratio.py",
                output_artifact="gate03_q_ratio_blocks.csv")
        w = q[q.estimator_variant == "wick_factorized"]
        if len(w) and w.iloc[0].get("covariance_status") == "ok":
            r = w.iloc[0]
            add("g3_q_ratio_wick", "gate03",
                "q_g = c_Wick,g / c_reg,g^analytic (GLS common ratio)",
                "accepted",
                f"The Wick estimator reproduces the analytic regulated target "
                f"at every separation to under 1% "
                f"(q = {r.q_mean}); GLS "
                f"{r.gls_common_ratio:.5f} +/- {r.gls_common_ratio_stderr:.5f}, "
                f"p vs 1 = {r.gls_p_vs_one:.3f}, Hotelling p = "
                f"{r.hotelling_p:.3f}. With the fit weighting removed there is "
                f"no anomaly: the 23-sigma gap against c = 1 was the weighting "
                f"mismatch plus the finite regulator, not a defect.",
                value=float(r.gls_common_ratio),
                uncertainty=float(r.gls_common_ratio_stderr),
                ci=(float(r.gls_ci_low), float(r.gls_ci_high)),
                estimator_class="sampled_finite_network",
                script="scripts/run_gate03_qratio.py",
                output_artifact="gate03_q_ratio_test.csv")
    add("g3_sharp_annulus_n750", "gate03",
        "sharp-annulus Wick central charge at N=750", "fails_current_claim",
        "1.381 +/- 0.116, 3.28 sigma above 1 and incompatible with it. The "
        "frozen ledger labels this `failed_gate`, which the hardening schema "
        "does not allow; it is remapped here. Present the sharp-cutoff failure "
        "as the discovered regulator theorem it is.",
        value=1.381033, uncertainty=0.116016,
        estimator_class="sampled_finite_network",
        script="scripts/run_sampled_tt.py",
        output_artifact="sampled_tt_estimator_comparison.csv")

    # ---------------- Gate 4 -------------------------------------------
    cw = rd("ward_metric_crosswalk.csv")
    mc = rd("sampled_ward_monte_carlo_scaling.csv")
    bl = rd("ward_paired_block_level.csv")
    if cw is not None:
        qv = cw[cw.kind == "manuscript_quoted_value"]
        add("g4_ward_numbers_are_absolute", "gate04",
            "the four quoted Ward numbers", "accepted_with_reformulation",
            f"All four reproduce exactly as MAXIMUM ABSOLUTE stress-Ward errors "
            f"of the deterministic_finite_feature campaign, not normalized ones. "
            f"The normalized value at the headline setting is 0.919, not 0.129 "
            f"(7.1x); every other normalization lands in 0.68-3.33. Also: "
            f"'largest N and M' is undefined -- that file has no n_networks "
            f"column. Relabel, and drop M.",
            value=float(qv.iloc[0].actual_value),
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate04_ward.py",
            output_artifact="ward_metric_crosswalk.csv")
        add("g4_shared_label_collision", "gate04",
            "ward_stress_max_absolute_error across two campaigns",
            "accepted_with_reformulation",
            "Published by the WP-A controls grid (2 vertices at +/-0.9, 3 "
            "probes, N to 1e6) and by the WP-C Ward campaign (3 configurations, "
            "4 probes, N to 2e5). Different geometries and probe counts: the "
            "values are not comparable and require the crosswalk (rule 11).",
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate04_ward.py",
            output_artifact="ward_metric_crosswalk.csv")
    if mc is not None:
        f = mc[mc.m_scaling_exponent.notna()]
        add("g4_sampled_ward_is_monte_carlo_noise", "gate04",
            "sampled Ward normalized error vs M", "diagnostic_only",
            f"E ~ M^({f.m_scaling_exponent.mean():.3f} +/- "
            f"{f.m_scaling_exponent.std(ddof=1):.3f}) over {len(f)} cells, flat "
            f"in N, R^2 0.94-0.9999. Reaching E < 1 would need M ~ 1e9-1e10 "
            f"against the 3.2e4 run. It measures Monte Carlo convergence, not "
            f"architecture restoration, and cannot carry architecture evidence. "
            f"Preserve it as validation that explicit network sampling converges "
            f"to the finite-feature expectation.",
            value=float(f.m_scaling_exponent.mean()),
            uncertainty=float(f.m_scaling_exponent.std(ddof=1)),
            estimator_class="sampled_finite_network",
            script="scripts/run_gate04_ward.py",
            output_artifact="sampled_ward_monte_carlo_scaling.csv")
    if bl is not None:
        sel = bl[(bl.architecture == "rbf_gaussian_prior")
                 & (bl.n_features == 200000)
                 & (bl.spectral_regulator_class == "tapered_log_window")]
        if len(sel):
            r = sel.iloc[0]
            add("g4_rbf_vs_selected_block_level", "gate04",
                "paired Ward ratio, RBF vs selected, N=2e5 (tapered)",
                "accepted",
                f"Geometric mean ratio {r.ratio_geometric_mean:.2f} "
                f"[{r.ratio_ci_low:.2f}, {r.ratio_ci_high:.2f}], p = "
                f"{r.p_value:.2e}, aggregated over {int(r.n_independent_blocks)} "
                f"INDEPENDENT seed blocks. The {int(r.n_raw_pairs)} raw pairs "
                f"are configurations x seeds and are not independent trials; a "
                f"binomial on them would overstate the evidence.",
                value=float(r.ratio_geometric_mean),
                ci=(float(r.ratio_ci_low), float(r.ratio_ci_high)),
                estimator_class="deterministic_finite_feature",
                script="scripts/run_gate04_ward.py",
                output_artifact="ward_paired_block_level.csv")
    add("g4_ward_excludes_local_exponent_direction", "gate04",
        "Ward-based exclusion of local_exponent_direction",
        "fails_current_claim",
        "It converges with exponent -0.911, at least as steep as the selected "
        "architecture's -0.783, and is only 2.65x behind in amplitude. The Ward "
        "identities do not exclude it; intrinsic physical admissibility does "
        "(lambda_min = -0.644 < 0).",
        estimator_class="deterministic_finite_feature",
        script="scripts/run_gate04_ward.py",
        output_artifact="ward_paired_ratio.csv")

    # ---------------- Gate 6 -------------------------------------------
    g6 = rd("gate06_kernel_normalization_checks.csv")
    if g6 is not None:
        graded = g6[g6.check_kind != "diagnostic"]
        add("g6_normalization_conventions", "gate06",
            "kernel normalization and convention suite", "accepted",
            f"{int(graded.passes.sum())}/{len(graded)} graded checks pass, "
            f"including two controls that MUST fail (a spin-zero radial moment "
            f"and a single contour ordering). Pins sigma_phi^2 = 4 L_w = 2Z/pi, "
            f"K(0) = sigma_phi^2/2, dK/dlog r = -2, the spin-two J_2 current "
            f"channel, and the ordered pairing of the ideal kernel (1.0e-15).",
            value=float(graded.passes.sum()),
            estimator_class="analytic_closed_kernel",
            script="scripts/run_gate06_normalization.py",
            output_artifact="gate06_kernel_normalization_checks.csv",
            units="checks")
        add("g6_pointwise_ideal_limit", "gate06",
            "pointwise convergence of K_JJ^Lambda to 1/(z-w)^2",
            "fails_current_claim",
            "It does not converge: the UV tail oscillates with envelope "
            "sqrt(Lambda_UV) sqrt(2/pi r)/(2r), confirmed to 0.6-1.5%. The ideal "
            "limit is meaningful only after ordered contour pairing, where it is "
            "exact. No gate uses a pointwise kernel comparison.",
            estimator_class="analytic_closed_kernel",
            script="scripts/run_gate06_normalization.py",
            output_artifact="gate06_kernel_normalization_checks.csv")

    # ---------------- Gate 7 -------------------------------------------
    g7 = rd("ns_super_virasoro_fits.csv")
    alg = rd("ns_super_virasoro_seed_rows.csv")
    if g7 is not None and alg is not None:
        s = g7[g7.architecture == "selected_log_kernel"]
        big = alg[(alg.architecture == "selected_log_kernel")
                  & (alg.n_features == alg.n_features.max())]
        add("g7_ns_independent_architecture", "gate07",
            "NS sector has an architecture with its own width and randomness",
            "accepted",
            "Explicit randomized Cauchy factorization: importance-sampled NS "
            "modes, width = number of drawn modes, randomness = the mode draw. "
            "The native derived construction is RETAINED as primary and the "
            "Cauchy construction serves as an independent analytic/numerical "
            "cross-check; the closed-form and torus-quadrature paths agree to "
            "5.6e-16.", estimator_class="deterministic_finite_feature",
            script="scripts/run_gate07_ns.py",
            output_artifact="ns_pairing_seed_rows.csv")
        add("g7_ns_convergence", "gate07",
            "NS convergence exponents (6 fits, both constructions)", "accepted",
            f"All six consistent with -1/2; the null is not rejected "
            f"(p >= {s.null_half_p_value.min():.2f}).",
            value=float(s.exponent.mean()),
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate07_ns.py",
            output_artifact="ns_super_virasoro_fits.csv")
        for cons, g in big.groupby("ns_construction"):
            add(f"g7_central_charge_{cons}", "gate07",
                f"NS central charge ({cons})", "accepted",
                f"Target 3/2 at N = {int(big.n_features.max()):.0g}.",
                value=float(g.raw_central_charge_fit.mean()),
                estimator_class="deterministic_finite_feature",
                script="scripts/run_gate07_ns.py",
                output_artifact="ns_super_virasoro_seed_rows.csv")
        add("g7_proposal_variance_condition", "gate07",
            "finite-variance condition q > (r_in/r_out)^2", "accepted",
            "Demonstrated by the EXACT estimator variance against truncation: "
            "flat above the threshold (1.9694 at every n_max for q = 0.75), "
            "divergent below (7.3e19 -> inf at q = 0.10). A sample standard "
            "deviation cannot show this -- with finitely many draws the rare "
            "heavy-weight modes are never drawn.",
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate07_ns.py",
            output_artifact="ns_kernel_seed_rows.csv")
        add("g7_super_virasoro_residual_is_bosonic_dominated", "gate07",
            "[L,G] residual as a test of the fermionic sector",
            "accepted_with_reformulation",
            "Native and Cauchy differ 10x in CAR relative error but only "
            "0.26-0.41% in [L,G]: the residual is set almost entirely by the "
            "bosonic Omega. Their agreement on [L,G] and {G,G} should not be "
            "over-read; the CAR error is the fermion-specific observable.",
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate07_ns.py",
            output_artifact="ns_super_virasoro_seed_rows.csv")
        add("g7_car_leakage", "gate07",
            "CAR leakage for the randomized Cauchy construction",
            "diagnostic_only",
            "Identically zero by construction (the closed form is exactly "
            "antidiagonal), against 0.0099-0.859 for the native construction. "
            "It is not evidence of better behaviour. Predicted in "
            "NS_IMPLEMENTATION_MAP before the run.",
            value=0.0, estimator_class="deterministic_finite_feature",
            script="scripts/run_gate07_ns.py",
            output_artifact="ns_pairing_seed_rows.csv")

    # ---------------- Gate 8 -------------------------------------------
    syn = rd("architecture_synthesis.csv")
    jf = rd("convergence_joint_fit.csv")
    ti = rd("intrinsic_stationarity_seed_rows.csv")
    if syn is not None and ti is not None:
        sel = syn[syn.is_selected].iloc[0]
        pc = syn[syn.architecture == "phase_correlated_features"].iloc[0]
        indep = ti[ti.phase_pairing_is_trivial]
        add("g8_selected_uniquely_isolated", "gate08",
            "joint structural suite isolates the selected architecture",
            "accepted",
            f"Five axes: translation invariance, rotational invariance, pairing "
            f"fidelity, raw Virasoro fidelity, intrinsic physical "
            f"admissibility. The selected architecture fails none; every one of "
            f"the seven controls fails the axis it was designed to violate; "
            f"exactly one architecture has a clean sheet.",
            value=1.0, estimator_class="deterministic_finite_feature",
            script="scripts/run_gate08_synthesis.py",
            output_artifact="architecture_synthesis.csv", units="architectures")
        add("g8_stationarity_exact_for_independent_phases", "gate08",
            "scalar translation defect, independent-phase architectures",
            "accepted",
            f"Maximum {indep.scalar_translation_defect.max():.2e} over 7 "
            f"architectures x 5 widths x 4 seeds: conditional phase averaging "
            f"makes stationarity exact up to floating point. Computed on the "
            f"TRUE two-argument kernel; the one-argument helpers are "
            f"translation invariant by construction and would return zero for "
            f"any architecture.",
            value=float(indep.scalar_translation_defect.max()),
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate08_synthesis.py",
            output_artifact="intrinsic_stationarity_seed_rows.csv")
        add("g8_phase_correlated_excluded", "gate08",
            "phase_correlated_features translation defect", "accepted",
            f"delta_TI = {pc.scalar_translation_defect:.4f}, flat across a 100x "
            f"width range -- structural, not statistical, and fifteen orders of "
            f"magnitude above the independent-phase floor.",
            value=float(pc.scalar_translation_defect),
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate08_synthesis.py",
            output_artifact="intrinsic_stationarity_seed_rows.csv")
        add("g8_phase_correlated_excluded_by_preexisting_axes", "gate08",
            "exclusion of phase_correlated_features before this gate",
            "fails_current_claim",
            "It passes physical admissibility (lambda_min 0.953 > selected's "
            "0.918), pairing fidelity (delta_iso exponent -0.591, steeper than "
            "selected's -0.488), Ward behaviour (0.0099 at N=1e6, smaller than "
            "selected's 0.0128) and rotational invariance. No pre-existing axis "
            "excluded it; only the stationarity audit does.",
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate08_synthesis.py",
            output_artifact="architecture_synthesis.csv")
        add("g8_anisotropic_excluded_by_ward", "gate08",
            "Ward-based exclusion of anisotropic_angular_prior",
            "fails_current_claim",
            "Its Ward exponent -0.270 is converging, so the Ward proxy does not "
            "exclude it. The isotropy defect does: pinned at 0.0618 with "
            "exponent -0.005 across a 100x width range.",
            estimator_class="deterministic_finite_feature",
            script="scripts/run_gate08_synthesis.py",
            output_artifact="architecture_synthesis.csv")
    if jf is not None:
        p = jf[jf.quantity == "POOLED_EXPONENT"]
        if len(p):
            r = p.iloc[0]
            add("g8_pooled_convergence_exponent", "gate08",
                "pooled N^(-1/2) exponent", "accepted",
                f"Inverse-variance pool of the {int(r.n_pooled)} rows where an "
                f"N^-1/2 law is actually predicted (Gate 1 delta_iso, Gate 2 "
                f"b_Omega, Gate 7 x3 under both NS constructions). Cochran "
                f"Q = {r.cochran_Q:.2f} on {int(r.cochran_dof)} dof "
                f"(p = {r.cochran_p:.3f}), I^2 = {r.I_squared_percent:.0f}%; "
                f"null -1/2 not rejected (p = {r.null_half_p_value:.2f}). Ward, "
                f"stationarity and isotropy exponents are reported but never "
                f"pooled -- no N^-1/2 law is predicted for them.",
                value=float(r.exponent), uncertainty=float(r.exponent_stderr),
                estimator_class="deterministic_finite_feature",
                script="scripts/run_gate08_synthesis.py",
                output_artifact="convergence_joint_fit.csv")

    write_rows(outdir / "hardening_claim_ledger.csv", rows, required=[
        "claim_id", "gate", "metric", "status", "notes", "source_script",
        "output_artifact"],
        sort_by=["gate", "claim_id"])
    d = pd.DataFrame(rows)
    print(f"  {len(rows)} claims")
    for st, g in d.groupby("status"):
        print(f"    {st:<30} {len(g):3d}   ({', '.join(sorted(g.gate.unique()))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
