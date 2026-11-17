#!/usr/bin/env python3
"""Gates 3+5: TT estimator provenance audit and direct/Wick separation.

This is a PROVENANCE AUDIT first. The direct campaign is extended only if the
existing direct result cannot support the manuscript wording.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro.cli import base_parser, setup            # noqa: E402
from nnft_virasoro.config import rng_for                    # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows  # noqa: E402
from scipy.special import jv                               # noqa: E402

from nnft_virasoro import plotstyle as PS                  # noqa: E402
from nnft_virasoro import stress_tensor as ST              # noqa: E402
from nnft_virasoro.architectures import _tukey_log_window  # noqa: E402
from nnft_virasoro.ordered_pairing import _gauss_legendre_panels  # noqa: E402
from nnft_virasoro.sampling import (block_statistics, bootstrap_ci,  # noqa: E402
                                    iter_batches)

SCRIPT = "scripts/run_gate03_tt.py"


def cumulant_identity_check(cfg, sec, stamper):
    r"""Regression: G_TT^dir = G_TT^Wick + (1/4) kappa_JJJJ(z,z,w,w).

    The two sides are computed by *different* expressions from the same draws:
    the left uses the per-realization normal-ordering subtraction c_N, the right
    uses ensemble second moments. Agreement therefore tests the subtraction, not
    just an algebraic rearrangement.
    """
    ic = sec["identity_check"]
    lir, luv = ic["lambda_ir"], ic["lambda_uv"]
    R = float(ic["circle_radius"])
    P = int(ic["n_circle_points"])
    z = R * np.exp(2j * np.pi * np.arange(P) / P)
    rows = []
    for M in ic["n_networks"]:
      for seed_name in ic["seeds"]:
        rng = rng_for(cfg.section("named_seeds"), seed_name, "gate03id", M)
        acc = {k: 0j for k in ("ab", "a2b2", "a2", "b2", "AB", "c", "c2")}
        cnt = 0
        for batch in iter_batches(rng, n_features=ic["n_features"],
                                  n_realizations=M, points=z,
                                  lambda_ir=lir, lambda_uv=luv,
                                  params={"regulator_taper": ic["taper"]},
                                  batch_size=ic["batch_size"]):
            d = batch.dphi
            c = batch.self_contraction[:, None]
            a, b = d[:, 0:1], d[:, P // 2:P // 2 + 1]     # antipodal pair
            acc["ab"] += complex(np.sum(a * b))
            acc["a2b2"] += complex(np.sum(a**2 * b**2))
            acc["a2"] += complex(np.sum(a**2))
            acc["b2"] += complex(np.sum(b**2))
            acc["AB"] += complex(np.sum((a**2 - c) * (b**2 - c)))
            acc["c"] += complex(np.sum(c))
            acc["c2"] += complex(np.sum(c**2))
            cnt += batch.n_realizations
        m = {k: v / cnt for k, v in acc.items()}
        K_hat = -m["ab"]                                   # estimates K_JJ(z,w)
        g_dir = 0.25 * m["AB"]
        g_wick = 0.5 * K_hat**2
        kappa4 = m["a2b2"] - m["a2"] * m["b2"] - 2.0 * m["ab"] ** 2
        # With a PER-REALIZATION subtraction c_N (rather than the ensemble
        # constant E[c]) the identity carries an exact extra term:
        #   E[(a^2-c)(b^2-c)] = E[a^2b^2] - E[c^2]   since E_gamma[a^2|k] = c,
        # giving  G_dir - (G_wick + kappa4/4) = -Var(c_N)/4.
        var_c = m["c2"] - m["c"] ** 2
        lhs = g_dir
        rhs = g_wick + 0.25 * kappa4 - 0.25 * var_c
        rows.append(stamper.stamp(dict(
            gate="gate03", check="direct_wick_cumulant_identity",
            seed_name=seed_name, n_features=ic["n_features"], n_networks=cnt,
            lambda_ir=lir, lambda_uv=luv, regulator_taper=ic["taper"],
            separation=float(abs(z[0] - z[P // 2])),
            K_JJ_hat=abs(K_hat), G_TT_direct=abs(g_dir), G_TT_wick=abs(g_wick),
            kappa_JJJJ=abs(kappa4), var_c_N=abs(var_c),
            identity_lhs=abs(lhs), identity_rhs=abs(rhs),
            identity_abs_error=abs(lhs - rhs),
            identity_rel_error=abs(lhs - rhs) / max(abs(lhs), 1e-300)),
            estimator_class="sampled_finite_network", seed=seed_name))
        print(f"    identity M={M} {seed_name}: rel err "
              f"{rows[-1]['identity_rel_error']:.3e}")
    # the identity is exact; the residual must fall like M^-1/2
    import pandas as _pd
    _d = _pd.DataFrame(rows).groupby("n_networks")["identity_rel_error"].mean()
    if len(_d) >= 2:
        lo, hi = _d.index.min(), _d.index.max()
        shrink = _d[lo] / _d[hi]
        for r in rows:
            r["identity_residual_shrinks_with_M"] = bool(shrink > 1.3)
            r["identity_shrink_factor"] = float(shrink)
            r["identity_expected_shrink"] = float(np.sqrt(hi / lo))
        print(f"    residual shrink {shrink:.2f}x for M x{hi/lo:.0f} "
              f"(M^-1/2 predicts {np.sqrt(hi/lo):.2f}x)")
    return rows


def analytic_c_curve(seps, lir, luv, taper):
    """``c(r) = (r^2 K_JJ^Lambda(r))^2`` at the campaign geometry."""
    out = []
    for r in np.asarray(seps, dtype=float):
        pts, wts = _gauss_legendre_panels(lir, luv, np.pi / (8 * max(r, 1e-12)),
                                          nodes=8)
        w = (_tukey_log_window(np.log(pts), np.log(lir), np.log(luv), taper)
             if taper > 0 else 1.0)
        out.append((r * r * float(np.sum(wts * pts * w * jv(2, pts * r)) / 2.0))**2)
    return np.asarray(out)


def weighting_crosswalk(cfg, sec, stamper):
    """Rule-11 crosswalk: one c(r) curve, two fit weightings, two `c_exp`.

    The frozen `analytic_closed_kernel_reference` rows are fitted with UNIFORM
    weights (their per-separation stderrs are identically zero, so
    ``fit_central_charge`` falls back to ``w = 1``), while every sampled row is
    fitted with INVERSE-VARIANCE weights. Comparing them directly is comparing
    two different functionals of the same curve.
    """
    ex = sec["direct_extension"]
    wc = sec["weighting_crosswalk"]
    lir, luv = float(ex["lambda_ir"]), float(ex["lambda_uv"])
    taper = float(ex["taper"])
    geo = ST.circle_geometry(int(ex["n_circle_points"]), float(ex["circle_radius"]))
    seps = geo["separations"]
    lo, hi = [float(x) for x in ex["fit_window"]]
    sel = (seps >= lo - 1e-12) & (seps <= hi + 1e-12)
    ana = analytic_c_curve(seps, lir, luv, taper)

    rng = rng_for(cfg.section("named_seeds"), wc["seeds"][0], "gate03weights")
    res = ST.sampled_tt(rng, n_features=int(wc["n_features"]),
                        n_networks=int(wc["n_networks"]), geometry=geo,
                        lambda_ir=lir, lambda_uv=luv,
                        params={"regulator_taper": taper},
                        batch_size=int(wc["batch_size"]))
    err = np.asarray(res["c_wick_stderr"])[sel]
    w = 1.0 / np.where(err > 0, err, np.nan) ** 2
    a = ana[sel]
    rows = []
    for scheme, weights in (("uniform", np.ones_like(a)),
                            ("inverse_variance", w)):
        rows.append(stamper.stamp(dict(
            gate="gate03", check="fit_weighting_crosswalk",
            shared_column="c_exp", weighting=scheme,
            used_by=("analytic_closed_kernel_reference" if scheme == "uniform"
                     else "wick_factorized;direct_TT"),
            analytic_c_exp=float(np.sum(weights * a) / np.sum(weights)),
            n_fit_points=int(sel.sum()),
            fit_window_low=lo, fit_window_high=hi,
            c_of_r_min=float(a.min()), c_of_r_max=float(a.max()),
            c_of_r_spread=float(a.std(ddof=1)),
            max_weight_fraction=float((weights / weights.sum()).max()),
            lambda_ir=lir, lambda_uv=luv, regulator_taper=taper,
            n_features=int(wc["n_features"]),
            n_networks=int(wc["n_networks"])),
            estimator_class="analytic_closed_kernel", seed=wc["seeds"][0]))
    uni, inv = rows[0]["analytic_c_exp"], rows[1]["analytic_c_exp"]
    for r in rows:
        r["analytic_c_exp_uniform"] = uni
        r["analytic_c_exp_inverse_variance"] = inv
        r["weighting_shift"] = inv - uni
    print(f"    analytic target: uniform {uni:.6f} vs inverse-variance {inv:.6f} "
          f"(shift {inv - uni:+.6f})")
    return rows


def regulator_convergence(sec, stamper):
    """Does the window-summarized analytic target tend to 1 as Lambda grows?"""
    ex = sec["direct_extension"]
    rc = sec["regulator_convergence"]
    geo = ST.circle_geometry(int(ex["n_circle_points"]), float(ex["circle_radius"]))
    seps = geo["separations"]
    lo, hi = [float(x) for x in ex["fit_window"]]
    sel = (seps >= lo - 1e-12) & (seps <= hi + 1e-12)
    rows = []
    for lir, luv in rc["windows"]:
        a = analytic_c_curve(seps, float(lir), float(luv), float(rc["taper"]))[sel]
        rows.append(stamper.stamp(dict(
            gate="gate03", check="analytic_regulator_convergence",
            lambda_ir=float(lir), lambda_uv=float(luv),
            regulator_taper=float(rc["taper"]),
            c_exp_uniform=float(a.mean()),
            max_abs_deviation_from_one=float(np.abs(a - 1.0).max()),
            c_of_r_min=float(a.min()), c_of_r_max=float(a.max()),
            n_fit_points=int(sel.sum()),
            fit_window_low=lo, fit_window_high=hi),
            estimator_class="analytic_closed_kernel", seed=None))
    d = [r["max_abs_deviation_from_one"] for r in rows]
    for r in rows:
        r["converges_to_one"] = bool(d[-1] < 1e-4 and d[-1] < d[0])
    print(f"    analytic target -> 1: max|c-1| {d[0]:.5f} -> {d[-1]:.2e} "
          f"across {len(rows)} windows")
    return rows


def estimator_comparison_figure(cmp_rows, figdir):
    """Wick vs direct central charge across regulator class and width.

    Both estimators are plotted on the SAME axes with their own error bars so
    the reader can see what the headline selection costs: the Wick estimator is
    precise but assumes the Gaussian factorization it is being used to check,
    while the direct estimator assumes nothing and is 37-90x wider.
    """
    d = pd.DataFrame(cmp_rows).sort_values(["spectral_regulator_class",
                                            "n_features"]).reset_index(drop=True)
    _short = {"sharp_annulus_manuscript": "sharp\nannulus",
              "tapered_log_window": "tapered\nlog window"}
    labels = [f"{_short.get(r.spectral_regulator_class, r.spectral_regulator_class)}\n"
              f"$N={int(r.n_features)}$" for r in d.itertuples()]
    x = np.arange(len(d), dtype=float)
    m = PS.MARKERS["sampled_finite_network"]
    fig, (ax, az) = PS.new_figure(1, 2, figsize=(9.6, 4.3),
                                  gridspec_kw={"width_ratios": [1.0, 1.0]})
    hl = d.query("spectral_regulator_class == 'tapered_log_window' "
                 "and n_features == 250")

    for a, zoom in ((ax, False), (az, True)):
        a.axhline(1.0, color="0.4", lw=1.0, ls="--", zorder=1)
        dx = 0.0 if zoom else 0.09
        a.errorbar(x - dx, d["wick_c"], yerr=2 * d["wick_se"], fmt=m,
                   color=PS.COLORS[0], capsize=3, lw=1.2, ms=5,
                   label="Wick-factorized (assumes Gaussian factorization)")
        if not zoom:      # direct bars are ~50x wider; they swamp the zoom panel
            a.errorbar(x + 0.09, d["direct_c"], yerr=2 * d["direct_se"], fmt=m,
                       color=PS.COLORS[1], capsize=3, lw=1.2, ms=5, mfc="none",
                       label="direct four-field (no analytic input)")
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=7.5)
        a.set_xlim(-0.5, len(d) - 0.5)
        a.set_ylabel("sampled central charge $c$")
    ax.set_ylim(-20, 30)
    ax.annotate("$c=1$", xy=(len(d) - 0.55, 1.0), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=8, color="0.4")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("both estimators, full scale", fontsize=9)

    # zoom: the Wick band, where the regulator/width dependence actually lives
    az.set_ylim(0.90, 1.70)
    az.set_title("Wick estimator only (direct bars are $\\sim$50x wider)",
                 fontsize=9)
    if len(hl):
        i = int(hl.index[0])
        az.annotate("manuscript headline\n$c=1.016\\pm0.011$",
                    xy=(i, float(hl["wick_c"].iloc[0])), xytext=(-2, 40),
                    textcoords="offset points", fontsize=8, ha="center",
                    color=PS.COLORS[0],
                    arrowprops=dict(arrowstyle="->", color=PS.COLORS[0], lw=0.8))
    bad = d[(d["wick_c"] - 2 * d["wick_se"]) > 1.0]
    for i, r in bad.iterrows():
        az.annotate("$>2\\sigma$ above $c=1$", xy=(i, float(r["wick_c"])),
                    xytext=(12, 0), textcoords="offset points", fontsize=8,
                    color=PS.COLORS[1], va="center")
    fig.suptitle("Gate 3: TT central charge by estimator variant "
                 "(error bars $2\\sigma$, 12 blocks, $M=60000$)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return PS.save(fig, "gate03_tt_estimator_comparison.pdf", outdir=figdir)


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    sec = cfg.section("gate03")
    prov_path = outdir / "tt_estimator_provenance.csv"
    if should_skip(prov_path, args.skip_existing):
        print("  [skip] gate03")
        return 0

    seed_rows = pd.read_csv(ROOT / sec["frozen_seed_rows"])
    agg = pd.read_csv(ROOT / sec["frozen_aggregate"])
    hv, hu = float(sec["headline_value"]), float(sec["headline_uncertainty"])

    # ---- locate the headline row exactly -------------------------------
    hit = agg[(agg.c_exp.round(6) == round(hv, 6))
              & (agg.c_exp_stderr.round(6) == round(hu, 6))]
    if len(hit) != 1:
        raise SystemExit(f"headline value {hv} did not resolve uniquely "
                         f"({len(hit)} matches)")
    h = hit.iloc[0]
    src = seed_rows[(seed_rows.regulator_class == h.regulator_class)
                    & (seed_rows.architecture == h.architecture)
                    & (seed_rows.estimator_variant == h.estimator_variant)
                    & (seed_rows.n_features == h.n_features)
                    & (seed_rows.fit_window_index == h.fit_window_index)]
    prov = [stamper.stamp(dict(
        gate="gate03",
        claim_ledger_row="wpB_c_exp_tapered_log_window_wick_factorized_N250",
        manuscript_claim="c_sampled = 1.016 +/- 0.011, 95% CI [0.994, 1.036]",
        source_file="results/jhep_frozen/sampled_tt_central_charge.csv",
        source_rows=int(len(src)),
        source_row_selector=(f"regulator_class={h.regulator_class}; "
                             f"architecture={h.architecture}; "
                             f"estimator_variant={h.estimator_variant}; "
                             f"n_features={int(h.n_features)}; "
                             f"fit_window_index={int(h.fit_window_index)}"),
        script="scripts/run_sampled_tt.py",
        function="stress_tensor.sampled_tt + fit_central_charge",
        estimator="wick_factorized: c = (mean_blocks S)^2, "
                  "S = -d phi(z) d phi(w) (z-w)^2",
        estimator_class="sampled_finite_network",
        is_wick_only=True, is_direct_only=False, is_combined=False,
        normal_ordering_definition="c_N = (L_w/(2N)) sum_j conj(k_j)^2, computed "
                                   "in closed form per realization from that "
                                   "realization's own momenta",
        normal_ordering_class="conditional_phase_averaged",
        self_estimation_bias="none: c_N is an exact conditional expectation "
                             "given {k_j}, not estimated from the field samples",
        n_features=int(h.n_features), n_networks=int(h.n_networks),
        n_blocks=int(h.n_blocks),
        seed_set=";".join(sorted(src.seed_name.dropna().unique())),
        geometry_set=f"{int(src.n_circle_points.iloc[0])} points on a circle of "
                     f"radius {src.circle_radius.iloc[0]}; chord separations",
        fit_window=f"[{h.fit_window_low}, {h.fit_window_high}]",
        uncertainty_method="block SE over independent named seeds; percentile "
                           "bootstrap over blocks (20000 resamples, seed 20260730)",
        value=float(h.c_exp), uncertainty=float(h.c_exp_stderr),
        ci_low=float(h.c_exp_boot_ci_low), ci_high=float(h.c_exp_boot_ci_high),
        is_largest_width=bool(h.n_features == agg.n_features.max()),
        widths_available=";".join(str(int(x)) for x in sorted(agg.n_features.unique()))),
        estimator_class="sampled_finite_network", seed="multi")]
    write_rows(prov_path, prov, required=[
        "gate", "claim_ledger_row", "source_file", "source_rows", "script",
        "function", "estimator", "estimator_class",
        "normal_ordering_definition", "n_features", "n_networks", "n_blocks",
        "seed_set", "geometry_set", "fit_window", "uncertainty_method"],
        sort_by=["claim_ledger_row"])
    print(f"  headline resolves to: {h.estimator_variant}, N={int(h.n_features)}, "
          f"{int(h.n_blocks)} blocks, largest_width={h.n_features == agg.n_features.max()}")

    # ---- split the frozen blocks by estimator, never merged -------------
    for variant, fname in (("wick_factorized", "sampled_tt_wick_blocks.csv"),
                           ("direct_TT", "sampled_tt_direct_blocks.csv")):
        v = seed_rows[seed_rows.estimator_variant == variant].copy()
        rows = [stamper.stamp(dict(
            gate="gate03", estimator_variant=variant,
            spectral_regulator_class=r.regulator_class,
            architecture=r.architecture, seed_name=r.seed_name,
            block_id=r.seed_name, n_features=int(r.n_features),
            n_networks=int(r.n_networks),
            fit_window_index=int(r.fit_window_index),
            fit_window_low=float(r.fit_window_low),
            fit_window_high=float(r.fit_window_high),
            c_exp=float(r.c_exp), chi2_per_dof=float(r.chi2_per_dof),
            propagated_stderr=float(r.propagated_stderr)),
            estimator_class="sampled_finite_network", seed=r.seed_name)
            for _, r in v.iterrows()]
        write_rows(outdir / fname, rows, required=[
            "gate", "estimator_variant", "spectral_regulator_class",
            "architecture", "seed_name", "block_id", "n_features", "c_exp"],
            sort_by=["spectral_regulator_class", "architecture", "n_features",
                     "fit_window_index", "seed_name"])

    # ---- estimator comparison ------------------------------------------
    cmp_rows = []
    sel = seed_rows[(seed_rows.architecture == "selected_log_kernel")
                    & (seed_rows.fit_window_index == 0)
                    & (seed_rows.estimator_variant != "analytic_closed_kernel_reference")]
    for (rc, nf), sub in sel.groupby(["regulator_class", "n_features"]):
        rec = dict(gate="gate03", spectral_regulator_class=rc,
                   n_features=int(nf), architecture="selected_log_kernel")
        for variant in ("wick_factorized", "direct_TT"):
            v = sub[sub.estimator_variant == variant]["c_exp"].to_numpy(float)
            st = block_statistics(v)
            lo, hi = bootstrap_ci(v, seed=20260730)
            tag = "wick" if variant == "wick_factorized" else "direct"
            rec[f"{tag}_c"] = st["mean"]
            rec[f"{tag}_se"] = st["stderr"]
            rec[f"{tag}_ci_low"], rec[f"{tag}_ci_high"] = lo, hi
            rec[f"{tag}_n_blocks"] = st["n_blocks"]
            rec[f"{tag}_z_vs_c1"] = ((st["mean"] - 1.0) / st["stderr"]
                                     if st["stderr"] > 0 else float("nan"))
            rec[f"{tag}_compatible_with_c1"] = bool(
                st["stderr"] > 0 and abs(st["mean"] - 1.0) <= 2 * st["stderr"])
        diff = rec["wick_c"] - rec["direct_c"]
        se = float(np.hypot(rec["wick_se"], rec["direct_se"]))
        rec["wick_minus_direct"] = diff
        rec["wick_minus_direct_se"] = se
        rec["wick_direct_z"] = diff / se if se > 0 else float("nan")
        rec["wick_direct_agree_2sigma"] = bool(se > 0 and abs(diff) <= 2 * se)
        rec["direct_se_over_wick_se"] = (rec["direct_se"] / rec["wick_se"]
                                         if rec["wick_se"] > 0 else float("nan"))
        cmp_rows.append(stamper.stamp(rec,
                                      estimator_class="sampled_finite_network",
                                      seed="multi"))
    write_rows(outdir / "sampled_tt_estimator_comparison.csv", cmp_rows, required=[
        "gate", "spectral_regulator_class", "n_features", "wick_c", "wick_se",
        "direct_c", "direct_se", "wick_direct_z", "wick_direct_agree_2sigma"],
        sort_by=["spectral_regulator_class", "n_features"])

    # ---- direct/Wick cumulant identity regression ----------------------
    print("  running direct-Wick cumulant identity check...")
    idr = cumulant_identity_check(cfg, sec, stamper)
    write_rows(outdir / "gate03_cumulant_identity.csv", idr, required=[
        "gate", "check", "seed_name", "identity_lhs", "identity_rhs",
        "identity_rel_error", "identity_residual_shrinks_with_M"],
        sort_by=["n_networks", "seed_name"])
    # ---- rule-11 fit-weighting crosswalk and the regulator limit -------
    print("  fit-weighting crosswalk...")
    wrows = weighting_crosswalk(cfg, sec, stamper)
    write_rows(outdir / "gate03_fit_weighting_crosswalk.csv", wrows, required=[
        "gate", "check", "shared_column", "weighting", "used_by",
        "analytic_c_exp", "weighting_shift"], sort_by=["weighting"])
    print("  analytic regulator convergence...")
    rrows = regulator_convergence(sec, stamper)
    write_rows(outdir / "gate03_regulator_convergence.csv", rrows, required=[
        "gate", "check", "lambda_uv", "c_exp_uniform",
        "max_abs_deviation_from_one", "converges_to_one"],
        sort_by=["lambda_uv"])

    figpath = estimator_comparison_figure(cmp_rows, args.figure_dir)
    print(f"  wrote provenance, wick/direct blocks, comparison ({len(cmp_rows)}), "
          f"identity ({len(idr)}), figure {figpath.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
