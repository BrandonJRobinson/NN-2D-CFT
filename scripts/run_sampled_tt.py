#!/usr/bin/env python3
"""WP-B: independent sampled neural-ensemble <TT> central charge.

Regenerates c from native code.  The legacy value 0.9958 +- 0.0196 is *not*
copied anywhere; it appears only as a labelled comparison field in the summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnft_virasoro import stress_tensor as ST  # noqa: E402
from nnft_virasoro.cli import base_parser, setup  # noqa: E402
from nnft_virasoro.config import rng_for  # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows  # noqa: E402
from nnft_virasoro.kernels import physical_window  # noqa: E402
from nnft_virasoro.sampling import block_statistics, bootstrap_ci, jackknife  # noqa: E402

SCRIPT = "scripts/run_sampled_tt.py"


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    rows_path = outdir / "sampled_tt_seed_rows.csv"
    agg_path = outdir / "sampled_tt_central_charge.csv"
    if all(should_skip(p, args.skip_existing) for p in (rows_path, agg_path)):
        print("  [skip] WP-B")
        return 0

    sec = cfg.section("wp_b")
    g = cfg.section("global")
    lambda_ir, lambda_uv = sec["regulator"]
    r_min, r_max = physical_window(lambda_ir, lambda_uv,
                                   margin_uv=g["window_margin_uv"],
                                   margin_ir=g["window_margin_ir"])
    geo = ST.circle_geometry(sec["n_circle_points"], sec["circle_radius"])
    seps = geo["separations"]
    windows = [tuple(sec["fit_window"])] + [tuple(w) for w in sec["fit_window_variants"]]
    rclasses = [(str(k), float(v)) for k, v in g["regulator_classes"].items()]

    # analytic_closed_kernel reference at the SAME geometry: c(r) = (r^2 K_JJ)^2
    from nnft_virasoro.kernels import current_kernel_analytic
    from nnft_virasoro.ordered_pairing import _gauss_legendre_panels
    from nnft_virasoro.architectures import _tukey_log_window
    from scipy.special import jv

    def analytic_c(r, taper):
        if taper <= 0:
            kjj = float(np.real(current_kernel_analytic(r + 0j, 0j,
                                                        lambda_ir, lambda_uv)))
        else:
            pts, wts = _gauss_legendre_panels(lambda_ir, lambda_uv,
                                              np.pi / (8 * max(r, 1e-12)), nodes=8)
            w = _tukey_log_window(np.log(pts), np.log(lambda_ir),
                                  np.log(lambda_uv), taper)
            kjj = float(np.sum(wts * pts * w * jv(2, pts * r)) / 2.0)
        return (r * r * kjj) ** 2

    rows = []
    for rclass, taper in rclasses:
        params = {"regulator_taper": taper}
        ana = np.array([analytic_c(float(r), taper) for r in seps])
        for wi, win in enumerate(windows):
            fit = ST.fit_central_charge(seps, ana, np.zeros_like(ana), win)
            rows.append(stamper.stamp(dict(
                regulator_class=rclass, regulator_taper=taper,
                architecture="selected_log_kernel", seed_name="",
                estimator_variant="analytic_closed_kernel_reference",
                lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                r_min_physical=r_min, r_max_physical=r_max,
                n_features=0, n_networks=0,
                fit_window_index=wi, is_reference_window=bool(wi == 0),
                normal_ordering_subtraction=0.0, **fit,
            ), estimator_class="analytic_closed_kernel", seed=None))

        jobs = [("selected_log_kernel", nf, sd)
                for nf in sec["n_features"] for sd in sec["seeds"]]
        jobs += [(a, sec["control_n_features"], sd)
                 for a in sec["controls"] for sd in sec["control_seeds"]]
        for arch, n_features, seed_name in jobs:
            rng = rng_for(cfg.section("named_seeds"), seed_name, "tt", arch,
                          n_features, rclass)
            res = ST.sampled_tt(
                rng, n_features=n_features, n_networks=sec["n_networks"],
                geometry=geo, lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                architecture=arch, params=params, batch_size=sec["batch_size"])
            for variant, vals, errs in (
                    ("wick_factorized", res["c_wick_per_separation"],
                     res["c_wick_stderr"]),
                    ("direct_TT", res["c_direct_per_separation"],
                     res["c_direct_stderr"])):
                for wi, win in enumerate(windows):
                    fit = ST.fit_central_charge(seps, vals, errs, win)
                    rows.append(stamper.stamp(dict(
                        regulator_class=rclass, regulator_taper=taper,
                        architecture=arch, seed_name=seed_name,
                        estimator_variant=variant,
                        lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                        r_min_physical=r_min, r_max_physical=r_max,
                        n_features=res["n_features"], n_networks=res["n_networks"],
                        n_circle_points=geo["n_points"], circle_radius=geo["radius"],
                        fit_window_index=wi, is_reference_window=bool(wi == 0),
                        normal_ordering_subtraction=abs(
                            res["normal_ordering_subtraction"]),
                        **fit,
                    ), estimator_class="sampled_finite_network", seed=seed_name))
            print(f"  WP-B {rclass}/{arch} N={n_features} {seed_name} done "
                  f"({len(rows)} rows)")

    write_rows(rows_path, rows, required=[
        "regulator_class", "architecture", "estimator_variant", "seed_name",
        "n_features", "n_networks", "fit_window_index", "c_exp", "chi2_per_dof",
        "residual_rms"],
        sort_by=["regulator_class", "architecture", "estimator_variant",
                 "n_features", "fit_window_index", "seed_name"])

    import pandas as pd

    df = pd.DataFrame(rows)
    agg = []
    df = df[df["estimator_variant"] != "analytic_closed_kernel_reference"]
    for (rclass, arch, variant, n_features, wi), sub in df.groupby(
            ["regulator_class", "architecture", "estimator_variant", "n_features",
             "fit_window_index"]):
        vals = sub["c_exp"].to_numpy()
        st = block_statistics(vals)
        lo, hi = bootstrap_ci(vals, seed=20260730)
        jk = jackknife(vals)
        agg.append(stamper.stamp(dict(
            regulator_class=rclass, architecture=arch, estimator_variant=variant,
            n_features=int(n_features),
            fit_window_index=int(wi),
            fit_window_low=float(sub["fit_window_low"].iloc[0]),
            fit_window_high=float(sub["fit_window_high"].iloc[0]),
            n_networks=int(sub["n_networks"].iloc[0]),
            c_exp=st["mean"], c_exp_sd=st["sd"], c_exp_stderr=st["stderr"],
            c_exp_boot_ci_low=lo, c_exp_boot_ci_high=hi,
            c_exp_jackknife_stderr=jk["stderr"], n_blocks=st["n_blocks"],
            residual_rms_mean=float(sub["residual_rms"].mean()),
            chi2_per_dof_mean=float(sub["chi2_per_dof"].mean()),
            spread_across_separations_mean=float(
                sub["spread_across_separations"].mean()),
            compatible_with_c1=bool(
                st["stderr"] > 0 and abs(st["mean"] - 1.0) <= 2.0 * st["stderr"]),
            z_vs_c1=float((st["mean"] - 1.0) / st["stderr"]) if st["stderr"] > 0
            else float("nan"),
        ), estimator_class="sampled_finite_network", seed="multi"))
    write_rows(agg_path, agg, required=[
        "regulator_class", "architecture", "estimator_variant", "n_features",
        "fit_window_index", "c_exp",
        "c_exp_stderr", "c_exp_boot_ci_low", "c_exp_boot_ci_high", "n_blocks",
        "compatible_with_c1"],
        sort_by=["regulator_class", "architecture", "estimator_variant",
                 "n_features", "fit_window_index"])

    print(f"Wrote {rows_path.name} and {agg_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
