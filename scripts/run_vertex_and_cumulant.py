#!/usr/bin/env python3
"""WP-H: sampled vertex weights (H1) and finite-width connected cumulant (H2).

The cumulant is interpreted only as finite-width non-Gaussianity, never as
evidence of a local interacting CFT (appendix F).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnft_virasoro import vertex as VX  # noqa: E402
from nnft_virasoro.cli import base_parser, setup  # noqa: E402
from nnft_virasoro.config import rng_for  # noqa: E402
from nnft_virasoro.fits import null_test, power_law_fit  # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows  # noqa: E402
from nnft_virasoro.kernels import physical_window  # noqa: E402
from nnft_virasoro.sampling import block_statistics, bootstrap_ci  # noqa: E402

SCRIPT = "scripts/run_vertex_and_cumulant.py"

CUMULANT_INTERPRETATION = (
    "Interpreted only as finite-width non-Gaussianity of the neural ensemble "
    "(appendix F). It is NOT evidence of a local interacting CFT."
)


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    paths = {k: outdir / v for k, v in dict(
        vrows="sampled_vertex_weights_seed_rows.csv",
        vagg="sampled_vertex_weights.csv",
        crows="finite_width_cumulant_seed_rows.csv",
        cagg="finite_width_cumulant.csv").items()}
    if all(should_skip(p, args.skip_existing) for p in paths.values()):
        print("  [skip] WP-H")
        return 0
    g = cfg.section("global")
    rclasses = [(str(k), float(v)) for k, v in g["regulator_classes"].items()]

    # ---------------- H1 vertex weights ----------------
    sec = cfg.section("wp_h")["vertex"]
    lambda_ir, lambda_uv = sec["regulator"]
    r_min, r_max = physical_window(lambda_ir, lambda_uv,
                                   margin_uv=g["window_margin_uv"],
                                   margin_ir=g["window_margin_ir"])
    seps = np.asarray(sec["separations"], dtype=float)
    windows = [tuple(w) for w in sec["fit_windows"]]
    vrows = []
    for rclass, taper in rclasses:
        for charge in sec["charges"]:
            for n_features in sec["n_features"]:
                for seed_name in sec["seeds"]:
                    rng = rng_for(cfg.section("named_seeds"), seed_name, "vertex",
                                  charge, n_features, rclass)
                    res = VX.sampled_vertex_two_point(
                        rng, charge=charge, n_features=n_features,
                        n_networks=sec["n_networks"], separations=seps,
                        lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                        params={"regulator_taper": taper},
                        batch_size=sec["batch_size"])
                    for wi, win in enumerate(windows):
                        fit = VX.fit_vertex_weight(res["separations"],
                                                   res["correlator"], win, charge)
                        vrows.append(stamper.stamp(dict(
                            regulator_class=rclass, regulator_taper=taper,
                            charge=float(charge), n_features=int(n_features),
                            n_networks=res["n_networks"], seed_name=seed_name,
                            lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                            r_min_physical=r_min, r_max_physical=r_max,
                            fit_window_index=wi,
                            is_reference_window=bool(wi == 0),
                            normal_ordering_constant=res["normal_ordering_constant"],
                            **fit),
                            estimator_class="sampled_finite_network",
                            seed=seed_name))
                print(f"  WP-H1 {rclass} a={charge} N={n_features} done")
    write_rows(paths["vrows"], vrows, required=[
        "regulator_class", "charge", "n_features", "n_networks", "seed_name",
        "fit_window_index", "slope", "h_fit", "h_target", "r_squared"],
        sort_by=["regulator_class", "charge", "n_features", "fit_window_index",
                 "seed_name"])

    import pandas as pd

    dfv = pd.DataFrame(vrows)
    vagg = []
    for (rclass, charge, n_features, wi), sub in dfv.groupby(
            ["regulator_class", "charge", "n_features", "fit_window_index"]):
        v = sub["h_fit"].to_numpy(dtype=float)
        st = block_statistics(v)
        lo, hi = bootstrap_ci(v, seed=20260730)
        target = float(charge) ** 2 / 2.0
        vagg.append(stamper.stamp(dict(
            regulator_class=rclass, charge=float(charge),
            n_features=int(n_features), fit_window_index=int(wi),
            fit_window_low=float(sub["fit_window_low"].iloc[0]),
            fit_window_high=float(sub["fit_window_high"].iloc[0]),
            n_networks=int(sub["n_networks"].iloc[0]),
            h_fit=st["mean"], h_sd=st["sd"], h_stderr=st["stderr"],
            h_ci_low=lo, h_ci_high=hi, h_target=target, n_blocks=st["n_blocks"],
            slope_mean=float(sub["slope"].mean()),
            r_squared_mean=float(sub["r_squared"].mean()),
            compatible_with_target=bool(
                st["stderr"] > 0 and abs(st["mean"] - target) <= 2 * st["stderr"]),
            z_vs_target=float((st["mean"] - target) / st["stderr"])
            if st["stderr"] > 0 else float("nan")),
            estimator_class="sampled_finite_network", seed="multi"))
    write_rows(paths["vagg"], vagg, required=[
        "regulator_class", "charge", "n_features", "fit_window_index", "h_fit",
        "h_stderr", "h_target", "n_blocks", "compatible_with_target"],
        sort_by=["regulator_class", "charge", "n_features", "fit_window_index"])

    # ---------------- H2 finite-width cumulant ----------------
    sec = cfg.section("wp_h")["cumulant"]
    lambda_ir, lambda_uv = sec["regulator"]
    crows = []
    for rclass, taper in rclasses:
        for n_features in sec["n_features"]:
            for seed_name in sec["seeds"]:
                rng = rng_for(cfg.section("named_seeds"), seed_name, "cumulant",
                              n_features, rclass)
                res = VX.sampled_cumulant4(
                    rng, n_features=n_features, n_samples=sec["n_samples"],
                    radius=sec["evaluation_radius"], lambda_ir=lambda_ir,
                    lambda_uv=lambda_uv, params={"regulator_taper": taper})
                crows.append(stamper.stamp(dict(
                    regulator_class=rclass, regulator_taper=taper,
                    seed_name=seed_name, lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                    interpretation=CUMULANT_INTERPRETATION, **res),
                    estimator_class="sampled_finite_network", seed=seed_name))
            print(f"  WP-H2 {rclass} N={n_features} done")
    write_rows(paths["crows"], crows, required=[
        "regulator_class", "n_features", "n_samples", "seed_name", "variance",
        "cumulant4", "cumulant4_stderr", "excess_kurtosis",
        "kappa4_prediction", "interpretation"],
        sort_by=["regulator_class", "n_features", "seed_name"])

    dfc = pd.DataFrame(crows)
    cagg = []
    lo_n, hi_n = sec["fit_range"]
    for rclass, sub in dfc.groupby("regulator_class"):
        per_n = sub.groupby("n_features").agg(
            k4=("cumulant4", "mean"), k4_sd=("cumulant4", "std"),
            n=("cumulant4", "size"), pred=("kappa4_prediction", "first")).reset_index()
        sel = (per_n["n_features"] >= lo_n) & (per_n["n_features"] <= hi_n)
        fit = power_law_fit(per_n.loc[sel, "n_features"].to_numpy(),
                            np.abs(per_n.loc[sel, "k4"].to_numpy()),
                            notes="|kappa_4| ~ N^p; theory p = -1")
        nt = null_test(fit.coef("exponent"), fit.se("exponent"), -1.0, fit.dof)
        for _, r in per_n.iterrows():
            blocks = sub[sub["n_features"] == r["n_features"]]["cumulant4"].to_numpy()
            st = block_statistics(blocks)
            blo, bhi = bootstrap_ci(blocks, seed=20260730)
            cagg.append(stamper.stamp(dict(
                regulator_class=rclass, n_features=int(r["n_features"]),
                n_samples=int(sub["n_samples"].iloc[0]),
                cumulant4=st["mean"], cumulant4_sd=st["sd"],
                cumulant4_stderr=st["stderr"], cumulant4_ci_low=blo,
                cumulant4_ci_high=bhi, n_blocks=st["n_blocks"],
                kappa4_prediction=float(r["pred"]),
                relative_deviation_from_prediction=float(
                    abs(st["mean"] - r["pred"]) / abs(r["pred"])),
                in_fit_range=bool(lo_n <= r["n_features"] <= hi_n),
                fit_exponent=fit.coef("exponent"),
                fit_exponent_stderr=fit.se("exponent"),
                fit_exponent_ci_low=fit.interval("exponent")[0],
                fit_exponent_ci_high=fit.interval("exponent")[1],
                fit_exponent_boot_ci_low=float(
                    fit.boot_ci_low[fit.names.index("exponent")]),
                fit_exponent_boot_ci_high=float(
                    fit.boot_ci_high[fit.names.index("exponent")]),
                fit_r_squared=fit.r_squared, fit_n_rows=fit.n_rows,
                fit_excluded_rows=fit.excluded_rows,
                fit_range_low=int(lo_n), fit_range_high=int(hi_n),
                null_exponent=nt["null_value"], null_t_statistic=nt["t_statistic"],
                null_p_value=nt["p_value"],
                null_rejected_at_5pct=nt["reject_at_5pct"],
                interpretation=CUMULANT_INTERPRETATION),
                estimator_class="sampled_finite_network", seed="multi"))
        print(f"  WP-H2 {rclass}: exponent={fit.coef('exponent'):.4f} "
              f"+-{fit.se('exponent'):.4f}  R2={fit.r_squared:.4f}  "
              f"p(null=-1)={nt['p_value']:.4g}")
    write_rows(paths["cagg"], cagg, required=[
        "regulator_class", "n_features", "cumulant4", "cumulant4_stderr",
        "kappa4_prediction", "fit_exponent", "fit_exponent_stderr",
        "fit_r_squared", "null_p_value", "interpretation"],
        sort_by=["regulator_class", "n_features"])
    print(f"  wrote {len(vrows)}/{len(vagg)} vertex rows, "
          f"{len(crows)}/{len(cagg)} cumulant rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
