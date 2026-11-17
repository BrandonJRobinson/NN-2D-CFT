#!/usr/bin/env python3
"""WP-C: current and stress Ward identities in three separate estimator classes.

Emits `analytic_ward_points.csv`, `deterministic_feature_ward_points.csv` and
`sampled_ward_seed_rows.csv` / `sampled_ward_summary.csv`.  The three classes
are never combined in one uncertainty calculation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnft_virasoro import architectures as ARCH  # noqa: E402
from nnft_virasoro import ward as WD  # noqa: E402
from nnft_virasoro.cli import base_parser, setup  # noqa: E402
from nnft_virasoro.config import rng_for  # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows  # noqa: E402
from nnft_virasoro.kernels import physical_window  # noqa: E402
from nnft_virasoro.sampling import block_statistics, bootstrap_ci  # noqa: E402

SCRIPT = "scripts/run_ward_audit.py"


def as_complex(pairs):
    return np.array([complex(p[0], p[1]) for p in pairs])


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    paths = {k: outdir / v for k, v in dict(
        analytic="analytic_ward_points.csv",
        deterministic="deterministic_feature_ward_points.csv",
        sampled="sampled_ward_seed_rows.csv",
        summary="sampled_ward_summary.csv").items()}
    if all(should_skip(p, args.skip_existing) for p in paths.values()):
        print("  [skip] WP-C")
        return 0

    sec = cfg.section("wp_c")
    g = cfg.section("global")
    lambda_ir, lambda_uv = sec["regulator"]
    r_min, r_max = physical_window(lambda_ir, lambda_uv,
                                   margin_uv=g["window_margin_uv"],
                                   margin_ir=g["window_margin_ir"])
    probes = as_complex(sec["probe_positions"])
    rclasses = [(str(k), float(v)) for k, v in g["regulator_classes"].items()]
    configs = {name: (np.asarray(d["charges"], dtype=float),
                      as_complex(d["positions"]))
               for name, d in sec["configurations"].items()}

    common = dict(lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                  r_min_physical=r_min, r_max_physical=r_max,
                  complementary_interpretation=WD.COMPLEMENTARY_INTERPRETATION)

    # ---- 1. analytic closed kernel -----------------------------------
    a_rows = []
    for cname, (charges, positions) in configs.items():
        res = WD.analytic_ward(lambda_ir, lambda_uv, charges, positions, probes)
        for i, zp in enumerate(probes):
            a_rows.append(stamper.stamp(dict(
                configuration=cname, n_vertices=len(charges),
                charges=";".join(f"{c:g}" for c in charges),
                positions=";".join(f"{p.real:g}{p.imag:+g}j" for p in positions),
                probe_position=f"{zp.real:g}{zp.imag:+g}j",
                probe_index=i, normal_ordering_subtraction=0.0,
                current_estimate=abs(res["current_estimate"][i]),
                current_target=abs(res["current_target"][i]),
                current_absolute_error=float(res["current_absolute_error"][i]),
                current_relative_error=float(res["current_relative_error"][i]),
                stress_estimate=abs(res["stress_estimate"][i]),
                stress_target=abs(res["stress_target"][i]),
                stress_absolute_error=float(res["stress_absolute_error"][i]),
                stress_relative_error=float(res["stress_relative_error"][i]),
                **common), estimator_class="analytic_closed_kernel", seed=None))
    write_rows(paths["analytic"], a_rows, required=[
        "configuration", "probe_index", "current_absolute_error",
        "stress_absolute_error"], sort_by=["configuration", "probe_index"])
    print(f"  analytic: {len(a_rows)} rows")

    # ---- 2. deterministic finite feature ------------------------------
    d_rows = []
    for rclass, taper in rclasses:
        params = dict(sec.get("params", {}))
        params["regulator_taper"] = taper
        for arch in ["selected_log_kernel"] + list(sec["controls"]):
            for n_features in sec["deterministic_n_features"]:
                for seed_name in sec["deterministic_seeds"]:
                    rng = rng_for(cfg.section("named_seeds"), seed_name, "warddet",
                                  arch, n_features, rclass)
                    fs = ARCH.sample_architecture(arch, rng, n_features, lambda_ir,
                                                  lambda_uv, params=params)
                    for cname, (charges, positions) in configs.items():
                        res = WD.deterministic_ward(fs, charges, positions, probes)
                        for i, zp in enumerate(probes):
                            d_rows.append(stamper.stamp(dict(
                                regulator_class=rclass, architecture=arch,
                                configuration=cname, n_vertices=len(charges),
                                n_features=n_features, seed_name=seed_name,
                                probe_index=i,
                                probe_position=f"{zp.real:g}{zp.imag:+g}j",
                                charges=";".join(f"{c:g}" for c in charges),
                                normal_ordering_subtraction=0.0,
                                current_absolute_error=float(
                                    res["current_absolute_error"][i]),
                                current_relative_error=float(
                                    res["current_relative_error"][i]),
                                stress_absolute_error=float(
                                    res["stress_absolute_error"][i]),
                                stress_relative_error=float(
                                    res["stress_relative_error"][i]),
                                **common),
                                estimator_class="deterministic_finite_feature",
                                seed=seed_name))
            print(f"  deterministic {rclass}/{arch} done ({len(d_rows)} rows)")
    write_rows(paths["deterministic"], d_rows, required=[
        "regulator_class", "architecture", "configuration", "n_features",
        "seed_name", "probe_index", "current_absolute_error",
        "stress_absolute_error"],
        sort_by=["regulator_class", "architecture", "configuration", "n_features",
                 "seed_name", "probe_index"])

    # ---- 3. sampled finite network ------------------------------------
    s_rows = []
    for rclass, taper in rclasses:
        params = dict(sec.get("params", {}))
        params["regulator_taper"] = taper
        jobs = [("selected_log_kernel", nf, m, sd)
                for nf in sec["n_features"] for m in sec["n_networks"]
                for sd in sec["seeds"]]
        jobs += [(a, max(sec["n_features"]), max(sec["n_networks"]), sd)
                 for a in sec["controls"] for sd in sec["control_seeds"]]
        if True:
            for arch, n_features, n_networks, seed_name in jobs:
                    if True:
                        rng = rng_for(cfg.section("named_seeds"), seed_name, "ward",
                                      arch, n_features, n_networks, rclass)
                        for cname, (charges, positions) in configs.items():
                            res = WD.sampled_ward(
                                rng, n_features=n_features, n_networks=n_networks,
                                charges=charges, positions=positions, probes=probes,
                                lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                                architecture=arch, params=params,
                                batch_size=sec["batch_size"])
                            for i, zp in enumerate(probes):
                                s_rows.append(stamper.stamp(dict(
                                    regulator_class=rclass, architecture=arch,
                                    configuration=cname, n_vertices=len(charges),
                                    n_features=n_features, n_networks=n_networks,
                                    seed_name=seed_name, block_id=seed_name,
                                    probe_index=i,
                                    probe_position=f"{zp.real:g}{zp.imag:+g}j",
                                    charges=";".join(f"{c:g}" for c in charges),
                                    positions=";".join(
                                        f"{p.real:g}{p.imag:+g}j" for p in positions),
                                    normal_ordering_subtraction=abs(
                                        res["normal_ordering_subtraction"]),
                                    vertex_normal_ordering_constant=res[
                                        "vertex_normal_ordering_constant"],
                                    denominator_abs=res["denominator_abs"],
                                    current_estimate=abs(res["current_estimate"][i]),
                                    current_target=abs(res["current_target"][i]),
                                    stress_estimate=abs(res["stress_estimate"][i]),
                                    stress_target=abs(res["stress_target"][i]),
                                    current_absolute_error=float(
                                        res["current_absolute_error"][i]),
                                    current_relative_error=float(
                                        res["current_relative_error"][i]),
                                    stress_absolute_error=float(
                                        res["stress_absolute_error"][i]),
                                    stress_relative_error=float(
                                        res["stress_relative_error"][i]),
                                    **common),
                                    estimator_class="sampled_finite_network",
                                    seed=seed_name))
                    print(f"  sampled {rclass}/{arch} N={n_features} "
                          f"M={n_networks} done ({len(s_rows)} rows)")
    write_rows(paths["sampled"], s_rows, required=[
        "regulator_class", "architecture", "configuration", "n_features",
        "n_networks", "seed_name", "block_id", "charges", "positions",
        "probe_position", "normal_ordering_subtraction", "current_estimate",
        "current_target", "current_absolute_error", "current_relative_error",
        "stress_absolute_error", "stress_relative_error"],
        sort_by=["regulator_class", "architecture", "configuration", "n_features",
                 "n_networks", "seed_name", "probe_index"])

    import pandas as pd

    df = pd.DataFrame(s_rows)
    summary = []
    for (rclass, arch, cname, n_features, n_networks), sub in df.groupby(
            ["regulator_class", "architecture", "configuration", "n_features",
             "n_networks"]):
        row = dict(regulator_class=rclass, architecture=arch, configuration=cname,
                   n_features=int(n_features), n_networks=int(n_networks),
                   n_vertices=int(sub["n_vertices"].iloc[0]),
                   n_probe_points=int(sub["probe_index"].nunique()))
        for field in ("current_absolute_error", "stress_absolute_error",
                      "current_relative_error", "stress_relative_error"):
            per_seed = sub.groupby("seed_name")[field].max().to_numpy()
            st = block_statistics(per_seed)
            lo, hi = bootstrap_ci(per_seed, seed=20260730)
            row[f"{field}_mean"] = st["mean"]
            row[f"{field}_sd"] = st["sd"]
            row[f"{field}_stderr"] = st["stderr"]
            row[f"{field}_boot_ci_low"] = lo
            row[f"{field}_boot_ci_high"] = hi
            row[f"{field}_max"] = float(sub[field].max())
        row["n_blocks"] = int(sub["seed_name"].nunique())
        summary.append(stamper.stamp(
            row, estimator_class="sampled_finite_network", seed="multi"))
    write_rows(paths["summary"], summary, required=[
        "regulator_class", "architecture", "configuration", "n_features",
        "n_networks", "current_absolute_error_mean", "current_absolute_error_stderr",
        "stress_absolute_error_mean", "stress_absolute_error_stderr", "n_blocks"],
        sort_by=["regulator_class", "architecture", "configuration", "n_features",
                 "n_networks"])
    print(f"  wrote {len(s_rows)} sampled rows, {len(summary)} summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
