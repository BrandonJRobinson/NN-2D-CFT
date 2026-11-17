#!/usr/bin/env python3
"""WP-D: finite-feature NS fermionic pairing, CAR, and super-Virasoro audit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnft_virasoro import algebra as AL  # noqa: E402
from nnft_virasoro import architectures as ARCH  # noqa: E402
from nnft_virasoro import darboux as DB  # noqa: E402
from nnft_virasoro import ns_fermion as NS  # noqa: E402
from nnft_virasoro import ordered_pairing as OP  # noqa: E402
from nnft_virasoro.cli import base_parser, setup  # noqa: E402
from nnft_virasoro.config import rng_for  # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows  # noqa: E402
from nnft_virasoro.kernels import physical_window  # noqa: E402
from nnft_virasoro.sampling import block_statistics, bootstrap_ci  # noqa: E402

SCRIPT = "scripts/run_ns_audit.py"


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    p_pair = outdir / "ns_pairing_seed_rows.csv"
    p_alg = outdir / "ns_super_virasoro_seed_rows.csv"
    p_met = outdir / "ns_frozen_metrics.csv"
    p_cor = outdir / "ns_sampled_correlators.csv"
    if all(should_skip(p, args.skip_existing) for p in (p_pair, p_alg, p_met, p_cor)):
        print("  [skip] WP-D")
        return 0

    sec = cfg.section("wp_d")
    g = cfg.section("global")
    lambda_ir, lambda_uv = sec["regulator"]
    r_min, r_max = physical_window(lambda_ir, lambda_uv,
                                   margin_uv=g["window_margin_uv"],
                                   margin_ir=g["window_margin_ir"])
    rho = g["rho_reference"]
    r0 = float(np.sqrt(r_min * r_max))
    r_in, r_out = r0 / np.sqrt(rho), r0 * np.sqrt(rho)
    tt = NS.fermion_mode_window(sec["fermion_cutoff"])
    bmodes = OP.mode_window(sec["boson_mode_cutoff"])
    rclasses = [(str(k), float(v)) for k, v in g["regulator_classes"].items()]

    pair_rows, alg_rows = [], []
    for rclass, taper in rclasses:
        params = {"regulator_taper": taper}
        ana = NS.fermionic_pairing_analytic_spectral(tt, r_in, r_out, r0, lambda_ir,
                                                     lambda_uv, taper=taper)
        ana_m = NS.car_metrics(ana.omega, tt)
        pair_rows.append(stamper.stamp(dict(
            regulator_class=rclass, regulator_taper=taper,
            architecture="selected_log_kernel", seed_name="", n_features=0,
            lambda_ir=lambda_ir, lambda_uv=lambda_uv, r0=r0, rho=rho,
            r_in=r_in, r_out=r_out, r_min_physical=r_min, r_max_physical=r_max,
            fermion_cutoff=sec["fermion_cutoff"], **ana_m,
            post_canonicalization_car_defect=float("nan"),
            star_compatibility_defect=float("nan"), condition_number=float("nan"),
        ), estimator_class="analytic_closed_kernel", seed=None))

        jobs = [("selected_log_kernel", nf, sd)
                for nf in sec["n_features"] for sd in sec["seeds"]]
        jobs += [(a, max(sec["n_features"]), sd)
                 for a in sec["controls"] for sd in sec["seeds"]]
        if True:
            for arch, n_features, seed_name in jobs:
                if True:
                    rng = rng_for(cfg.section("named_seeds"), seed_name, "ns", arch,
                                  n_features, rclass)
                    fs = ARCH.sample_architecture(arch, rng, n_features, lambda_ir,
                                                  lambda_uv, params=params)
                    res = NS.fermionic_pairing_empirical_exact(
                        tt, r_in, r_out, r0, fs.kabs, fs.theta, lambda_ir,
                        lambda_uv, log_measure=fs.log_measure)
                    eta = res.omega
                    met = NS.car_metrics(eta, tt)
                    can_rep = DB.car_canonicalization_report(eta, tt)
                    row = dict(
                        regulator_class=rclass, regulator_taper=taper,
                        architecture=arch, seed_name=seed_name,
                        n_features=n_features, lambda_ir=lambda_ir,
                        lambda_uv=lambda_uv, r0=r0, rho=rho, r_in=r_in,
                        r_out=r_out, r_min_physical=r_min, r_max_physical=r_max,
                        fermion_cutoff=sec["fermion_cutoff"],
                        fermionic_pairing_sampling_error=float(
                            np.linalg.norm(eta - ana.omega)
                            / np.linalg.norm(NS.canonical_car(tt))),
                        # car_canonicalization_report also reports
                        # raw_car_defect; keep the pairing module's value.
                        **met,
                        **{k: v for k, v in can_rep.items()
                           if k not in met})
                    pair_rows.append(stamper.stamp(
                        row, estimator_class="deterministic_finite_feature",
                        seed=seed_name))

                    # super-Virasoro residuals from the measured pairings
                    bres = OP.pairing_empirical_exact(
                        bmodes, r_in, r_out, r0, fs.kabs, fs.theta, fs.prefactor,
                        lambda_ir, lambda_uv, idx_a=fs.idx_a, idx_b=fs.idx_b,
                        log_measure=fs.log_measure)
                    raw = AL.super_virasoro_residuals(
                        bres.omega, bmodes, eta, tt, sec["boson_mode_cutoff"],
                        sec["fermion_cutoff"], level=sec["algebra_level"],
                        r_alg=sec["r_alg"])
                    # canonicalized versions (implementation check only)
                    try:
                        Sb = DB.darboux_antisymmetric(bres.omega, bmodes)
                        Sf = DB.isotropic_symmetric(eta, tt)
                        can = AL.super_virasoro_residuals(
                            Sb @ bres.omega @ Sb.T, bmodes, Sf @ eta @ Sf.T, tt,
                            sec["boson_mode_cutoff"], sec["fermion_cutoff"],
                            level=sec["algebra_level"], r_alg=sec["r_alg"])
                        canon_fail = ""
                    except np.linalg.LinAlgError as exc:
                        can = dict(LG_residual=float("nan"), GG_residual=float("nan"),
                                   central_charge_fit=float("nan"),
                                   central_charge_fit_spread=float("nan"))
                        canon_fail = str(exc)
                    alg_rows.append(stamper.stamp(dict(
                        regulator_class=rclass, architecture=arch,
                        seed_name=seed_name, n_features=n_features,
                        boson_mode_cutoff=sec["boson_mode_cutoff"],
                        fermion_cutoff=sec["fermion_cutoff"],
                        algebra_level=sec["algebra_level"], r_alg=sec["r_alg"],
                        raw_LG_residual=raw["LG_residual"],
                        raw_GG_residual=raw["GG_residual"],
                        raw_central_charge_fit=raw["central_charge_fit"],
                        raw_central_charge_spread=raw["central_charge_fit_spread"],
                        canonical_LG_residual=can["LG_residual"],
                        canonical_GG_residual=can["GG_residual"],
                        central_charge_fit=can["central_charge_fit"],
                        central_charge_target=1.5,
                        canonicalization_failure=canon_fail,
                        n_LG_pairs=len(raw["LG_pairs"]),
                        n_GG_pairs=len(raw["GG_pairs"]),
                        raw_car_defect=met["raw_car_defect"],
                        post_canonicalization_car_defect=can_rep[
                            "post_canonicalization_car_defect"],
                        star_compatibility_defect=can_rep[
                            "star_compatibility_defect"],
                        polarization_min_eigenvalue=float(
                            DB.admissibility_report(
                                bres.omega, bmodes).polarization_min_eigenvalue),
                    ), estimator_class="deterministic_finite_feature",
                        seed=seed_name))
                print(f"  WP-D {rclass}/{arch} N={n_features} done")

    write_rows(p_pair, pair_rows, required=[
        "regulator_class", "architecture", "seed_name", "n_features",
        "fermionic_pairing_relative_error", "fermionic_pairing_leakage",
        "raw_car_defect", "post_canonicalization_car_defect"],
        sort_by=["regulator_class", "architecture", "n_features", "seed_name"])
    write_rows(p_alg, alg_rows, required=[
        "regulator_class", "architecture", "seed_name", "n_features",
        "raw_LG_residual", "raw_GG_residual", "canonical_LG_residual",
        "canonical_GG_residual", "central_charge_fit", "star_compatibility_defect",
        "polarization_min_eigenvalue"],
        sort_by=["regulator_class", "architecture", "n_features", "seed_name"])

    # aggregate frozen metrics
    import pandas as pd

    dfp = pd.DataFrame(pair_rows)
    dfp = dfp[dfp["n_features"] > 0]
    dfa = pd.DataFrame(alg_rows)
    met_rows = []
    for (rclass, arch, n_features), sub in dfp.groupby(
            ["regulator_class", "architecture", "n_features"]):
        suba = dfa[(dfa["regulator_class"] == rclass)
                   & (dfa["architecture"] == arch)
                   & (dfa["n_features"] == n_features)]
        row = dict(regulator_class=rclass, architecture=arch,
                   n_features=int(n_features), n_blocks=int(sub["seed_name"].nunique()))
        for src, fields in ((sub, ["fermionic_pairing_relative_error",
                                   "fermionic_pairing_leakage", "raw_car_defect",
                                   "post_canonicalization_car_defect",
                                   "fermionic_pairing_sampling_error",
                                   "star_compatibility_defect"]),
                            (suba, ["raw_LG_residual", "raw_GG_residual",
                                    "canonical_LG_residual", "canonical_GG_residual",
                                    "central_charge_fit",
                                    "polarization_min_eigenvalue"])):
            for f in fields:
                v = src[f].to_numpy(dtype=float)
                st = block_statistics(v)
                lo, hi = bootstrap_ci(v, seed=20260730)
                row[f"{f}_mean"] = st["mean"]
                row[f"{f}_sd"] = st["sd"]
                row[f"{f}_stderr"] = st["stderr"]
                row[f"{f}_ci_low"] = lo
                row[f"{f}_ci_high"] = hi
        met_rows.append(stamper.stamp(
            row, estimator_class="deterministic_finite_feature", seed="multi"))
    write_rows(p_met, met_rows, required=[
        "regulator_class", "architecture", "n_features",
        "fermionic_pairing_relative_error_mean", "fermionic_pairing_leakage_mean",
        "raw_car_defect_mean", "post_canonicalization_car_defect_mean",
        "raw_LG_residual_mean", "raw_GG_residual_mean",
        "canonical_LG_residual_mean", "canonical_GG_residual_mean",
        "central_charge_fit_mean", "star_compatibility_defect_mean",
        "polarization_min_eigenvalue_mean", "n_blocks"],
        sort_by=["regulator_class", "architecture", "n_features"])

    # sampled fermion / supercurrent correlators
    sc = sec["sampled_correlators"]
    seps = np.asarray(sc["separations"], dtype=float)
    cor_rows = []
    for rclass, taper in rclasses:
        for seed_name in sec["seeds"][:3]:
            rng = rng_for(cfg.section("named_seeds"), seed_name, "nscorr", rclass)
            res = NS.sampled_fermion_correlator(
                rng, n_features=sc["n_features"], n_networks=sc["n_networks"],
                separations=seps, lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                params={"regulator_taper": taper}, batch_size=sc["batch_size"])
            for i, r in enumerate(seps):
                cor_rows.append(stamper.stamp(dict(
                    regulator_class=rclass, seed_name=seed_name, separation=float(r),
                    n_features=res["n_features"], n_networks=res["n_networks"],
                    fermion_correlator=float(np.real(res["fermion_correlator"][i])),
                    fermion_correlator_stderr=float(
                        res["fermion_correlator_stderr"][i]),
                    fermion_analytic=float(np.real(res["fermion_analytic"][i])),
                    fermion_ideal=float(np.real(res["fermion_ideal"][i])),
                    supercurrent_correlator=float(
                        np.real(res["supercurrent_correlator"][i])),
                    relative_error_vs_analytic=float(
                        abs(res["fermion_correlator"][i] - res["fermion_analytic"][i])
                        / max(abs(res["fermion_analytic"][i]), 1e-300)),
                ), estimator_class="sampled_finite_network", seed=seed_name))
        print(f"  WP-D sampled correlators {rclass} done")
    write_rows(p_cor, cor_rows, required=[
        "regulator_class", "seed_name", "separation", "n_features", "n_networks",
        "fermion_correlator", "fermion_correlator_stderr", "fermion_analytic"],
        sort_by=["regulator_class", "separation", "seed_name"])
    print(f"  wrote {len(pair_rows)} pairing, {len(alg_rows)} algebra, "
          f"{len(met_rows)} metric, {len(cor_rows)} correlator rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
