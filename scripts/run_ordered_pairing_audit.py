#!/usr/bin/env python3
"""WP-A: operational two-radius pairing, radius plateau, convergence, controls.

Stages (each independently `--skip-existing`-able):

  A3  ordered_pairing_grid.csv           radius/rho campaign + non-holomorphy
      ordered_pairing_radius_plateau.csv stated plateau algorithm
  A4  robustness_grid_recomputed.csv     the 81 released configurations
  A5  predarboux_convergence_seed_rows.csv / _fits.csv
  A6  architecture_controls_seed_rows.csv / darboux_physical_admissibility.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnft_virasoro import algebra as AL  # noqa: E402
from nnft_virasoro import architectures as ARCH  # noqa: E402
from nnft_virasoro import darboux as DB  # noqa: E402
from nnft_virasoro import kernels as KR  # noqa: E402
from nnft_virasoro import nonholomorphy as NH  # noqa: E402
from nnft_virasoro import ordered_pairing as OP  # noqa: E402
from nnft_virasoro import ward as WD  # noqa: E402
from nnft_virasoro.cli import base_parser, setup  # noqa: E402
from nnft_virasoro.config import rng_for  # noqa: E402
from nnft_virasoro.fits import loglog_scaling_fit, null_test  # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_json, write_rows  # noqa: E402
from nnft_virasoro.sampling import block_statistics, bootstrap_ci  # noqa: E402

SCRIPT = "scripts/run_ordered_pairing_audit.py"


# ---------------------------------------------------------------------------
def window_of(cfg, lambda_ir, lambda_uv):
    g = cfg.section("global")
    return KR.physical_window(lambda_ir, lambda_uv,
                              margin_uv=g["window_margin_uv"],
                              margin_ir=g["window_margin_ir"])


def radii_for(r_min, r_max, rho, frac=None):
    """Admissible R_0 band: r_min < R_0/sqrt(rho) and R_0*sqrt(rho) < r_max."""
    lo = r_min * np.sqrt(rho)
    hi = r_max / np.sqrt(rho)
    if hi <= lo:
        return None
    r0 = float(np.sqrt(lo * hi)) if frac is None else float(lo * (hi / lo) ** frac)
    return OP.radii_from_window(r_min, r_max, rho, r0=r0)


_ANALYTIC_CACHE: dict = {}


def analytic_pairing_cached(modes, r_in, r_out, r0, lambda_ir, lambda_uv, taper):
    """Cache the analytic closed-kernel reference: it depends only on the
    geometry and the regulator, never on the seed or the feature width."""
    key = (int(modes[-1]), round(r_in, 12), round(r_out, 12), round(r0, 12),
           lambda_ir, lambda_uv, round(float(taper), 6))
    hit = _ANALYTIC_CACHE.get(key)
    if hit is None:
        hit = OP.pairing_analytic_spectral(modes, r_in, r_out, r0, lambda_ir,
                                           lambda_uv, taper=taper)
        _ANALYTIC_CACHE[key] = hit
    return hit


def regulator_classes(cfg):
    """(name, taper) pairs; the sharp annulus is the manuscript claim."""
    rc = cfg.section("global")["regulator_classes"]
    return [(str(k), float(v)) for k, v in rc.items()]


def pairing_row(cfg, arch, seed_name, lambda_ir, lambda_uv, n_features, mode_cutoff,
                rho, r0, r_in, r_out, r_min, r_max, *, params=None,
                with_analytic=False, with_proxy=False, with_algebra=True,
                stream_label=(), regulator_class="sharp_annulus_manuscript",
                taper=0.0):
    """One frozen pairing measurement (deterministic_finite_feature)."""
    g = cfg.section("global")
    modes = OP.mode_window(mode_cutoff)
    rng = rng_for(cfg.section("named_seeds"), seed_name,
                  arch, lambda_ir, lambda_uv, n_features, mode_cutoff, rho, r0,
                  *stream_label)
    params = dict(params or {})
    params["regulator_taper"] = float(taper)
    fs = ARCH.sample_architecture(arch, rng, n_features, lambda_ir, lambda_uv,
                                  params=params)
    res = OP.pairing_empirical_exact(modes, r_in, r_out, r0, fs.kabs, fs.theta,
                                     fs.prefactor, lambda_ir, lambda_uv,
                                     idx_a=fs.idx_a, idx_b=fs.idx_b,
                                     log_measure=fs.log_measure)
    omega = res.omega
    postmap = ARCH.mode_postmap(arch, modes, rng, params=params)
    if postmap is not None:
        omega = postmap @ omega @ postmap.T

    row = dict(
        architecture=arch, seed_name=seed_name,
        lambda_ir=lambda_ir, lambda_uv=lambda_uv, n_features=int(n_features),
        mode_cutoff=int(mode_cutoff), r_min_physical=r_min, r_max_physical=r_max,
        r0=r0, rho=rho, r_in=r_in, r_out=r_out,
        window_margin_uv=g["window_margin_uv"], window_margin_ir=g["window_margin_ir"],
        quadrature_points=0, regulator_taper=params["regulator_taper"],
        regulator_class=regulator_class, log_measure=fs.log_measure,
        mode_postmap_applied=bool(postmap is not None),
        n_terms=int(res.meta["n_terms"]),
    )
    row.update(OP.pairing_metrics(omega, modes))

    if with_analytic:
        sp = analytic_pairing_cached(modes, r_in, r_out, r0, lambda_ir,
                                     lambda_uv, taper)
        can = OP.canonical_pairing(modes)
        am = OP.pairing_metrics(sp.omega, modes)
        row["analytic_pairing_relative_error"] = am["pairing_relative_error"]
        row["analytic_pairing_leakage"] = am["pairing_leakage"]
        row["analytic_anti_diagonal_amplitude_error"] = am[
            "anti_diagonal_amplitude_error"
        ]
        row["pairing_sampling_error"] = float(
            np.linalg.norm(omega - sp.omega) / np.linalg.norm(can)
        )
        row["analytic_quadrature_points"] = int(sp.meta["quadrature_points"])
        row["analytic_asymptotic_tail"] = bool(sp.meta["asymptotic_tail"])

    if with_proxy:
        cov = OP.projection_proxy_covariance(modes, r0, r0, fs.kabs, fs.theta,
                                             fs.prefactor, lambda_ir, lambda_uv,
                                             idx_a=fs.idx_a, idx_b=fs.idx_b,
                                             log_measure=fs.log_measure)
        row.update(OP.projection_proxy_metrics(cov, modes))
        row["two_radius_pairing_leakage"] = row["pairing_leakage"]
        pl = row["projection_pairing_leakage"]
        row["projection_vs_two_radius_relative_error"] = (
            float(abs(pl - row["pairing_leakage"]) / max(row["pairing_leakage"], 1e-300))
        )

    if with_algebra:
        try:
            rr = AL.raw_virasoro_residual(
                omega, modes, mode_cutoff, level=g["algebra_level"],
                m_alg=g["algebra_mode_span"], charge=g["algebra_charges"][0],
                central_charge=g["central_charge"],
            )
            row["raw_virasoro_max_residual"] = rr.value
            row["raw_virasoro_residual"] = rr.value
            row["algebra_pairs"] = len(rr.pairs)
            row["algebra_level"] = rr.level
        except ValueError as exc:
            row["raw_virasoro_max_residual"] = float("nan")
            row["raw_virasoro_residual"] = float("nan")
            row["algebra_pairs"] = 0
            row["algebra_level"] = g["algebra_level"]
            row["algebra_note"] = str(exc)
    return row, omega, modes, fs


# ---------------------------------------------------------------------------
def stage_radius_plateau(cfg, stamper, outdir, figdir, skip):
    grid_path = outdir / "ordered_pairing_grid.csv"
    plateau_path = outdir / "ordered_pairing_radius_plateau.csv"
    if should_skip(grid_path, skip) and should_skip(plateau_path, skip):
        print("  [skip] A3 radius plateau")
        return
    sec = cfg.section("wp_a")["radius_plateau"]
    g = cfg.section("global")
    rows = []
    for rclass, taper in regulator_classes(cfg):
      for lambda_ir, lambda_uv in sec["regulators"]:
        r_min, r_max = window_of(cfg, lambda_ir, lambda_uv)
        for rho in sec["rho_values"]:
            for frac in sec["r0_fractions"]:
                rr = radii_for(r_min, r_max, rho, frac)
                if rr is None:
                    continue
                r0, r_in, r_out = rr
                nh = NH.nonholomorphy_norms(r_in, r_out, lambda_ir, lambda_uv,
                                            q=g["quadrature_points"], taper=taper)
                for seed_name in sec["seeds"]:
                    row, omega, modes, _ = pairing_row(
                        cfg, "selected_log_kernel", seed_name, lambda_ir, lambda_uv,
                        sec["n_features"], sec["mode_cutoff"], rho, r0, r_in, r_out,
                        r_min, r_max, with_analytic=True, stream_label=("plateau",),
                        regulator_class=rclass, taper=taper,
                    )
                    row.update(nh)
                    row["r0_fraction"] = frac
                    row["measured_pairing_variation"] = float("nan")
                    row["cauchy_green_bound"] = float("nan")
                    row["bound_ratio"] = float("nan")
                    rows.append(stamper.stamp(
                        row, estimator_class="deterministic_finite_feature",
                        seed=seed_name))
        print(f"  A3 {rclass} regulator ({lambda_ir:.0e},{lambda_uv:.0e}) done "
              f"({len(rows)} rows so far)")

    taper_of = {k: float(v) for k, v in
                cfg.section("global")["regulator_classes"].items()}
    metric = cfg.section("wp_a")["radius_plateau"]["plateau_metric"]
    k_sigma = float(sec["plateau_k_sigma"])

    # --- adjacent-node variation, Cauchy-Green bound, and plateau extraction
    plateau_rows = []
    import pandas as pd

    df = pd.DataFrame(rows)
    for (rclass, lambda_ir, lambda_uv, rho), sub in df.groupby(
            ["regulator_class", "lambda_ir", "lambda_uv", "rho"]):
        r_min, r_max = window_of(cfg, lambda_ir, lambda_uv)
        agg = (sub.groupby("r0_fraction")
                  .agg(r0=("r0", "first"), r_in=("r_in", "first"),
                       r_out=("r_out", "first"), y=(metric, "mean"),
                       y_sd=(metric, "std"), n=(metric, "size"),
                       nonholomorphy_sup=("nonholomorphy_sup", "first"))
                  .sort_values("r0").reset_index())
        agg["y_se"] = agg["y_sd"] / np.sqrt(agg["n"].clip(lower=1))
        edges = []
        for i in range(len(agg) - 1):
            a = agg.iloc[i]
            b = agg.iloc[i + 1]
            bound = NH.cauchy_green_variation_bound(
                OP.mode_window(sec["mode_cutoff"]),
                (a.r0, a.r_in, a.r_out), (b.r0, b.r_in, b.r_out),
                lambda_ir, lambda_uv,
                n_radial=sec["cauchy_green_radial_nodes"],
                n_angle=sec["cauchy_green_angular_nodes"],
                taper=float(taper_of[rclass]),
            )
            dy = float(abs(b.y - a.y))
            tol = k_sigma * float(np.hypot(a.y_se, b.y_se)) + bound[
                "cauchy_green_bound_antidiagonal"]
            edges.append(dict(
                i=i, r0_lo=float(a.r0), r0_hi=float(b.r0),
                measured_pairing_variation=dy,
                statistical_tolerance=k_sigma * float(np.hypot(a.y_se, b.y_se)),
                cauchy_green_bound=bound["cauchy_green_bound_antidiagonal"],
                cauchy_green_bound_full=bound["cauchy_green_bound"],
                tolerance=tol, compatible=bool(dy <= tol),
                bound_ratio=float(dy / bound["cauchy_green_bound_antidiagonal"])
                if bound["cauchy_green_bound_antidiagonal"] > 0 else float("inf"),
            ))
        # longest connected run of compatible edges
        best_len, best_start, cur_len, cur_start = 0, None, 0, None
        for e in edges:
            if e["compatible"]:
                if cur_len == 0:
                    cur_start = e["i"]
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        if best_len > 0:
            lo = float(agg.iloc[best_start].r0)
            hi = float(agg.iloc[best_start + best_len].r0)
            plateau_nodes = best_len + 1
        else:
            lo = hi = float("nan")
            plateau_nodes = 0
        for e in edges:
            plateau_rows.append(stamper.stamp(dict(
                regulator_class=rclass,
                lambda_ir=lambda_ir, lambda_uv=lambda_uv, rho=rho,
                mode_cutoff=sec["mode_cutoff"], n_features=sec["n_features"],
                r_min_physical=r_min, r_max_physical=r_max,
                plateau_metric=metric, plateau_k_sigma=k_sigma,
                plateau_algorithm=(
                    "adjacent R_0 nodes compatible when |dY| <= k_sigma*sqrt(se_i^2"
                    "+se_j^2) + CauchyGreen(i,j); plateau = longest connected run"),
                plateau_r0_lo=lo, plateau_r0_hi=hi, plateau_nodes=plateau_nodes,
                plateau_n_edges=len(edges), n_seeds=int(agg["n"].iloc[0]),
                **{k: v for k, v in e.items() if k != "i"},
            ), estimator_class="deterministic_finite_feature", seed="multi"))
        print(f"  A3 plateau {rclass} ({lambda_ir:.0e},{lambda_uv:.0e}) rho={rho}: "
              f"{plateau_nodes} nodes in [{lo:.4g},{hi:.4g}]")

    write_rows(grid_path, rows, required=[
        "regulator_class",
        "lambda_ir", "lambda_uv", "n_features", "mode_cutoff", "r_min_physical",
        "r_max_physical", "r0", "rho", "r_in", "r_out", "quadrature_points",
        "pairing_relative_error", "pairing_leakage",
        "anti_diagonal_amplitude_error", "raw_virasoro_max_residual",
        "nonholomorphy_sup", "cauchy_green_bound"],
        sort_by=["regulator_class", "lambda_ir", "lambda_uv", "rho", "r0",
                 "seed_name"])
    write_rows(plateau_path, plateau_rows, required=[
        "regulator_class",
        "lambda_ir", "lambda_uv", "rho", "plateau_r0_lo", "plateau_r0_hi",
        "measured_pairing_variation", "cauchy_green_bound", "bound_ratio",
        "compatible"],
        sort_by=["regulator_class", "lambda_ir", "lambda_uv", "rho", "r0_lo"])
    print(f"  wrote {grid_path.name} ({len(rows)}), {plateau_path.name} "
          f"({len(plateau_rows)})")


# ---------------------------------------------------------------------------
def stage_robustness(cfg, stamper, outdir, skip):
    path = outdir / "robustness_grid_recomputed.csv"
    if should_skip(path, skip):
        print("  [skip] A4 robustness recompute")
        return
    sec = cfg.section("wp_a")["robustness_grid"]
    rho = cfg.section("global")["rho_reference"]
    rows = []
    for rclass, taper in regulator_classes(cfg):
      for lambda_ir in sec["lambda_ir"]:
        for lambda_uv in sec["lambda_uv"]:
            r_min, r_max = window_of(cfg, lambda_ir, lambda_uv)
            rr = radii_for(r_min, r_max, rho)
            for n_features in sec["n_features"]:
                for mode_cutoff in sec["mode_cutoff"]:
                    if rr is None:
                        rows.append(stamper.stamp(dict(
                            regulator_class=rclass,
                            lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                            n_features=n_features, mode_cutoff=mode_cutoff,
                            r_min_physical=r_min, r_max_physical=r_max,
                            window_empty=True,
                            note="physical window empty at these cutoffs; row "
                                 "retained and labelled as a stress test"),
                            estimator_class="deterministic_finite_feature",
                            seed=sec["seeds"][0]))
                        continue
                    r0, r_in, r_out = rr
                    row, _, _, _ = pairing_row(
                        cfg, "selected_log_kernel", sec["seeds"][0], lambda_ir,
                        lambda_uv, n_features, mode_cutoff, rho, r0, r_in, r_out,
                        r_min, r_max,
                        with_analytic=sec["include_analytic_reference"],
                        with_proxy=sec["include_projection_proxy"],
                        stream_label=("robustness",),
                        regulator_class=rclass, taper=taper)
                    row["window_empty"] = False
                    row["projection_raw_virasoro_residual"] = float("nan")
                    row["two_radius_raw_virasoro_residual"] = row[
                        "raw_virasoro_residual"]
                    rows.append(stamper.stamp(
                        row, estimator_class="deterministic_finite_feature",
                        seed=sec["seeds"][0]))
            print(f"  A4 {rclass} ({lambda_ir:.0e},{lambda_uv:.0e}) done")
    write_rows(path, rows, required=[
        "regulator_class", "lambda_ir", "lambda_uv", "n_features", "mode_cutoff",
        "window_empty"],
        sort_by=["regulator_class", "lambda_ir", "lambda_uv", "n_features",
                 "mode_cutoff"])
    print(f"  wrote {path.name} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
def stage_convergence(cfg, stamper, outdir, skip):
    rows_path = outdir / "predarboux_convergence_seed_rows.csv"
    fits_path = outdir / "predarboux_convergence_fits.csv"
    if should_skip(rows_path, skip) and should_skip(fits_path, skip):
        print("  [skip] A5 convergence")
        return
    sec = cfg.section("wp_a")["convergence"]
    rho = cfg.section("global")["rho_reference"]
    regs = [tuple(r) for r in sec["regulators"]]
    top_width = max(sec["n_features"])
    n_top_regs = int(sec["adaptive_top_width_regulators"])
    rows = []
    for rclass, taper in regulator_classes(cfg):
      for reg_index, (lambda_ir, lambda_uv) in enumerate(regs):
        r_min, r_max = window_of(cfg, lambda_ir, lambda_uv)
        rr = radii_for(r_min, r_max, rho)
        if rr is None:
            continue
        r0, r_in, r_out = rr
        for n_features in sec["n_features"]:
            if n_features == top_width and reg_index >= n_top_regs:
                continue  # documented adaptive design
            for mode_cutoff in sec["mode_cutoff"]:
                for seed_name in sec["seeds"]:
                    row, _, _, _ = pairing_row(
                        cfg, "selected_log_kernel", seed_name, lambda_ir, lambda_uv,
                        n_features, mode_cutoff, rho, r0, r_in, r_out, r_min, r_max,
                        with_analytic=True, stream_label=("convergence",),
                        regulator_class=rclass, taper=taper)
                    row["regulator_label"] = f"{lambda_ir:.0e}_{lambda_uv:.0e}"
                    row["adaptive_design"] = (
                        "top width restricted to the widest "
                        f"{n_top_regs} regulator pairs" if n_features == top_width
                        else "full factorial")
                    rows.append(stamper.stamp(
                        row, estimator_class="deterministic_finite_feature",
                        seed=seed_name))
            print(f"  A5 {rclass} ({lambda_ir:.0e},{lambda_uv:.0e}) N={n_features} "
                  f"done ({len(rows)} rows)")
    write_rows(rows_path, rows, required=[
        "regulator_class",
        "lambda_ir", "lambda_uv", "n_features", "mode_cutoff", "seed_name",
        "regulator_label", "pairing_relative_error", "pairing_leakage",
        "pairing_sampling_error", "raw_virasoro_residual"],
        sort_by=["regulator_class", "regulator_label", "n_features", "mode_cutoff",
                 "seed_name"])

    import pandas as pd

    df = pd.DataFrame(rows)
    fit_rows = []
    targets = {
        "pairing_leakage": "operational two-radius pairing leakage",
        "pairing_relative_error": "operational two-radius pairing relative error",
        "pairing_sampling_error":
            "finite-N sampling error about the analytic closed-kernel pairing",
        "raw_virasoro_residual": "raw filtered-Fock Sugawara residual",
    }
    for rclass in sorted(df["regulator_class"].unique()):
      sub = df[df["regulator_class"] == rclass]
      for metric, desc in targets.items():
        res = loglog_scaling_fit(
            sub["n_features"].to_numpy(), sub["mode_cutoff"].to_numpy(),
            sub[metric].to_numpy(),
            regulator_labels=sub["regulator_label"].to_numpy(), notes=desc)
        b = res.coef("log_n_features")
        a = res.coef("log_mode_cutoff")
        nt = null_test(b, res.se("log_n_features"), sec["null_width_exponent"],
                       res.dof)
        fit_rows.append(stamper.stamp(dict(
            regulator_class=rclass, metric=metric, description=desc,
            intercept=res.coef("intercept"),
            n_features_exponent=b, n_features_stderr=res.se("log_n_features"),
            n_features_ci_low=res.interval("log_n_features")[0],
            n_features_ci_high=res.interval("log_n_features")[1],
            n_features_boot_ci_low=float(
                res.boot_ci_low[res.names.index("log_n_features")]),
            n_features_boot_ci_high=float(
                res.boot_ci_high[res.names.index("log_n_features")]),
            mode_cutoff_exponent=a, mode_cutoff_stderr=res.se("log_mode_cutoff"),
            mode_cutoff_ci_low=res.interval("log_mode_cutoff")[0],
            mode_cutoff_ci_high=res.interval("log_mode_cutoff")[1],
            r_squared=res.r_squared, n_rows=res.n_rows, dof=res.dof,
            excluded_rows=res.excluded_rows, residual_sd=res.residual_sd,
            regulator_fixed_effects=sum(1 for n in res.names
                                        if n.startswith("regulator[")),
            null_width_exponent=nt["null_value"], null_t_statistic=nt["t_statistic"],
            null_p_value=nt["p_value"], null_rejected_at_5pct=nt["reject_at_5pct"],
            sufficient_joint_scaling=(
                f"Lambda^{-a / b:.4g}/N -> 0" if b and abs(b) > 1e-12 else "undefined"),
            model="log Y = c + b log N + a log Lambda + regulator fixed effects",
        ), estimator_class="deterministic_finite_feature", seed="multi"))
    write_rows(fits_path, fit_rows, required=[
        "regulator_class", "metric", "n_features_exponent", "mode_cutoff_exponent",
        "r_squared", "n_rows", "null_p_value"],
        sort_by=["regulator_class", "metric"])
    print(f"  wrote {rows_path.name} ({len(rows)}), {fits_path.name} "
          f"({len(fit_rows)})")


# ---------------------------------------------------------------------------
def stage_controls(cfg, stamper, outdir, skip):
    rows_path = outdir / "architecture_controls_seed_rows.csv"
    adm_path = outdir / "darboux_physical_admissibility.csv"
    if should_skip(rows_path, skip) and should_skip(adm_path, skip):
        print("  [skip] A6 architecture controls")
        return
    sec = cfg.section("wp_a")["controls"]
    g = cfg.section("global")
    rho = g["rho_reference"]
    lambda_ir, lambda_uv = sec["regulator"]
    r_min, r_max = window_of(cfg, lambda_ir, lambda_uv)
    r0, r_in, r_out = radii_for(r_min, r_max, rho)
    probe = sec["ward_probe"]
    charges = np.asarray(probe["charges"], dtype=float)
    positions = np.array([complex(p[0], p[1]) for p in probe["positions"]])
    probes = np.array([complex(p[0], p[1]) for p in probe["probe_positions"]])

    rows, adm_rows = [], []
    for rclass, taper in regulator_classes(cfg):
      for arch in sec["architectures"]:
        for n_features in sec["n_features"]:
            for seed_name in sec["seeds"]:
                row, omega, modes, fs = pairing_row(
                    cfg, arch, seed_name, lambda_ir, lambda_uv, n_features,
                    sec["mode_cutoff"], rho, r0, r_in, r_out, r_min, r_max,
                    params=sec["params"], with_analytic=False, with_proxy=True,
                    stream_label=("controls",), regulator_class=rclass,
                    taper=taper)
                row["control_description"] = ARCH.control_description(arch)
                row["pre_darboux_pairing_error"] = row["pairing_relative_error"]
                row["pre_darboux_leakage"] = row["pairing_leakage"]

                adm = DB.admissibility_report(omega, modes)
                row["post_darboux_pairing_error"] = adm.post_darboux_pairing_error
                row["star_compatibility_defect"] = adm.star_compatibility_defect
                row["polarization_min_eigenvalue"] = adm.polarization_min_eigenvalue
                row["polarization_positive"] = adm.polarization_positive
                row["vacuum_normalizable"] = adm.vacuum_normalizable
                row["condition_number"] = adm.condition_number
                row["symplectic_defect"] = adm.symplectic_defect

                # post-Darboux Virasoro residual: exactly the implementation check
                if np.isfinite(adm.symplectic_defect):
                    try:
                        S = DB.darboux_antisymmetric(omega, modes)
                        post_om = S @ omega @ S.T
                        rr = AL.raw_virasoro_residual(
                            post_om, modes, sec["mode_cutoff"],
                            level=g["algebra_level"], m_alg=g["algebra_mode_span"],
                            central_charge=g["central_charge"])
                        row["post_darboux_virasoro_residual"] = rr.value
                    except (np.linalg.LinAlgError, ValueError) as exc:
                        row["post_darboux_virasoro_residual"] = float("nan")
                        row["post_darboux_note"] = str(exc)
                else:
                    row["post_darboux_virasoro_residual"] = float("nan")

                # Ward probe (deterministic finite-feature): chirality-sensitive
                wd = WD.deterministic_ward(fs, charges, positions, probes)
                row["ward_error_if_available"] = wd["stress_max_absolute_error"]
                row["ward_current_max_absolute_error"] = wd[
                    "current_max_absolute_error"]
                row["ward_stress_max_absolute_error"] = wd[
                    "stress_max_absolute_error"]
                row["ward_current_mean_absolute_error"] = wd[
                    "current_mean_absolute_error"]
                row["ward_stress_mean_absolute_error"] = wd[
                    "stress_mean_absolute_error"]

                rows.append(stamper.stamp(
                    row, estimator_class="deterministic_finite_feature",
                    seed=seed_name))
                adm_rows.append(stamper.stamp(dict(
                    regulator_class=rclass,
                    architecture=arch, seed_name=seed_name, n_features=n_features,
                    mode_cutoff=sec["mode_cutoff"], lambda_ir=lambda_ir,
                    lambda_uv=lambda_uv, r0=r0, rho=rho,
                    symplectic_defect=adm.symplectic_defect,
                    star_compatibility_defect=adm.star_compatibility_defect,
                    condition_number=adm.condition_number,
                    polarization_min_eigenvalue=adm.polarization_min_eigenvalue,
                    polarization_hermiticity_defect=(
                        adm.polarization_hermiticity_defect),
                    polarization_positive=adm.polarization_positive,
                    vacuum_normalizable=adm.vacuum_normalizable,
                    post_darboux_pairing_error=adm.post_darboux_pairing_error,
                    failure=adm.failure,
                    interpretation=(
                        "Proposition 3.5: symplectic closure is universal for "
                        "nondegenerate antisymmetric forms and is NOT architecture "
                        "evidence; the discriminating columns are the involution "
                        "defect and the polarization positivity."),
                ), estimator_class="exact_symbolic", seed=seed_name))
        print(f"  A6 {rclass}/{arch} done ({len(rows)} rows)")
    write_rows(rows_path, rows, required=[
        "regulator_class",
        "architecture", "seed_name", "n_features", "pre_darboux_pairing_error",
        "pre_darboux_leakage", "raw_virasoro_residual",
        "post_darboux_pairing_error", "post_darboux_virasoro_residual",
        "star_compatibility_defect", "polarization_min_eigenvalue",
        "vacuum_normalizable", "condition_number", "ward_error_if_available"],
        sort_by=["regulator_class", "architecture", "n_features", "seed_name"])
    # --- appendix C admissibility vs width -----------------------------
    import pandas as pd

    from nnft_virasoro.fits import power_law_fit

    fit_widths = sec.get("admissibility_fit_widths", sec["n_features"])
    dfa = pd.DataFrame(adm_rows)
    fit_rows = []
    for (rclass, arch), g in dfa.groupby(["regulator_class", "architecture"]):
        g = g[g["n_features"].isin(fit_widths)]
        if g["n_features"].nunique() < 3:
            continue
        per_n = g.groupby("n_features").agg(
            star=("star_compatibility_defect", "mean"),
            herm=("polarization_hermiticity_defect", "mean"),
            mineig=("polarization_min_eigenvalue", "mean"),
            mineig_sd=("polarization_min_eigenvalue", "std"),
            pos=("polarization_positive", "mean"),
            n=("seed_name", "size")).reset_index()
        row = dict(regulator_class=rclass, architecture=arch,
                   n_widths=int(len(per_n)),
                   widths=";".join(str(int(x)) for x in per_n["n_features"]),
                   n_blocks_per_width=int(per_n["n"].iloc[0]),
                   min_eigenvalue_at_max_width=float(per_n["mineig"].iloc[-1]),
                   min_eigenvalue_sd_at_max_width=float(
                       per_n["mineig_sd"].iloc[-1]),
                   star_defect_at_max_width=float(per_n["star"].iloc[-1]),
                   hermiticity_defect_at_max_width=float(per_n["herm"].iloc[-1]),
                   polarization_positive_fraction_at_max_width=float(
                       per_n["pos"].iloc[-1]),
                   interpretation=(
                       "Appendix C. polarization_positive (min eig > 0) is the "
                       "physical statement; vacuum_normalizable additionally "
                       "imposes an ABSOLUTE Hermiticity tolerance that a "
                       "finite-width O(N^-1/2) defect cannot meet, so it must be "
                       "read with these fits, never alone."))
        for field, col in (("star_compatibility_defect", "star"),
                           ("polarization_hermiticity_defect", "herm")):
            try:
                f = power_law_fit(per_n["n_features"].to_numpy(),
                                  per_n[col].to_numpy(),
                                  notes=f"{field} ~ N^p; convergence to a "
                                        f"star-compatible, Hermitian polarization "
                                        f"requires p < 0")
                row[f"{field}_exponent"] = f.coef("exponent")
                row[f"{field}_exponent_stderr"] = f.se("exponent")
                row[f"{field}_exponent_ci_low"] = f.interval("exponent")[0]
                row[f"{field}_exponent_ci_high"] = f.interval("exponent")[1]
                row[f"{field}_r_squared"] = f.r_squared
                row[f"{field}_n_rows"] = f.n_rows
                row[f"{field}_converging"] = bool(
                    f.interval("exponent")[1] < 0.0)
            except (ValueError, np.linalg.LinAlgError) as exc:
                for suf in ("exponent", "exponent_stderr", "exponent_ci_low",
                            "exponent_ci_high", "r_squared"):
                    row[f"{field}_{suf}"] = float("nan")
                row[f"{field}_n_rows"] = 0
                row[f"{field}_converging"] = False
                row[f"{field}_note"] = str(exc)
        fit_rows.append(stamper.stamp(row, estimator_class="exact_symbolic",
                                      seed="multi"))
    if fit_rows:
        write_rows(outdir / "darboux_admissibility_vs_width.csv", fit_rows,
                   required=["regulator_class", "architecture", "n_widths",
                             "min_eigenvalue_at_max_width",
                             "star_defect_at_max_width",
                             "star_compatibility_defect_exponent",
                             "star_compatibility_defect_converging"],
                   sort_by=["regulator_class", "architecture"])
        print(f"  wrote darboux_admissibility_vs_width.csv ({len(fit_rows)} rows)")

    write_rows(adm_path, adm_rows, required=[
        "regulator_class", "architecture", "symplectic_defect",
        "star_compatibility_defect", "condition_number",
        "polarization_min_eigenvalue", "polarization_positive",
        "vacuum_normalizable"],
        sort_by=["regulator_class", "architecture", "n_features", "seed_name"])
    print(f"  wrote {rows_path.name} ({len(rows)}), {adm_path.name} "
          f"({len(adm_rows)})")


# ---------------------------------------------------------------------------
def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--stages", default="a3,a4,a5,a6")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    stages = {s.strip().lower() for s in args.stages.split(",")}
    print(f"WP-A ordered pairing audit  (config_hash={cfg.hash})")
    if "a3" in stages:
        stage_radius_plateau(cfg, stamper, outdir, figdir, args.skip_existing)
    if "a4" in stages:
        stage_robustness(cfg, stamper, outdir, args.skip_existing)
    if "a5" in stages:
        stage_convergence(cfg, stamper, outdir, args.skip_existing)
    if "a6" in stages:
        stage_controls(cfg, stamper, outdir, args.skip_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
