#!/usr/bin/env python3
"""Gate 2: pre-Darboux Virasoro stability, post-processing the Gate-1 Omega."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro import intrinsic_admissibility as IA   # noqa: E402
from nnft_virasoro import ordered_pairing as OP           # noqa: E402
from nnft_virasoro import pbw_residual as PB              # noqa: E402
from nnft_virasoro.cli import base_parser, setup          # noqa: E402
from nnft_virasoro.fits import null_test, power_law_fit   # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows  # noqa: E402

SCRIPT = "scripts/run_gate02_pbw.py"


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    rows_path = outdir / "gate02_predarboux_virasoro_seed_rows.csv"
    fits_path = outdir / "gate02_predarboux_virasoro_fits.csv"
    if should_skip(rows_path, args.skip_existing) and should_skip(fits_path,
                                                                 args.skip_existing):
        print("  [skip] gate02")
        return 0

    sec = cfg.section("gate02")
    g1 = cfg.section("gate01")
    L = int(g1["mode_cutoff"])
    modes = OP.mode_window(L)
    pairs = [tuple(x) for x in sec["mode_pairs"]]
    g1rows = pd.read_csv(outdir / "gate01_intrinsic_admissibility_seed_rows.csv")
    keep = set(sec["architectures"]) | set(sec.get("control_architectures", []))
    g1rows = g1rows[g1rows.architecture.isin(keep)]
    if args.quick:
        g1rows = g1rows[g1rows.n_features <= 30000]

    rows = []
    for _, r in g1rows.iterrows():
        om = np.load(outdir / r.omega_file)               # physical ordering
        # back to the ascending mode ordering the algebra module expects
        order = list(IA.physical_mode_order(L))
        idx = [order.index(int(m)) for m in modes]
        om_asc = om[np.ix_(idx, idx)]
        eps = PB.pairing_error_frobenius(om_asc, modes)
        d = PB.raw_pbw_residual(om_asc, modes, L, pairs)
        ratio = (d["combined_pbw_norm_excess"] / (eps + eps**2)
                 if eps > 0 else float("nan"))
        rows.append(stamper.stamp(dict(
            gate="gate02",
            spectral_regulator_class=r.spectral_regulator_class,
            architecture=r.architecture, seed_name=r.seed_name,
            n_features=int(r.n_features), mode_cutoff=L,
            lambda_ir=r.lambda_ir, lambda_uv=r.lambda_uv,
            n_mode_pairs=d["n_mode_pairs"], n_pbw_basis=d["n_basis"],
            pairing_error_frobenius=eps,
            raw_virasoro_residual_pbw=d["combined_pbw_norm_raw"],
            raw_scalar_coefficient_norm=d["scalar_norm_raw"],
            raw_quadratic_coefficient_norm=d["quadratic_norm_raw"],
            excess_virasoro_residual_pbw=d["combined_pbw_norm_excess"],
            excess_scalar_coefficient_norm=d["scalar_norm_excess"],
            excess_quadratic_coefficient_norm=d["quadratic_norm_excess"],
            edge_baseline_combined_norm=d["edge_baseline_combined_norm"],
            residual_over_pairing_plus_square=ratio),
            estimator_class="deterministic_finite_feature", seed=r.seed_name))
    write_rows(rows_path, rows, required=[
        "gate", "spectral_regulator_class", "architecture", "seed_name",
        "n_features", "pairing_error_frobenius", "raw_virasoro_residual_pbw",
        "excess_virasoro_residual_pbw", "residual_over_pairing_plus_square"],
        sort_by=["spectral_regulator_class", "architecture", "n_features",
                 "seed_name"])

    df = pd.DataFrame(rows)
    fit_rows = []
    for (rclass, arch), sub in df.groupby(["spectral_regulator_class",
                                           "architecture"]):
        per = (sub.groupby("n_features")
               .agg(eps=("pairing_error_frobenius", "mean"),
                    rex=("excess_virasoro_residual_pbw", "mean"),
                    rraw=("raw_virasoro_residual_pbw", "mean"),
                    ratio=("residual_over_pairing_plus_square", "mean"),
                    n=("seed_name", "size")).reset_index())
        row = dict(gate="gate02", spectral_regulator_class=rclass,
                   architecture=arch, n_widths=int(len(per)),
                   n_seeds_per_width=int(per["n"].iloc[0]),
                   fit_window=";".join(str(int(x)) for x in per["n_features"]),
                   eps_at_max_width=float(per["eps"].iloc[-1]),
                   excess_at_max_width=float(per["rex"].iloc[-1]),
                   raw_at_max_width=float(per["rraw"].iloc[-1]),
                   ratio_at_max_width=float(per["ratio"].iloc[-1]),
                   ratio_max=float(per["ratio"].max()),
                   ratio_bounded=bool(np.isfinite(per["ratio"]).all()
                                      and per["ratio"].max() < 1e3))
        for label, col in (("b_omega", "eps"), ("b_R", "rex")):
            y = per[col].to_numpy(dtype=float)
            if len(per) >= 3 and np.all(y > 0):
                f = power_law_fit(per["n_features"].to_numpy(), y,
                                  notes=f"{col} ~ N^p")
                nt = null_test(f.coef("exponent"), f.se("exponent"), -0.5, f.dof)
                row[f"{label}_exponent"] = f.coef("exponent")
                row[f"{label}_stderr"] = f.se("exponent")
                row[f"{label}_ci_low"] = f.interval("exponent")[0]
                row[f"{label}_ci_high"] = f.interval("exponent")[1]
                row[f"{label}_r_squared"] = f.r_squared
                row[f"{label}_n_fit_rows"] = f.n_rows
                row[f"{label}_p_null_minus_half"] = nt["p_value"]
                row[f"{label}_null_rejected"] = nt["reject_at_5pct"]
                row[f"{label}_fit_status"] = "fitted"
            else:
                for suf in ("exponent", "stderr", "ci_low", "ci_high",
                            "r_squared", "p_null_minus_half"):
                    row[f"{label}_{suf}"] = float("nan")
                row[f"{label}_n_fit_rows"] = 0
                row[f"{label}_null_rejected"] = None
                row[f"{label}_fit_status"] = "not fitted: non-positive series"
        fit_rows.append(stamper.stamp(row, estimator_class="exact_symbolic",
                                      seed="multi"))
    write_rows(fits_path, fit_rows, required=[
        "gate", "spectral_regulator_class", "architecture", "b_omega_exponent",
        "b_R_exponent", "ratio_bounded", "ratio_max"],
        sort_by=["spectral_regulator_class", "architecture"])
    # ---- controlled epsilon sweep of the stability theorem --------------
    # Five decades of EPSILON at fixed mode cutoff, as distinct from the
    # architecture grid's five decades of WIDTH. Recorded as a release artifact
    # so the quoted plateau is reproducible from the CSV rather than from a
    # test assertion.
    es = sec.get("epsilon_sweep")
    sweep_rows = []
    if es:
        Ls = int(es["mode_cutoff"])
        modes_s = OP.mode_window(Ls)
        can_s = OP.canonical_pairing(modes_s)
        pairs_s = [tuple(x) for x in sec["mode_pairs"]]
        for direction in range(int(es["n_directions"])):
            drng = np.random.default_rng(int(es["direction_seed"]) + direction)
            X = drng.standard_normal(can_s.shape)
            A = (X - X.T) / np.linalg.norm(X - X.T)
            for eps_t in es["epsilons"]:
                om = can_s + float(eps_t) * A
                eps = PB.pairing_error_frobenius(om, modes_s)
                exc = PB.raw_pbw_residual(om, modes_s, Ls,
                                          pairs_s)["combined_pbw_norm_excess"]
                sweep_rows.append(stamper.stamp(dict(
                    gate="gate02", check="stability_theorem_epsilon_sweep",
                    sweep_variable="epsilon", direction_index=int(direction),
                    direction_seed=int(es["direction_seed"]) + direction,
                    mode_cutoff=Ls, epsilon_target=float(eps_t),
                    pairing_error_frobenius=float(eps),
                    excess_virasoro_residual_pbw=float(exc),
                    residual_over_pairing_plus_square=float(
                        exc / (eps + eps**2))),
                    estimator_class="exact_symbolic", seed=None))
        sd = pd.DataFrame(sweep_rows)
        per = sd.groupby("direction_index")["residual_over_pairing_plus_square"]
        span = float((per.max() / per.min()).max())
        ref = sd[sd.direction_index == 0]["residual_over_pairing_plus_square"]
        for r in sweep_rows:
            r["worst_span_over_five_decades"] = span
            r["reference_direction_low"] = float(ref.min())
            r["reference_direction_high"] = float(ref.max())
            r["all_directions_low"] = float(
                sd.residual_over_pairing_plus_square.min())
            r["all_directions_high"] = float(
                sd.residual_over_pairing_plus_square.max())
            r["ratio_is_bounded"] = bool(span < 2.0)
        write_rows(outdir / "gate02_stability_epsilon_sweep.csv", sweep_rows,
                   required=["gate", "check", "sweep_variable",
                             "direction_index", "epsilon_target",
                             "residual_over_pairing_plus_square",
                             "ratio_is_bounded"],
                   sort_by=["direction_index", "epsilon_target"])
        print(f"  epsilon sweep: reference direction "
              f"{ref.min():.4f}-{ref.max():.4f}; over "
              f"{int(es['n_directions'])} directions "
              f"{sd.residual_over_pairing_plus_square.min():.4f}-"
              f"{sd.residual_over_pairing_plus_square.max():.4f}; "
              f"worst span across five decades {span:.3f}x")
    print(f"  wrote {rows_path.name} ({len(rows)}), {fits_path.name} "
          f"({len(fit_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
