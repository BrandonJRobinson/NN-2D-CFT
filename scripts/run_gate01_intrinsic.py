#!/usr/bin/env python3
"""Gate 1: intrinsic physical admissibility of the pre-Darboux pairing.

The August release never persisted Omega, so this recomputes the pairing over
the frozen controls grid and PERSISTS the matrices (Gate 2 post-processes them).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro import architectures as ARCH          # noqa: E402
from nnft_virasoro import intrinsic_admissibility as IA  # noqa: E402
from nnft_virasoro import ordered_pairing as OP          # noqa: E402
from nnft_virasoro.cli import base_parser, setup         # noqa: E402
from nnft_virasoro.config import rng_for                 # noqa: E402
from nnft_virasoro.fits import power_law_fit             # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows  # noqa: E402

SCRIPT = "scripts/run_gate01_intrinsic.py"


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    rows_path = outdir / "gate01_intrinsic_admissibility_seed_rows.csv"
    fits_path = outdir / "gate01_intrinsic_admissibility_fits.csv"
    if should_skip(rows_path, args.skip_existing) and should_skip(fits_path,
                                                                 args.skip_existing):
        print("  [skip] gate01")
        return 0

    sec = cfg.section("gate01")
    g = cfg.section("global")
    lir, luv = sec["lambda_ir"], sec["lambda_uv"]
    L = int(sec["mode_cutoff"])
    r0, rho = float(sec["r0"]), float(sec["rho"])
    rin, rout = r0 / np.sqrt(rho), r0 * np.sqrt(rho)
    modes = OP.mode_window(L)
    widths = sec["n_features"][:2] if args.quick else sec["n_features"]
    seeds = sec["seeds"][:1] if args.quick else sec["seeds"]
    archs = sec["architectures"][:2] if args.quick else sec["architectures"]
    omdir = outdir / "omega"
    omdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rclass, taper in g["spectral_regulator_classes"].items():
        for arch in archs:
            for n_features in widths:
                for seed_name in seeds:
                    rng = rng_for(cfg.section("named_seeds"), seed_name, "gate01",
                                  arch, n_features, rclass)
                    params = dict(sec.get("params", {}))
                    params["regulator_taper"] = float(taper)
                    fs = ARCH.sample_architecture(arch, rng, n_features, lir, luv,
                                                  params=params)
                    res = OP.pairing_empirical_exact(
                        modes, rin, rout, r0, fs.kabs, fs.theta, fs.prefactor,
                        lir, luv, idx_a=fs.idx_a, idx_b=fs.idx_b,
                        log_measure=fs.log_measure)
                    omega = res.omega
                    post = ARCH.mode_postmap(arch, modes, rng, params=params)
                    if post is not None:
                        omega = post @ omega @ post.T

                    om_phys = IA.reorder_to_physical(omega, modes, L)
                    if sec.get("persist_omega", True):
                        tag = f"{rclass}__{arch}__N{n_features}__{seed_name}"
                        np.save(omdir / f"{tag}.npy", om_phys)

                    rep = IA.intrinsic_diagnostics(om_phys)
                    met = OP.pairing_metrics(omega, modes)
                    rows.append(stamper.stamp(dict(
                        gate="gate01",
                        spectral_regulator_class=rclass, regulator_taper=taper,
                        architecture=arch, seed_name=seed_name,
                        n_features=int(n_features), mode_cutoff=L,
                        lambda_ir=lir, lambda_uv=luv, r0=r0, rho=rho,
                        r_in=rin, r_out=rout,
                        omega_file=f"omega/{rclass}__{arch}__N{n_features}__{seed_name}.npy",
                        operational_pairing_error=met["pairing_relative_error"],
                        operational_pairing_leakage=met["pairing_leakage"],
                        # IntrinsicReport also carries mode_cutoff; the explicit
                        # column above wins.
                        **{k: v for k, v in rep.as_dict().items()
                           if k != "mode_cutoff"}),
                        estimator_class="deterministic_finite_feature",
                        seed=seed_name))
            print(f"  gate01 {rclass}/{arch} done ({len(rows)} rows)")

    write_rows(rows_path, rows, required=[
        "gate", "spectral_regulator_class", "architecture", "seed_name",
        "n_features", "mode_cutoff", "delta_star", "delta_A", "delta_H",
        "lambda_min_normalized", "delta_iso", "physically_admissible",
        "principal_symplectic_defect", "principal_involution_defect"],
        sort_by=["spectral_regulator_class", "architecture", "n_features",
                 "seed_name"])

    # ---- width fits, only where a power law is justified ----------------
    import pandas as pd

    df = pd.DataFrame(rows)
    fit_rows = []
    min_w = int(sec.get("fit_min_widths", 3))
    for (rclass, arch), sub in df.groupby(["spectral_regulator_class",
                                           "architecture"]):
        per = (sub.groupby("n_features")
               .agg(star=("delta_star", "mean"), A=("delta_A", "mean"),
                    H=("delta_H", "mean"), iso=("delta_iso", "mean"),
                    lam=("lambda_min_normalized", "mean"),
                    lam_sd=("lambda_min_normalized", "std"),
                    adm=("physically_admissible", "mean"),
                    n=("seed_name", "size")).reset_index())
        row = dict(gate="gate01", spectral_regulator_class=rclass,
                   architecture=arch, n_widths=int(len(per)),
                   widths=";".join(str(int(x)) for x in per["n_features"]),
                   n_seeds_per_width=int(per["n"].iloc[0]),
                   delta_star_at_max_width=float(per["star"].iloc[-1]),
                   delta_A_at_max_width=float(per["A"].iloc[-1]),
                   delta_H_at_max_width=float(per["H"].iloc[-1]),
                   delta_iso_at_max_width=float(per["iso"].iloc[-1]),
                   lambda_min_at_max_width=float(per["lam"].iloc[-1]),
                   lambda_min_sd_at_max_width=float(per["lam_sd"].iloc[-1]),
                   admissible_fraction_at_max_width=float(per["adm"].iloc[-1]))
        for field, col in (("delta_star", "star"), ("delta_A", "A"),
                           ("delta_H", "H"), ("delta_iso", "iso")):
            y = per[col].to_numpy(dtype=float)
            # a power law is only meaningful for a strictly positive, varying,
            # and actually decreasing series; otherwise we say so rather than
            # forcing every control into a decay model (mission section 3).
            positive = np.all(np.isfinite(y)) and np.all(y > 0)
            varying = positive and (y.max() / y.min() > 1.05)
            if len(per) >= min_w and positive and varying:
                f = power_law_fit(per["n_features"].to_numpy(), y,
                                  notes=f"{field} ~ N^p")
                row[f"{field}_exponent"] = f.coef("exponent")
                row[f"{field}_exponent_stderr"] = f.se("exponent")
                row[f"{field}_exponent_ci_low"] = f.interval("exponent")[0]
                row[f"{field}_exponent_ci_high"] = f.interval("exponent")[1]
                row[f"{field}_r_squared"] = f.r_squared
                row[f"{field}_fit_status"] = "fitted"
                row[f"{field}_converging"] = bool(f.interval("exponent")[1] < 0)
            else:
                for suf in ("exponent", "exponent_stderr", "exponent_ci_low",
                            "exponent_ci_high", "r_squared"):
                    row[f"{field}_{suf}"] = float("nan")
                row[f"{field}_converging"] = False
                row[f"{field}_fit_status"] = (
                    "not fitted: series is constant to within 5% "
                    "(no power law to estimate)" if positive and not varying
                    else "not fitted: non-positive or non-finite values")
        fit_rows.append(stamper.stamp(row, estimator_class="exact_symbolic",
                                      seed="multi"))
    write_rows(fits_path, fit_rows, required=[
        "gate", "spectral_regulator_class", "architecture", "n_widths",
        "delta_star_at_max_width", "lambda_min_at_max_width",
        "delta_star_fit_status"],
        sort_by=["spectral_regulator_class", "architecture"])
    print(f"  wrote {rows_path.name} ({len(rows)}), {fits_path.name} "
          f"({len(fit_rows)}), omega/ ({len(rows)} matrices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
