#!/usr/bin/env python3
"""Gate 7: NS sector with an explicit, independent randomized Cauchy architecture.

The native NS construction derives the fermion kernel from the bosonic seed,
``K_psipsi = -d_z K_phiphi``. Its width and randomness are the boson's, so it
cannot on its own establish that the NS claim rests on a Grassmann architecture.

This gate is ADDITIVE, as recorded in IMPLEMENTATION_MAP section 4: the native
construction is retained as a cross-check and an explicit randomized Cauchy
factorization is added alongside it, with its own width, its own randomness and
its own declared proposal. Both are carried through the same ordered pairing and
the same super-Virasoro residuals, and reported side by side.

Everything here is pre-Darboux. Canonicalized residuals are computed only as an
implementation check and are never used as architecture evidence (mission
rule 3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro import algebra as AL                      # noqa: E402
from nnft_virasoro import architectures as AR                # noqa: E402
from nnft_virasoro import fits as FT                         # noqa: E402
from nnft_virasoro import ns_cauchy as NC                    # noqa: E402
from nnft_virasoro import ns_fermion as NF                   # noqa: E402
from nnft_virasoro import ordered_pairing as OP              # noqa: E402
from nnft_virasoro.cli import base_parser, setup             # noqa: E402
from nnft_virasoro.config import rng_for                     # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows   # noqa: E402

SCRIPT = "scripts/run_gate07_ns.py"


def kernel_diagnostics(cfg, sec, stamper, r_in, r_out):
    """Pointwise unbiasedness and the finite-variance condition.

    The variance condition is scanned across proposals, including ones that
    violate it, so the condition is demonstrated rather than merely asserted.
    """
    cy = sec["cauchy"]
    rows = []
    z = r_out + 0j
    w = r_in * np.exp(0.7j)
    exact = 1.0 / (z - w)
    for q in cy["variance_scan_q"]:
        P = NC.geometric_proposal(float(q), int(cy["n_max"]))
        cond = NC.finite_variance_condition(float(q), r_in, r_out)
        # EXACT variance against the truncation: the demonstration that the
        # condition is real. A sample s.d. cannot show this (see the docstring
        # of ns_cauchy.estimator_variance).
        vs = {}
        for nm in cy["variance_scan_n_max"]:
            v = NC.estimator_variance(z, w, NC.geometric_proposal(float(q), int(nm)))
            vs[f"exact_variance_nmax_{nm}"] = float(v)
        seq = [vs[f"exact_variance_nmax_{nm}"] for nm in cy["variance_scan_n_max"]]
        finite_seq = [x for x in seq if np.isfinite(x)]
        vs["variance_is_flat_in_truncation"] = bool(
            len(finite_seq) == len(seq)
            and max(finite_seq) <= 1.01 * min(finite_seq))
        vs["variance_growth_over_truncation"] = (
            float(seq[-1] / seq[0]) if np.isfinite(seq[-1]) and seq[0] > 0
            else float("inf"))
        for n_features in cy["kernel_probe_widths"]:
            rng = rng_for(cfg.section("named_seeds"), "ns_s01", "gate07kern",
                          q, n_features)
            vals = np.array([
                complex(NC.cauchy_kernel_estimate(
                    z, w, NC.sample_ns_modes(rng, int(n_features), P), P))
                for _ in range(int(cy["kernel_probe_repeats"]))])
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(vals.size)
            rows.append(stamper.stamp(dict(
                gate="gate07", architecture="randomized_cauchy",
                check="cauchy_kernel_unbiasedness",
                proposal="geometric", q=float(q), n_max=int(cy["n_max"]),
                n_features=int(n_features), n_repeats=int(vals.size),
                z_abs=float(abs(z)), w_abs=float(abs(w)),
                exact_real=float(exact.real), exact_imag=float(exact.imag),
                mean_real=float(mean.real), mean_imag=float(mean.imag),
                bias_abs=float(abs(mean - exact)),
                bias_in_sem=float(abs(mean - exact) / sem) if sem > 0 else np.nan,
                sample_sd=float(vals.std(ddof=1)),
                unbiased_within_3sem=bool(sem > 0
                                          and abs(mean - exact) <= 3 * sem),
                **vs,
                **{k: v for k, v in cond.items() if k != "q"}),
                estimator_class="deterministic_finite_feature", seed="ns_s01"))
    return rows


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    sec = cfg.section("gate07")
    kern_path = outdir / "ns_kernel_seed_rows.csv"
    if should_skip(kern_path, args.skip_existing):
        print("  [skip] gate07")
        return 0

    lir, luv = float(sec["lambda_ir"]), float(sec["lambda_uv"])
    taper = float(sec["taper"])
    r0, rho = float(sec["r0"]), float(sec["rho"])
    _, r_in, r_out = OP.radii_from_window(r0 / rho, r0 * rho, rho, r0=r0)
    bmodes = OP.mode_window(int(sec["boson_mode_cutoff"]))
    tt = NF.fermion_mode_window(int(sec["fermion_cutoff"]))
    cy = sec["cauchy"]
    P = NC.geometric_proposal(float(cy["q"]), int(cy["n_max"]))
    cond = NC.finite_variance_condition(float(cy["q"]), r_in, r_out)
    if not cond["finite_variance"]:
        raise SystemExit(f"configured proposal has infinite variance: {cond}")

    widths = list(sec["n_features"])
    seeds = list(sec["seeds"])
    archs = list(sec["architectures"])
    if args.quick:
        widths, seeds, archs = widths[:3], seeds[:2], archs[:1]

    print(f"  finite-variance condition: q={cy['q']} > "
          f"(r_in/r_out)^2={cond['radius_ratio_squared']:.4f}  OK", flush=True)
    print("  kernel diagnostics (unbiasedness + variance scan)...", flush=True)
    krows = kernel_diagnostics(cfg, sec, stamper, r_in, r_out)
    write_rows(kern_path, krows, required=[
        "gate", "architecture", "check", "q", "n_features", "bias_abs",
        "unbiased_within_3sem", "finite_variance",
        "variance_is_flat_in_truncation", "variance_growth_over_truncation"],
        sort_by=["q", "n_features"])

    pair_rows, alg_rows = [], []
    for arch in archs:
        for n_features in widths:
            for seed_name in seeds:
                rng = rng_for(cfg.section("named_seeds"), seed_name, "gate07",
                              arch, n_features)
                fs = AR.sample_architecture(arch, rng, int(n_features), lir, luv,
                                            params={"regulator_taper": taper})
                # bosonic pairing -- shared by both NS constructions
                bres = OP.pairing_empirical_exact(
                    bmodes, r_in, r_out, r0, fs.kabs, fs.theta, fs.prefactor,
                    lir, luv, idx_a=fs.idx_a, idx_b=fs.idx_b,
                    log_measure=fs.log_measure)

                # --- the two NS constructions -------------------------
                eta_native = NF.fermionic_pairing_empirical_exact(
                    tt, r_in, r_out, r0, fs.kabs, fs.theta, lir, luv,
                    log_measure=fs.log_measure).omega
                modes_c = NC.sample_ns_modes(rng, int(n_features), P)
                cres = NC.cauchy_pairing_exact(tt, modes_c, P)
                eta_cauchy = cres.omega
                # Independent evaluation path for the SAME object. Cost is
                # O(q^2 N), so it is capped: it validates the closed form, and
                # the smaller widths establish that just as well.
                if int(n_features) <= int(cy["quadrature_max_width"]):
                    qres = NC.cauchy_pairing_quadrature(
                        tt, modes_c, P, r_in, r_out,
                        q=int(cy["quadrature_points"]), r0=r0)
                    path_gap = float(np.max(np.abs(cres.omega - qres.omega)))
                else:
                    path_gap = float("nan")
                pred = float(np.nanmax(
                    NC.car_error_prediction(tt, P, int(n_features))))

                for label, eta in (("native_derived_from_boson", eta_native),
                                   ("randomized_cauchy", eta_cauchy)):
                    met = NF.car_metrics(eta, tt)
                    row = dict(
                        gate="gate07", ns_construction=label, architecture=arch,
                        seed_name=seed_name, n_features=int(n_features),
                        fermion_cutoff=int(sec["fermion_cutoff"]),
                        r_in=r_in, r_out=r_out, r0=r0, rho=rho,
                        lambda_ir=lir, lambda_uv=luv, regulator_taper=taper,
                        derived_from_boson=bool(label.startswith("native")),
                        independent_width=not label.startswith("native"),
                        **met)
                    if label == "randomized_cauchy":
                        row.update(proposal="geometric", q=float(cy["q"]),
                                   exact_vs_quadrature_max_gap=path_gap,
                                   predicted_car_sd=pred,
                                   **{k: v for k, v in cond.items() if k != "q"})
                    pair_rows.append(stamper.stamp(
                        row, estimator_class="deterministic_finite_feature",
                        seed=seed_name))

                    raw = AL.super_virasoro_residuals(
                        bres.omega, bmodes, eta, tt,
                        int(sec["boson_mode_cutoff"]),
                        int(sec["fermion_cutoff"]),
                        level=int(sec["algebra_level"]),
                        r_alg=int(sec["r_alg"]))
                    alg_rows.append(stamper.stamp(dict(
                        gate="gate07", ns_construction=label, architecture=arch,
                        seed_name=seed_name, n_features=int(n_features),
                        boson_mode_cutoff=int(sec["boson_mode_cutoff"]),
                        fermion_cutoff=int(sec["fermion_cutoff"]),
                        algebra_level=int(sec["algebra_level"]),
                        r_alg=int(sec["r_alg"]),
                        # PRE-DARBOUX residuals; no canonicalization is applied
                        pre_darboux=True,
                        raw_LG_residual=float(raw["LG_residual"]),
                        raw_GG_residual=float(raw["GG_residual"]),
                        raw_central_charge_fit=float(raw["central_charge_fit"]),
                        raw_central_charge_spread=float(
                            raw["central_charge_fit_spread"]),
                        target_central_charge=1.5,
                        car_relative_error=float(
                            NF.car_metrics(eta, tt)
                            ["fermionic_pairing_relative_error"])),
                        estimator_class="deterministic_finite_feature",
                        seed=seed_name))
        print(f"    {arch}: {len(widths)} widths x {len(seeds)} seeds done",
              flush=True)

    write_rows(outdir / "ns_pairing_seed_rows.csv", pair_rows, required=[
        "gate", "ns_construction", "architecture", "seed_name", "n_features",
        "fermionic_pairing_relative_error", "derived_from_boson"],
        sort_by=["ns_construction", "architecture", "n_features", "seed_name"])
    write_rows(outdir / "ns_super_virasoro_seed_rows.csv", alg_rows, required=[
        "gate", "ns_construction", "architecture", "seed_name", "n_features",
        "raw_LG_residual", "raw_GG_residual", "raw_central_charge_fit",
        "pre_darboux"],
        sort_by=["ns_construction", "architecture", "n_features", "seed_name"])

    # ---- N^{-1/2} convergence fits, per construction --------------------
    fit_rows = []
    dp = pd.DataFrame(pair_rows)
    da = pd.DataFrame(alg_rows)
    for label, frame, col in (
            ("car_relative_error", dp, "fermionic_pairing_relative_error"),
            ("LG_residual", da, "raw_LG_residual"),
            ("GG_residual", da, "raw_GG_residual")):
        for (cons, arch), g in frame.groupby(["ns_construction", "architecture"]):
            agg = g.groupby("n_features")[col].mean()
            if agg.size < 3:
                continue
            res = FT.power_law_fit(agg.index.to_numpy(float), agg.to_numpy(float),
                                   notes=f"{label} vs N, {cons}, {arch}")
            null = FT.null_test(res.coef("exponent"), res.se("exponent"), -0.5,
                                dof=max(agg.size - 2, 1))
            lo, hi = res.interval("exponent")
            fit_rows.append(stamper.stamp(dict(
                gate="gate07", quantity=label, ns_construction=cons,
                architecture=arch, n_widths=int(agg.size),
                exponent=float(res.coef("exponent")),
                exponent_stderr=float(res.se("exponent")),
                exponent_ci_low=float(lo), exponent_ci_high=float(hi),
                value_at_smallest_N=float(agg.iloc[0]),
                value_at_largest_N=float(agg.iloc[-1]),
                smallest_N=int(agg.index[0]), largest_N=int(agg.index[-1]),
                **{f"null_half_{k}": v for k, v in null.items()}),
                estimator_class="deterministic_finite_feature", seed="multi"))
    write_rows(outdir / "ns_super_virasoro_fits.csv", fit_rows, required=[
        "gate", "quantity", "ns_construction", "architecture", "exponent",
        "exponent_stderr", "null_half_p_value"],
        sort_by=["quantity", "ns_construction", "architecture"])

    print(f"  wrote kernel({len(krows)}) pairing({len(pair_rows)}) "
          f"algebra({len(alg_rows)}) fits({len(fit_rows)})")
    for r in fit_rows:
        if r["architecture"] == "selected_log_kernel":
            print(f"    {r['quantity']:>18} {r['ns_construction']:>26}: "
                  f"exponent {r['exponent']:+.3f} +/- {r['exponent_stderr']:.3f} "
                  f"(vs -1/2: p={r['null_half_p_value']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
