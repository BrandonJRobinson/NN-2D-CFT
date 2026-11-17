#!/usr/bin/env python3
"""WP-E: extended modewise central-offset test (eq. 5.5 / eq. 5.8).

Two objects are computed and never conflated:

``canonical_truncated``  ``exact_symbolic`` null: the modewise anomaly of the
    *canonical* truncated Sugawara algebra at cutoff Lambda.  If truncation
    alone produced the released alternating offset, it would appear here.
``raw_empirical``        the modewise anomaly of the truncated algebra built
    from the **measured** finite-feature pairing, which is what the manuscript's
    "raw empirical release" refers to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnft_virasoro import algebra as AL  # noqa: E402
from nnft_virasoro import architectures as ARCH  # noqa: E402
from nnft_virasoro import ordered_pairing as OP  # noqa: E402
from nnft_virasoro.cli import base_parser, setup  # noqa: E402
from nnft_virasoro.config import rng_for  # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_json, write_rows  # noqa: E402
from nnft_virasoro.kernels import physical_window  # noqa: E402
from nnft_virasoro.sampling import block_statistics, bootstrap_ci  # noqa: E402
from nnft_virasoro.superalgebra import Pairings  # noqa: E402

SCRIPT = "scripts/run_central_offset.py"


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    csv_path = outdir / "central_offset_mode_extension.csv"
    json_path = outdir / "central_offset_fit.json"
    if should_skip(csv_path, args.skip_existing) and should_skip(
            json_path, args.skip_existing):
        print("  [skip] WP-E")
        return 0

    sec = cfg.section("wp_e")
    g = cfg.section("global")
    lambda_ir, lambda_uv = sec["regulator"]
    r_min, r_max = physical_window(lambda_ir, lambda_uv,
                                   margin_uv=g["window_margin_uv"],
                                   margin_ir=g["window_margin_ir"])
    rho = g["rho_reference"]
    r0 = float(np.sqrt(r_min * r_max))
    r_in, r_out = r0 / np.sqrt(rho), r0 * np.sqrt(rho)
    rclasses = [(str(k), float(v)) for k, v in g["regulator_classes"].items()]
    A_ref = float(sec["legacy_offset_reference"])
    canonical = Pairings.canonical()

    rows = []
    # ---- exact_symbolic null: canonical truncated algebra ----------------
    for lam in sec["mode_cutoffs"]:
        for m in sec["modes"]:
            if lam < m + 2:
                continue
            cs = complex(AL.central_scalar(m, lam, canonical)).real
            tgt = m * (m * m - 1) / 12.0
            rows.append(stamper.stamp(dict(
                source="canonical_truncated", regulator_class="not_applicable",
                architecture="canonical_pairing", seed_name="", n_features=0,
                m=int(m), **{"lambda": int(lam)},
                K_max=AL.k_max(lam, m, -m), central_scalar=cs,
                canonical_target=tgt, signed_offset=cs - tgt,
                absolute_offset=abs(cs - tgt), modewise_c=cs / tgt,
                model_prediction=1.0 + ((-1.0) ** m) * A_ref / tgt,
                fit_residual=float("nan"),
                edge_contaminated=bool(AL.k_max(lam, m, -m) < 0),
            ), estimator_class="exact_symbolic", seed=None))

    # ---- raw empirical modewise anomaly ---------------------------------
    for rclass, taper in rclasses:
        params = {"regulator_taper": taper}
        for lam in sec["mode_cutoffs"]:
            modes = OP.mode_window(lam)
            for n_features in sec["n_features"]:
                for seed_name in sec["seeds"]:
                    rng = rng_for(cfg.section("named_seeds"), seed_name, "offset",
                                  lam, n_features, rclass)
                    fs = ARCH.sample_architecture("selected_log_kernel", rng,
                                                  n_features, lambda_ir, lambda_uv,
                                                  params=params)
                    res = OP.pairing_empirical_exact(
                        modes, r_in, r_out, r0, fs.kabs, fs.theta, fs.prefactor,
                        lambda_ir, lambda_uv, idx_a=fs.idx_a, idx_b=fs.idx_b,
                        log_measure=fs.log_measure)
                    pair = Pairings.from_matrices(
                        modes=[int(x) for x in modes], omega=res.omega)
                    for m in sec["modes"]:
                        if lam < m + 2:
                            continue
                        cs = complex(AL.central_scalar(m, lam, pair)).real
                        tgt = m * (m * m - 1) / 12.0
                        rows.append(stamper.stamp(dict(
                            source="raw_empirical", regulator_class=rclass,
                            architecture="selected_log_kernel", seed_name=seed_name,
                            n_features=n_features, m=int(m), **{"lambda": int(lam)},
                            K_max=AL.k_max(lam, m, -m), central_scalar=cs,
                            canonical_target=tgt, signed_offset=cs - tgt,
                            absolute_offset=abs(cs - tgt), modewise_c=cs / tgt,
                            model_prediction=1.0 + ((-1.0) ** m) * A_ref / tgt,
                            fit_residual=float("nan"),
                            edge_contaminated=bool(AL.k_max(lam, m, -m) < 0),
                            lambda_ir=lambda_ir, lambda_uv=lambda_uv, r0=r0, rho=rho,
                        ), estimator_class="deterministic_finite_feature",
                            seed=seed_name))
                print(f"  WP-E {rclass} lambda={lam} N={n_features} done "
                      f"({len(rows)} rows)")

    write_rows(csv_path, rows, required=[
        "source", "m", "lambda", "K_max", "central_scalar", "canonical_target",
        "signed_offset", "absolute_offset", "modewise_c", "model_prediction",
        "fit_residual", "edge_contaminated"],
        sort_by=["source", "regulator_class", "lambda", "n_features", "m",
                 "seed_name"])

    # ---- fits -----------------------------------------------------------
    import pandas as pd

    from nnft_virasoro.fits import power_law_fit

    df = pd.DataFrame(rows)
    fits = {}
    for source in sorted(df["source"].unique()):
        sub = df[df["source"] == source]
        groups = ([("all", sub)] if source == "canonical_truncated"
                  else [(rc, sub[sub["regulator_class"] == rc])
                        for rc in sorted(sub["regulator_class"].unique())])
        # Also fit the legacy scope m in {2,3,4} on its own, so the frozen
        # recomputation can be compared like-for-like with the released value.
        groups += [(f"{g}|m2to4", d[d["m"].isin([2, 3, 4])])
                   for g, d in list(groups)]
        groups = [(g, d) for g, d in groups if len(d)]
        for gname, gsub in groups:
            per_m = gsub.groupby("m").agg(
                signed=("signed_offset", "mean"),
                signed_sd=("signed_offset", "std"),
                absolute=("absolute_offset", "mean"),
                target=("canonical_target", "first"),
                n=("signed_offset", "size")).reset_index()
            sign = (-1.0) ** per_m["m"].to_numpy()
            de_signed = sign * per_m["signed"].to_numpy()
            A_hat = float(np.mean(de_signed))
            A_sd = float(np.std(de_signed, ddof=1)) if len(de_signed) > 1 else 0.0
            A_lo, A_hi = bootstrap_ci(de_signed, seed=20260730)
            # model 1: alternating constant offset  -> residuals of c_m
            pred_c = 1.0 + sign * A_hat / per_m["target"].to_numpy()
            obs_c = gsub.groupby("m")["modewise_c"].mean().to_numpy()
            resid = obs_c - pred_c
            ss_tot = float(np.sum((obs_c - obs_c.mean()) ** 2))
            # model 2: |c_m - 1| ~ m^-3
            dev = np.abs(obs_c - 1.0)
            if int(np.sum(dev > 0)) >= 3:
                pw = power_law_fit(
                    per_m["m"].to_numpy(), dev,
                    notes="|c_m - 1| ~ m^p; alternating-offset model predicts p = -3")
                decay = dict(decay_exponent=pw.coef("exponent"),
                             decay_exponent_stderr=pw.se("exponent"),
                             decay_exponent_ci=list(pw.interval("exponent")),
                             decay_r_squared=pw.r_squared, decay_n_rows=pw.n_rows,
                             decay_excluded_rows=pw.excluded_rows,
                             decay_status="fitted")
            else:
                decay = dict(decay_exponent=float("nan"),
                             decay_exponent_stderr=float("nan"),
                             decay_exponent_ci=[float("nan"), float("nan")],
                             decay_r_squared=float("nan"),
                             decay_n_rows=0,
                             decay_excluded_rows=int(len(dev)),
                             decay_status=(
                                 "not fitted: |c_m - 1| is identically zero, so the "
                                 "m^-3 decay model has no support. This is itself "
                                 "the result -- truncation alone produces no "
                                 "modewise offset."))
            # null alternatives
            null_zero = float(np.sqrt(np.mean(per_m["signed"].to_numpy() ** 2)))
            null_const = float(np.std(per_m["signed"].to_numpy(), ddof=1)
                               if len(per_m) > 1 else 0.0)
            fits[f"{source}|{gname}"] = dict(
                source=source, group=gname, n_modes=int(len(per_m)),
                modes=[int(x) for x in per_m["m"]],
                alternating_offset_A=A_hat, alternating_offset_sd=A_sd,
                alternating_offset_ci=[A_lo, A_hi],
                legacy_offset_reference=A_ref,
                consistent_with_legacy=bool(
                    A_sd > 0 and abs(A_hat - A_ref) <= 2.0 * A_sd),
                model_r_squared=float(1.0 - np.sum(resid**2) / ss_tot)
                if ss_tot > 0 else float("nan"),
                model_residual_rms=float(np.sqrt(np.mean(resid**2))),
                decay_exponent_target=-3.0, **decay,
                null_zero_offset_rms=null_zero,
                null_mode_independent_signed_offset_sd=null_const,
                observed_c=[float(x) for x in obs_c],
                predicted_c=[float(x) for x in pred_c],
            )

    write_json(json_path, dict(
        description=(
            "WP-E extended central-offset test. 'canonical_truncated' is the "
            "exact_symbolic null: the modewise anomaly of the CANONICAL truncated "
            "Sugawara algebra. 'raw_empirical' uses the measured finite-feature "
            "pairing. Fitted models: (1) signed alternating constant offset "
            "c_m = 1 + (-1)^m A/(m(m^2-1)/12); (2) |c_m - 1| ~ m^p with the "
            "model target p = -3; (3) nulls: zero offset and a mode-independent "
            "signed (non-alternating) offset."),
        legacy_offset_reference=A_ref,
        config_hash=cfg.hash, fits=fits))
    print(f"  wrote {csv_path.name} ({len(rows)} rows) and {json_path.name}")
    for k, v in fits.items():
        print(f"    {k}: A={v['alternating_offset_A']:.6g} "
              f"+-{v['alternating_offset_sd']:.3g}  decay p="
              f"{v['decay_exponent']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
