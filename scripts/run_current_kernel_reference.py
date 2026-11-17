#!/usr/bin/env python3
"""WP-F: reference current-kernel result on a strict in-window evaluation interval.

`relative_error_definition` is emitted per row.  This value is deliberately NOT
compared with the broad robustness grid: it has its own fit window, its own
metric definition, and its own claim-ledger row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import jv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnft_virasoro import architectures as ARCH  # noqa: E402
from nnft_virasoro import kernels as KR  # noqa: E402
from nnft_virasoro.architectures import _tukey_log_window  # noqa: E402
from nnft_virasoro.cli import base_parser, setup  # noqa: E402
from nnft_virasoro.config import rng_for  # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows  # noqa: E402
from nnft_virasoro.ordered_pairing import _gauss_legendre_panels  # noqa: E402

SCRIPT = "scripts/run_current_kernel_reference.py"

METRIC_DEF = (
    "amplitude_fit A minimizes ||K_JJ(r) - A r^-2||_2 over the recorded "
    "evaluation interval on the real axis; relative_error = "
    "||K_JJ - A r^-2||_2 / ||A r^-2||_2 (scale-free shape error). "
    "amplitude_relative_error = |A - 1| measures the kappa = 1 calibration."
)


def analytic_kjj(r, lambda_ir, lambda_uv, taper):
    if taper <= 0:
        return np.array([float(np.real(KR.current_kernel_analytic(
            x + 0j, 0j, lambda_ir, lambda_uv))) for x in r])
    out = []
    for x in r:
        pts, wts = _gauss_legendre_panels(lambda_ir, lambda_uv,
                                          np.pi / (8 * max(x, 1e-12)), nodes=8)
        w = _tukey_log_window(np.log(pts), np.log(lambda_ir), np.log(lambda_uv),
                              taper)
        out.append(float(np.sum(wts * pts * w * jv(2, pts * x)) / 2.0))
    return np.array(out)


def fit_amplitude(r, kjj):
    basis = r**-2.0
    a = float(np.sum(kjj * basis) / np.sum(basis * basis))
    resid = kjj - a * basis
    return a, float(np.linalg.norm(resid) / np.linalg.norm(a * basis))


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    path = outdir / "current_kernel_reference.csv"
    if should_skip(path, args.skip_existing):
        print("  [skip] WP-F")
        return 0
    sec = cfg.section("wp_f")
    g = cfg.section("global")
    rclasses = [(str(k), float(v)) for k, v in g["regulator_classes"].items()]
    rows = []
    for rclass, taper in rclasses:
        for lambda_ir, lambda_uv in sec["regulators"]:
            r_min, r_max = KR.physical_window(lambda_ir, lambda_uv,
                                              margin_uv=g["window_margin_uv"],
                                              margin_ir=g["window_margin_ir"])
            r = np.exp(np.linspace(np.log(r_min), np.log(r_max),
                                   int(sec["n_eval_points"])))
            ana = analytic_kjj(r, lambda_ir, lambda_uv, taper)
            a_fit, rel = fit_amplitude(r, ana)
            base = dict(
                regulator_class=rclass, regulator_taper=taper,
                lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                r_min_eval=float(r_min), r_max_eval=float(r_max),
                n_eval_points=int(r.size),
                window_margin_uv=float(g["window_margin_uv"]),
                window_margin_ir=float(g["window_margin_ir"]),
                relative_error_definition=METRIC_DEF)
            rows.append(stamper.stamp(dict(
                **base, architecture="selected_log_kernel", n_features=0,
                seed_name="", amplitude_fit=a_fit, relative_error=rel,
                amplitude_relative_error=abs(a_fit - 1.0)),
                estimator_class="analytic_closed_kernel", seed=None))
            for n_features in sec["n_features"]:
                for seed_name in sec["seeds"]:
                    rng = rng_for(cfg.section("named_seeds"), seed_name, "fref",
                                  lambda_ir, lambda_uv, n_features, rclass)
                    fs = ARCH.sample_architecture(
                        "selected_log_kernel", rng, n_features, lambda_ir,
                        lambda_uv, params={"regulator_taper": taper})
                    emp = np.real(KR.empirical_current_kernel(
                        r.astype(complex), 0j, fs.kc, lambda_ir, lambda_uv,
                        log_measure=fs.log_measure))
                    a_e, rel_e = fit_amplitude(r, emp)
                    rows.append(stamper.stamp(dict(
                        **base, architecture="selected_log_kernel",
                        n_features=int(n_features), seed_name=seed_name,
                        amplitude_fit=a_e, relative_error=rel_e,
                        amplitude_relative_error=abs(a_e - 1.0),
                        sampling_error_vs_analytic=float(
                            np.linalg.norm(emp - ana) / np.linalg.norm(ana))),
                        estimator_class="deterministic_finite_feature",
                        seed=seed_name))
            print(f"  WP-F {rclass} ({lambda_ir:.0e},{lambda_uv:.0e}) "
                  f"A={a_fit:.4f} rel={rel:.4f}")
    write_rows(path, rows, required=[
        "regulator_class", "lambda_ir", "lambda_uv", "r_min_eval", "r_max_eval",
        "n_features", "seed_name", "amplitude_fit", "relative_error_definition",
        "relative_error", "window_margin_uv", "window_margin_ir"],
        sort_by=["regulator_class", "lambda_ir", "lambda_uv", "n_features",
                 "seed_name"])
    print(f"  wrote {path.name} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
