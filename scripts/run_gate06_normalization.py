#!/usr/bin/env python3
"""Gate 6: kernel normalization and convention crosswalk.

Purely analytic. This gate exists to pin the conventions that every later gate
silently depends on, BEFORE any expensive job runs: the prior normalization
(``Z``, ``L_w``), the field amplitude calibration (``sigma_phi^2``), the sign
and magnitude of the logarithmic scalar kernel, the spin of the current
channel, and the ideal-kernel limit of the ordered two-radius pairing.

Each check records the measured value, the convention target, and the tolerance
it is judged against, so a later convention change cannot pass silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import jv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro import kernels as K                       # noqa: E402
from nnft_virasoro import ordered_pairing as OP              # noqa: E402
from nnft_virasoro.architectures import _tukey_log_window    # noqa: E402
from nnft_virasoro.cli import base_parser, setup             # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows   # noqa: E402

SCRIPT = "scripts/run_gate06_normalization.py"


def _chk(rows, stamper, *, check, quantity, measured, target, tol,
         relative=True, must_exceed=None, diagnostic=False, **extra):
    """Record one convention check with its target and tolerance.

    Three kinds of row, distinguished so none of them can pass vacuously:

    ``tol``          the measured value must come within ``tol`` of the target;
    ``must_exceed``  the measured value must be at LEAST this far away -- used
                     for the controls that are supposed to fail (a spin-zero
                     radial moment, a single unordered contour integral);
    ``diagnostic``   recorded as evidence, never graded.
    """
    measured = float(np.real(measured))
    target = float(np.real(target))
    denom = max(abs(target), 1e-300) if relative else 1.0
    err = abs(measured - target) / denom
    if diagnostic:
        kind, passes = "diagnostic", None
    elif must_exceed is not None:
        kind, passes = "must_exceed", bool(err >= float(must_exceed))
    else:
        kind, passes = "must_match", bool(err <= tol)
    rows.append(stamper.stamp(dict(
        gate="gate06", check=check, quantity=quantity,
        measured=measured, convention_target=target,
        error=err, error_is_relative=bool(relative), check_kind=kind,
        tolerance=float(tol) if must_exceed is None else float("nan"),
        must_exceed=float(must_exceed) if must_exceed is not None else float("nan"),
        passes=passes, **extra),
        estimator_class="analytic_closed_kernel", seed=None))
    return rows[-1]


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    sec = cfg.section("gate06")
    out_path = outdir / "gate06_kernel_normalization_checks.csv"
    if should_skip(out_path, args.skip_existing):
        print("  [skip] gate06")
        return 0

    rows: list[dict] = []
    for lir, luv in sec["regulators"]:
        L = K.log_ratio(lir, luv)
        tag = dict(lambda_ir=float(lir), lambda_uv=float(luv))

        for taper in sec["tapers"]:
            Lw = K.tapered_log_measure(lir, luv, taper)
            t = dict(**tag, regulator_taper=float(taper))

            # --- C1/C2: the log measure itself -------------------------
            if taper <= 0:
                _chk(rows, stamper, check="C1_sharp_window_measure",
                     quantity="L_w vs L = log(Lambda_UV/Lambda_IR)",
                     measured=Lw, target=L, tol=1e-12, **t)
            else:
                # A raised-cosine taper removes exactly half of each roll-off
                # region, so L_w/L = 1 - taper for a two-sided window.
                _chk(rows, stamper, check="C2_tapered_window_measure",
                     quantity="L_w / L for the raised-cosine log window",
                     measured=Lw / L, target=1.0 - float(taper), tol=1e-6, **t)

            # --- C3: amplitude calibration ----------------------------
            s2 = K.sigma_phi_squared(lir, luv, log_measure=Lw)
            _chk(rows, stamper, check="C3_sigma_phi_squared",
                 quantity="sigma_phi^2 vs 4 L_w (kappa = 1 calibration)",
                 measured=s2, target=4.0 * Lw, tol=1e-12, **t)

            # --- C4: prior normalization Z, via sigma_phi^2 = 2 Z / pi --
            Z = 2.0 * np.pi * Lw
            _chk(rows, stamper, check="C4_prior_normalization_Z",
                 quantity="2 Z / pi vs sigma_phi^2, with Z = 2 pi L_w",
                 measured=2.0 * Z / np.pi, target=s2, tol=1e-12,
                 Z_Lambda_tau=float(Z), **t)

        # --- C5: normal-ordering constant K(0) = sigma_phi^2/2 ---------
        _chk(rows, stamper, check="C5_normal_ordering_constant",
             quantity="K_Lambda(0) vs sigma_phi^2/2 = 2 L",
             measured=K.scalar_kernel_zero(lir, luv),
             target=2.0 * L, tol=1e-12, **tag)

        # --- C6: logarithmic slope of the scalar kernel ----------------
        # Inside the scaling window K_Lambda(r) = -2 log r + const, so
        # dK/dlog r = -2. NOTE the sign: (K(r1) - K(r2))/log(r2/r1) is MINUS
        # the derivative. Two radii well inside the window, decade-separated.
        r1 = 10.0 / luv * np.sqrt(10.0)
        r2 = r1 * 10.0
        if r2 < 0.1 / lir:
            k1 = float(K.scalar_kernel(r1, lir, luv))
            k2 = float(K.scalar_kernel(r2, lir, luv))
            slope = (k2 - k1) / np.log(r2 / r1)          # = dK/dlog r
            _chk(rows, stamper, check="C6_scalar_kernel_log_slope",
                 quantity="dK_Lambda/dlog r vs -sigma_phi^2/(2L) = -2",
                 measured=slope, target=-2.0, tol=5e-3,
                 r_lower=float(r1), r_upper=float(r2), **tag)

        # --- C7: the current channel is spin two, not spin zero --------
        # K_JJ = (zbar-wbar)^2 k''(|xi|^2): with xi = rho e^{i theta} the phase
        # is exp(-2 i theta). A spin-zero (J_0) channel would have no theta
        # dependence at all. Measure the winding directly.
        for kr in sec["spin_two_kr"]:
            rho_sep = float(kr) / luv * 10.0
            for th in sec["spin_two_theta"]:
                xi = rho_sep * np.exp(1j * float(th))
                val = complex(K.current_kernel_analytic(xi, 0j, lir, luv))
                ref = complex(K.current_kernel_analytic(rho_sep + 0j, 0j, lir, luv))
                # phase relative to theta = 0 must be exp(-2 i theta)
                got = val / ref if abs(ref) > 0 else complex("nan")
                _chk(rows, stamper, check="C7_current_channel_is_spin_two",
                     quantity="Re[K_JJ(theta)/K_JJ(0)] vs cos(2 theta)",
                     measured=got.real, target=float(np.cos(2.0 * th)),
                     tol=1e-9, relative=False,
                     separation=float(rho_sep), theta=float(th), **tag)
                _chk(rows, stamper, check="C7_current_channel_is_spin_two",
                     quantity="Im[K_JJ(theta)/K_JJ(0)] vs -sin(2 theta)",
                     measured=got.imag, target=float(-np.sin(2.0 * th)),
                     tol=1e-9, relative=False,
                     separation=float(rho_sep), theta=float(th), **tag)

        # --- C8: the radial part is the J_2 moment, not J_0 ------------
        # K_JJ(r) = (1/2) int dk k w(k) J_2(k r) for the sharp annulus.
        for kr in sec["spin_two_kr"]:
            r = float(kr) / luv * 10.0
            pts, wts = OP._gauss_legendre_panels(lir, luv,
                                                 np.pi / (8 * max(r, 1e-12)),
                                                 nodes=8)
            quad_j2 = float(np.sum(wts * pts * jv(2, pts * r)) / 2.0)
            quad_j0 = float(np.sum(wts * pts * jv(0, pts * r)) / 2.0)
            exact = float(np.real(K.current_kernel_analytic(r + 0j, 0j, lir, luv)))
            _chk(rows, stamper, check="C8_radial_moment_is_J2",
                 quantity="closed-form K_JJ(r) vs the J_2 spectral quadrature",
                 measured=quad_j2, target=exact, tol=1e-6,
                 separation=float(r), **tag)
            # The spin-zero alternative must NOT reproduce it. Without this
            # control, C8 would pass for any radial moment that happened to be
            # close to the closed form at these separations.
            _chk(rows, stamper, check="C8_J0_alternative_is_excluded",
                 quantity="relative gap between the J_0 quadrature and "
                          "closed-form K_JJ(r) (must be large)",
                 measured=abs(quad_j0 - exact) / max(abs(exact), 1e-300),
                 target=0.0, tol=float("nan"), relative=False, must_exceed=0.1,
                 j0_quadrature=quad_j0, separation=float(r), **tag)

        # --- C9 (diagnostic): the pointwise kernel does NOT converge ---
        # Recorded, not graded: these values are the evidence that a pointwise
        # comparison to 1/(z-w)^2 is the wrong test. See the graded C9 below.
        r = 1.0
        reg = complex(K.current_kernel_analytic(r + 0j, 0j, lir, luv))
        ide = complex(K.current_kernel_ideal(r + 0j, 0j))
        _chk(rows, stamper, check="C9a_pointwise_ideal_limit_diagnostic",
             quantity="K_JJ^Lambda(1,0) vs ideal 1/(z-w)^2 (NOT expected to agree)",
             measured=reg.real, target=ide.real, tol=float("nan"),
             diagnostic=True, separation=float(r), **tag)

    # --- C9 (graded): the UV tail envelope obeys the sqrt(Lambda) law ---
    # J_2(kr) ~ sqrt(2/(pi k r)) cos(kr - 5pi/4), so
    #   K_JJ^Lambda(r) = (1/2) int^Lambda dk k J_2(kr)
    # has a boundary term of amplitude sqrt(Lambda) sqrt(2/(pi r)) / (2 r).
    # Confirming this law is what *establishes* the non-convergence quantitatively.
    pr = sec["uv_envelope_probe"]
    plir, pluv = float(pr["lambda_ir"]), float(pr["lambda_uv"])
    for r0s in pr["separations"]:
        r0s = float(r0s)
        half = float(pr["periods"]) * np.pi / (pluv * r0s)
        rr = np.linspace(r0s - half, r0s + half, 40001)
        env = float(np.max(np.abs(np.real(
            K.current_kernel_analytic(rr + 0j, 0j, plir, pluv)))))
        predicted = np.sqrt(pluv) * np.sqrt(2.0 / (np.pi * r0s)) / (2.0 * r0s)
        _chk(rows, stamper, check="C9b_uv_tail_envelope_sqrt_lambda",
             quantity="max|K_JJ^Lambda| near r vs sqrt(Lambda) stationary-phase "
                      "amplitude",
             measured=env, target=float(predicted),
             tol=float(pr["tolerance"]),
             separation=r0s, lambda_ir=plir, lambda_uv=pluv)

    # --- C10: ordered two-radius pairing of the IDEAL kernel -----------
    modes = OP.mode_window(int(sec["mode_cutoff"]))
    r0 = float(sec["r0"])
    _, r_in, r_out = OP.radii_from_window(r0 / sec["rho"], r0 * sec["rho"],
                                          float(sec["rho"]), r0=r0)
    res = OP.pairing_ideal_quadrature(modes, r_in, r_out, q=512, r0=r0)
    can = OP.canonical_pairing(modes)
    err = float(np.max(np.abs(res.omega - can)))
    _chk(rows, stamper, check="C10_ideal_ordered_pairing_is_canonical",
         quantity="max |Omega_ideal - m delta_{m+n,0}|",
         measured=err, target=0.0, tol=1e-8, relative=False,
         mode_cutoff=int(sec["mode_cutoff"]), r_in=float(r_in),
         r_out=float(r_out), rho=float(sec["rho"]), r0=r0)
    # The ordering matters: I^> - I^< is what is canonical, not either alone.
    # If a single ordering already gave the canonical form, the ordered
    # prescription would be unnecessary -- so this control must fail.
    half = float(np.max(np.abs(res.i_greater - can)))
    _chk(rows, stamper, check="C10_ordering_is_required",
         quantity="max |I^> - m delta_{m+n,0}| for ONE ordering (must NOT vanish)",
         measured=half, target=0.0, tol=float("nan"), relative=False,
         must_exceed=0.5, mode_cutoff=int(sec["mode_cutoff"]))

    write_rows(out_path, rows, required=[
        "gate", "check", "quantity", "measured", "convention_target",
        "error", "check_kind", "tolerance", "must_exceed", "passes"],
        sort_by=["check", "lambda_uv", "regulator_taper", "separation", "theta"])

    graded = [r for r in rows if r["check_kind"] != "diagnostic"]
    n_pass = sum(1 for r in graded if r["passes"])
    n_diag = len(rows) - len(graded)
    print(f"  {len(rows)} checks written: {n_pass}/{len(graded)} graded pass, "
          f"{n_diag} diagnostic")
    for r in graded:
        if not r["passes"]:
            bound = (f"tol={r['tolerance']:.3g}" if r["check_kind"] == "must_match"
                     else f"must_exceed={r['must_exceed']:.3g}")
            print(f"    FAIL {r['check']}: {r['quantity']} "
                  f"measured={r['measured']:.6g} target={r['convention_target']:.6g} "
                  f"err={r['error']:.3g} {bound}")
    return 0 if n_pass == len(graded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
