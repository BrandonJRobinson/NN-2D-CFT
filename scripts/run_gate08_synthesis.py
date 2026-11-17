#!/usr/bin/env python3
"""Gate 8: intrinsic stationarity audit and the architecture synthesis.

Two jobs.

**The missing axis.** Every structural diagnostic in the campaign so far --
ordered pairing fidelity, raw Virasoro stability, Ward behaviour, intrinsic
physical admissibility -- is blind to loss of translation invariance. A control
with the correct radial spectrum and the correct isotropic angular law,
differing only by phase correlation between features, therefore survives all of
them. This gate measures stationarity directly, on the true two-argument kernel,
for the scalar and the current channel.

**The synthesis.** Pool the convergence exponents where an N^(-1/2) law is
predicted, and assemble the per-architecture table. Each control is designed to
violate ONE structural property; the test is whether it fails THAT axis, not
whether it fails all of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro import architectures as AR                 # noqa: E402
from nnft_virasoro import fits as FT                          # noqa: E402
from nnft_virasoro import plotstyle as PS                     # noqa: E402
from nnft_virasoro import translation_invariance as TI        # noqa: E402
from nnft_virasoro.cli import base_parser, setup              # noqa: E402
from nnft_virasoro.config import rng_for                      # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows    # noqa: E402

SCRIPT = "scripts/run_gate08_synthesis.py"
RES = ROOT / "results" / "jhep_submission_hardening"
FROZEN = ROOT / "results" / "jhep_frozen"


def stationarity_audit(cfg, sec, stamper, args):
    ti = sec["translation_invariance"]
    geo = TI.evaluation_geometry(int(ti["n_separations"]), int(ti["n_shifts"]),
                                 float(ti["max_separation"]),
                                 float(ti["shift_radius"]))
    widths = list(ti["n_features"])
    seeds = list(ti["seeds"])
    archs = list(cfg.section("gate01")["architectures"])
    if args.quick:
        widths, seeds = widths[:2], seeds[:1]
    params = dict(cfg.section("gate01")["params"])
    params["regulator_taper"] = float(ti["taper"])
    rows = []
    for arch in archs:
        for n_features in widths:
            for seed_name in seeds:
                rng = rng_for(cfg.section("named_seeds"), seed_name, "gate08ti",
                              arch, n_features)
                fs = AR.sample_architecture(arch, rng, int(n_features),
                                            float(ti["lambda_ir"]),
                                            float(ti["lambda_uv"]),
                                            params=params)
                sc = TI.translation_defect(fs, geo, kind="scalar")
                cu = TI.translation_defect(fs, geo, kind="current")
                rs = TI.rotation_defect(fs, geo, kind="scalar",
                                        n_angles=int(ti["n_angles"]))
                rc = TI.rotation_defect(fs, geo, kind="current",
                                        n_angles=int(ti["n_angles"]))
                rows.append(stamper.stamp(dict(
                    gate="gate08", check="intrinsic_stationarity",
                    spectral_regulator_class=ti["reference_regulator_class"],
                    architecture=arch, seed_name=seed_name,
                    n_features=int(n_features),
                    lambda_ir=float(ti["lambda_ir"]),
                    lambda_uv=float(ti["lambda_uv"]),
                    regulator_taper=float(ti["taper"]),
                    phase_pairing_is_trivial=bool(
                        np.array_equal(np.asarray(fs.idx_a),
                                       np.asarray(fs.idx_b))),
                    n_kernel_terms=int(np.asarray(fs.idx_a).size),
                    scalar_translation_defect=sc["translation_defect"],
                    scalar_translation_defect_worst=sc[
                        "translation_defect_worst_shift"],
                    scalar_kernel_rms=sc["kernel_rms"],
                    current_translation_defect=cu["translation_defect"],
                    current_translation_defect_worst=cu[
                        "translation_defect_worst_shift"],
                    current_kernel_rms=cu["kernel_rms"],
                    scalar_rotation_defect=rs["rotation_defect"],
                    current_rotation_defect=rc["rotation_defect"],
                    scalar_rotation_defect_worst=rs["rotation_defect_worst_angle"],
                    current_rotation_defect_worst=rc["rotation_defect_worst_angle"],
                    n_angles=rs["n_angles"],
                    n_evaluations=sc["n_evaluations"],
                    n_separations=sc["n_separations"], n_shifts=sc["n_shifts"]),
                    estimator_class="deterministic_finite_feature",
                    seed=seed_name))
        print(f"    {arch}: done", flush=True)
    return rows


def _exponent(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r = FT.power_law_fit(x[ok], y[ok])
    return float(r.coef("exponent")), float(r.se("exponent"))


def joint_fit(stamper, ti_rows):
    """Pool exponents ONLY where an N^(-1/2) law is actually predicted."""
    rows = []

    def add(gate, quantity, arch, exp, se, pooled, reason):
        rows.append(stamper.stamp(dict(
            gate="gate08", source_gate=gate, quantity=quantity,
            architecture=arch, exponent=exp, exponent_stderr=se,
            pooled=bool(pooled), pooling_reason=reason),
            estimator_class="deterministic_finite_feature", seed="multi"))

    g1 = pd.read_csv(RES / "gate01_intrinsic_admissibility_fits.csv")
    g1 = g1[g1.spectral_regulator_class == "tapered_log_window"]
    for _, r in g1.iterrows():
        add("gate01", "delta_iso", r.architecture, r.delta_iso_exponent,
            r.delta_iso_exponent_stderr,
            r.architecture == "selected_log_kernel",
            "random-feature kernel defect: N^-1/2 predicted")
    g2 = pd.read_csv(RES / "gate02_predarboux_virasoro_fits.csv")
    g2 = g2[g2.spectral_regulator_class == "tapered_log_window"]
    for _, r in g2.iterrows():
        add("gate02", "b_omega", r.architecture, r.b_omega_exponent,
            r.b_omega_stderr, r.architecture == "selected_log_kernel",
            "pairing defect: N^-1/2 predicted")
    g7 = pd.read_csv(RES / "ns_super_virasoro_fits.csv")
    g7 = g7[g7.architecture == "selected_log_kernel"]
    for _, r in g7.iterrows():
        add("gate07", f"{r.quantity}[{r.ns_construction}]", r.architecture,
            r.exponent, r.exponent_stderr, True,
            "random-feature kernel defect: N^-1/2 predicted")
    # Ward: a different functional of the same kernel; its exponent is NOT
    # predicted to be -1/2, so it is reported but never pooled.
    ctl = pd.read_csv(FROZEN / "architecture_controls_seed_rows.csv")
    ctl = ctl[ctl.regulator_class == "tapered_log_window"]
    for arch, g in ctl.groupby("architecture"):
        m = g.groupby("n_features").ward_stress_max_absolute_error.mean()
        e, se = _exponent(m.index.to_numpy(float), m.to_numpy(float))
        add("gate04", "ward_stress_max_absolute_error", arch, e, se, False,
            "Ward error is a different functional; no N^-1/2 prediction")
    # stationarity: for a phase-correlated architecture the defect is
    # structural and must NOT converge, so it is never pooled either.
    d = pd.DataFrame(ti_rows)
    for arch, g in d.groupby("architecture"):
        m = g.groupby("n_features").scalar_translation_defect.mean()
        e, se = _exponent(m.index.to_numpy(float), m.to_numpy(float))
        add("gate08", "scalar_translation_defect", arch, e, se, False,
            "structural symmetry defect; convergence is not predicted")
        mr = g.groupby("n_features").scalar_rotation_defect.mean()
        er, ser = _exponent(mr.index.to_numpy(float), mr.to_numpy(float))
        add("gate08", "scalar_rotation_defect", arch, er, ser, False,
            "isotropy defect: N^-1/2 only for an isotropic prior, pinned "
            "otherwise -- the discriminator is convergence, so never pooled")

    # inverse-variance pooling with a heterogeneity test
    P = [r for r in rows if r["pooled"] and np.isfinite(r["exponent"])
         and np.isfinite(r["exponent_stderr"]) and r["exponent_stderr"] > 0]
    if P:
        e = np.array([r["exponent"] for r in P])
        s = np.array([r["exponent_stderr"] for r in P])
        w = 1.0 / s**2
        mu = float(np.sum(w * e) / np.sum(w))
        se_mu = float(np.sqrt(1.0 / np.sum(w)))
        Q = float(np.sum(w * (e - mu) ** 2))
        dof = len(P) - 1
        from scipy import stats
        pQ = float(stats.chi2.sf(Q, dof)) if dof > 0 else float("nan")
        I2 = float(max(0.0, (Q - dof) / Q) * 100.0) if Q > 0 else 0.0
        null = FT.null_test(mu, se_mu, -0.5, dof=max(dof, 1))
        rows.append(stamper.stamp(dict(
            gate="gate08", source_gate="pooled", quantity="POOLED_EXPONENT",
            architecture="selected_log_kernel", exponent=mu,
            exponent_stderr=se_mu, pooled=True,
            pooling_reason="inverse-variance pool of the N^-1/2-predicted rows",
            n_pooled=len(P), cochran_Q=Q, cochran_dof=dof, cochran_p=pQ,
            I_squared_percent=I2,
            heterogeneity_detected=bool(np.isfinite(pQ) and pQ < 0.05),
            **{f"null_half_{k}": v for k, v in null.items()}),
            estimator_class="deterministic_finite_feature", seed="multi"))
        print(f"  pooled exponent {mu:+.4f} +/- {se_mu:.4f} over {len(P)} rows "
              f"(vs -1/2: p={null['p_value']:.3f}; Q={Q:.2f}, p_Q={pQ:.3f}, "
              f"I2={I2:.0f}%)")
    return rows


def synthesis(cfg, sec, stamper, ti_rows):
    th = sec["thresholds"]
    dv = sec["designed_violation"]
    axis_of = sec["axis_for_violation"]
    ti = pd.DataFrame(ti_rows)
    nmax_ti = ti.n_features.max()

    g1 = pd.read_csv(RES / "gate01_intrinsic_admissibility_fits.csv")
    g1 = g1[g1.spectral_regulator_class == "tapered_log_window"].set_index(
        "architecture")
    g2 = pd.read_csv(RES / "gate02_predarboux_virasoro_fits.csv")
    g2 = g2[g2.spectral_regulator_class == "tapered_log_window"].set_index(
        "architecture")
    ctl = pd.read_csv(FROZEN / "architecture_controls_seed_rows.csv")
    ctl = ctl[ctl.regulator_class == "tapered_log_window"]
    nmax_w = ctl.n_features.max()

    rows = []
    for arch in g1.index:
        r1 = g1.loc[arch]
        t = ti[(ti.architecture == arch) & (ti.n_features == nmax_ti)]
        sc = float(t.scalar_translation_defect.mean()) if len(t) else np.nan
        cu = float(t.current_translation_defect.mean()) if len(t) else np.nan
        rs = float(t.scalar_rotation_defect.mean()) if len(t) else np.nan
        rc = float(t.current_rotation_defect.mean()) if len(t) else np.nan
        gti = ti[ti.architecture == arch].groupby(
            "n_features").scalar_rotation_defect.mean()
        rot_exp, rot_se = _exponent(gti.index.to_numpy(float),
                                    gti.to_numpy(float))
        wg = ctl[ctl.architecture == arch].groupby(
            "n_features").ward_stress_max_absolute_error.mean()
        w_exp, w_se = _exponent(wg.index.to_numpy(float), wg.to_numpy(float))
        w_val = float(wg.loc[nmax_w]) if nmax_w in wg.index else np.nan

        # --- axis verdicts, each pre-specified ------------------------
        ax = {}
        ax["translation_invariance"] = bool(
            np.isfinite(sc) and sc <= float(th["translation_defect_zero"]))
        ax["rotational_invariance"] = bool(
            np.isfinite(rot_exp)
            and rot_exp <= float(th["rotation_converging_exponent"]))
        ax["physical_admissibility"] = bool(
            r1.lambda_min_at_max_width > float(th["lambda_min_positive"]))
        ax["pairing_fidelity"] = bool(
            np.isfinite(r1.delta_iso_exponent)
            and r1.delta_iso_exponent <= float(th["delta_iso_converging_exponent"]))
        ax["ward_behaviour"] = bool(
            np.isfinite(w_exp)
            and w_exp <= float(th["ward_converging_exponent"]))
        if arch in g2.index:
            ax["raw_virasoro_fidelity"] = bool(
                not bool(g2.loc[arch].b_omega_null_rejected))
            vir_note = "measured"
        else:
            ax["raw_virasoro_fidelity"] = None
            vir_note = "not run for this architecture in gate02"

        designed = dv.get(arch, "unknown")
        target_axis = axis_of.get(designed)
        fails_designed = (None if target_axis is None
                          else (ax.get(target_axis) is False))
        passed = [k for k, v in ax.items() if v is True]
        failed = [k for k, v in ax.items() if v is False]
        rows.append(stamper.stamp(dict(
            gate="gate08", architecture=arch,
            is_selected=bool(arch == "selected_log_kernel"),
            designed_violation=designed, designed_axis=target_axis,
            fails_designed_axis=fails_designed,
            # the new axis
            scalar_translation_defect=sc, current_translation_defect=cu,
            scalar_rotation_defect=rs, current_rotation_defect=rc,
            scalar_rotation_exponent=rot_exp,
            scalar_rotation_exponent_stderr=rot_se,
            translation_defect_width=int(nmax_ti),
            # existing axes
            delta_star_at_max_width=float(r1.delta_star_at_max_width),
            delta_iso_at_max_width=float(r1.delta_iso_at_max_width),
            delta_iso_exponent=float(r1.delta_iso_exponent),
            delta_iso_fit_status=str(r1.delta_iso_fit_status),
            lambda_min_at_max_width=float(r1.lambda_min_at_max_width),
            ward_at_max_width=w_val, ward_exponent=w_exp,
            ward_exponent_stderr=w_se, ward_width=int(nmax_w),
            b_omega_exponent=(float(g2.loc[arch].b_omega_exponent)
                              if arch in g2.index else np.nan),
            b_omega_null_rejected=(bool(g2.loc[arch].b_omega_null_rejected)
                                   if arch in g2.index else None),
            raw_virasoro_note=vir_note,
            # verdicts
            axis_translation_invariance=ax["translation_invariance"],
            axis_rotational_invariance=ax["rotational_invariance"],
            axis_physical_admissibility=ax["physical_admissibility"],
            axis_pairing_fidelity=ax["pairing_fidelity"],
            axis_ward_behaviour=ax["ward_behaviour"],
            axis_raw_virasoro_fidelity=ax["raw_virasoro_fidelity"],
            n_axes_passed=len(passed), n_axes_failed=len(failed),
            axes_failed=";".join(sorted(failed)) or "none",
            passes_all_measured_axes=bool(not failed)),
            estimator_class="deterministic_finite_feature", seed="multi"))
    return rows


def symmetry_figure(ti_rows, figdir):
    """The two symmetry audits side by side, on a log axis.

    Stationarity separates structurally (a flat line fifteen decades above a
    floating-point floor); isotropy separates by convergence (a pinned line
    against a family falling at N^-1/2). Plotting them together shows why the
    two are read differently.
    """
    d = pd.DataFrame(ti_rows)
    fig, (ax, az) = PS.new_figure(1, 2, figsize=(9.8, 4.3))
    order = sorted(d.architecture.unique(),
                   key=lambda a: (a != "selected_log_kernel", a))
    m = PS.MARKERS["deterministic_finite_feature"]
    for i, arch in enumerate(order):
        g = d[d.architecture == arch]
        nf = np.array(sorted(g.n_features.unique()), dtype=float)
        lbl = arch.replace("_", " ")
        sel = arch == "selected_log_kernel"
        kw = dict(color=PS.COLORS[i % len(PS.COLORS)], ms=5,
                  lw=2.2 if sel else 1.2, zorder=5 if sel else 2)
        ti_v = [max(g[g.n_features == n].scalar_translation_defect.mean(), 1e-18)
                for n in nf]
        ri_v = [g[g.n_features == n].scalar_rotation_defect.mean() for n in nf]
        ax.plot(nf, ti_v, m + "-", label=lbl, **kw)
        az.plot(nf, ri_v, m + "-", label=lbl, **kw)
    ax.axhspan(1e-18, 1e-14, color="0.88", zorder=0)
    ax.annotate("floating-point floor", xy=(nf[0], 3e-16), fontsize=7.5,
                color="0.35", va="bottom")
    ax.set_ylim(1e-18, 1.0)
    pc = d[d.architecture == "phase_correlated_features"]
    if len(pc):
        nfp = np.array(sorted(pc.n_features.unique()), dtype=float)
        yv = float(pc[pc.n_features == nfp[-1]].scalar_translation_defect.mean())
        ax.annotate("phase correlated features\n(the only one broken)",
                    xy=(nfp[-1], yv), xytext=(-10, -46),
                    textcoords="offset points", fontsize=8, ha="right",
                    arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.annotate("all seven independent-phase architectures",
                    xy=(nfp[1], 1e-16), xytext=(0, 26),
                    textcoords="offset points", fontsize=8, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("stationarity: exact or broken\n"
                 r"$\delta_{\rm TI}$ (scalar channel)", fontsize=9)
    PS.annotate_slope(az, np.array([1e4, 1e6]), np.array([8e-3, 8e-4]), -0.5,
                      r"$N^{-1/2}$")
    az.set_title("isotropy: converging or pinned\n"
                 r"$\delta_{\rm RI}$ (scalar channel)", fontsize=9)
    for a in (ax, az):
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("$N$ (features)")
    ax.set_ylabel("defect")
    az.legend(fontsize=6.8, loc="lower left", ncol=1)
    fig.suptitle("Gate 8: the two intrinsic symmetry audits "
                 "(true two-argument kernel)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return PS.save(fig, "gate08_intrinsic_symmetry_audit.pdf", outdir=figdir)


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    sec = cfg.section("gate08")
    out = outdir / "architecture_synthesis.csv"
    if should_skip(out, args.skip_existing):
        print("  [skip] gate08")
        return 0

    print("  intrinsic stationarity audit (true two-argument kernel)...",
          flush=True)
    ti_rows = stationarity_audit(cfg, sec, stamper, args)
    write_rows(outdir / "intrinsic_stationarity_seed_rows.csv", ti_rows,
               required=["gate", "architecture", "seed_name", "n_features",
                         "scalar_translation_defect",
                         "current_translation_defect",
                         "scalar_rotation_defect", "current_rotation_defect",
                         "phase_pairing_is_trivial"],
               sort_by=["architecture", "n_features", "seed_name"])

    jrows = joint_fit(stamper, ti_rows)
    write_rows(outdir / "convergence_joint_fit.csv", jrows, required=[
        "gate", "source_gate", "quantity", "architecture", "exponent",
        "pooled", "pooling_reason"],
        sort_by=["pooled", "source_gate", "quantity", "architecture"])

    figpath = symmetry_figure(ti_rows, figdir)
    print(f"  figure {figpath.name}")

    srows = synthesis(cfg, sec, stamper, ti_rows)
    write_rows(out, srows, required=[
        "gate", "architecture", "is_selected", "designed_violation",
        "fails_designed_axis", "scalar_translation_defect",
        "current_translation_defect", "scalar_rotation_defect",
        "current_rotation_defect", "passes_all_measured_axes"],
        sort_by=["is_selected", "architecture"])

    print(f"\n  {'architecture':<32}{'sc_TI':>8}{'cu_TI':>8}{'sc_RI':>8}"
          f"{'RI_exp':>8}{'lam_min':>9}{'d_iso_e':>8}{'ward_e':>8}  failed axes")
    for r in sorted(srows, key=lambda x: (not x["is_selected"],
                                          x["architecture"])):
        print(f"  {r['architecture']:<32}{r['scalar_translation_defect']:8.4f}"
              f"{r['current_translation_defect']:8.4f}"
              f"{r['scalar_rotation_defect']:8.4f}"
              f"{r['scalar_rotation_exponent']:8.3f}"
              f"{r['lambda_min_at_max_width']:9.3f}"
              f"{r['delta_iso_exponent']:8.3f}{r['ward_exponent']:8.3f}  "
              f"{r['axes_failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
