#!/usr/bin/env python3
"""Generate every frozen figure from the frozen CSVs only.

Figure rules (mission section 4): one shared style; log axes only on strictly
positive data; fitted lines and theory guides; slopes labelled in-panel;
uncertainty bars; deterministic values visually distinguished from sampled ones
(marker convention in `plotstyle.MARKERS`); never a floating-point zero on a log
axis; every caption states the estimator class and the source CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro import plotstyle as PS  # noqa: E402
from nnft_virasoro.cli import base_parser, setup  # noqa: E402
from nnft_virasoro.io_utils import write_json  # noqa: E402

SCRIPT = "scripts/make_figures.py"
CAPTIONS: dict[str, str] = {}


def load(outdir, name):
    p = Path(outdir) / name
    if not p.exists():
        print(f"  [skip] {name} not present")
        return None
    return pd.read_csv(p)


def cap(fig_name, source, estimator, text):
    CAPTIONS[fig_name] = (f"{text} Estimator class: {estimator}. "
                          f"Source: results/jhep_frozen/{source}.")


def _errbar(ax, x, y, e, label, color, marker):
    m = PS.positive_only(y) & PS.positive_only(x)
    if m.sum() == 0:
        return
    ax.errorbar(np.asarray(x)[m], np.asarray(y)[m],
                yerr=np.asarray(e)[m] if e is not None else None,
                fmt=marker + "-", color=color, label=label, capsize=2, ms=4.5)


# --------------------------------------------------------------------------
def fig_radius_plateau(outdir, figdir):
    df = load(outdir, "ordered_pairing_grid.csv")
    pl = load(outdir, "ordered_pairing_radius_plateau.csv")
    if df is None:
        return
    classes = sorted(df["regulator_class"].unique())
    fig, axes = PS.new_figure(1, len(classes), figsize=(3.4 * len(classes), 3.6),
                              sharey=True, squeeze=False)
    for ax, rc in zip(axes[0], classes):
        sub = df[df["regulator_class"] == rc]
        for i, (rho, s) in enumerate(sub.groupby("rho")):
            g = (s.groupby("r0")["anti_diagonal_amplitude_error"]
                 .agg(["mean", "std", "size"]).reset_index())
            _errbar(ax, g["r0"], g["mean"], g["std"] / np.sqrt(g["size"]),
                    rf"$\rho={rho:g}$", PS.COLORS[i % len(PS.COLORS)],
                    PS.MARKERS["deterministic_finite_feature"])
        if pl is not None:
            p = pl[pl["regulator_class"] == rc]
            if len(p):
                best = p.loc[p["plateau_nodes"].idxmax()]
                if np.isfinite(best["plateau_r0_lo"]):
                    ax.axvspan(best["plateau_r0_lo"], best["plateau_r0_hi"],
                               color="0.85", zorder=0,
                               label="plateau (stated algorithm)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$R_0$")
        ax.set_title(rc.replace("_", " "), fontsize=9)
        ax.legend(ncol=1)
    axes[0][0].set_ylabel(r"antidiagonal amplitude error of $\omega^{(N,\Lambda)}$")
    PS.save(fig, "ordered_pairing_radius_plateau.pdf", outdir=figdir)
    cap("ordered_pairing_radius_plateau.pdf", "ordered_pairing_grid.csv",
        "deterministic_finite_feature",
        "Operational two-radius pairing versus contour radius $R_0$ at several "
        "$\\rho$, for both regulator classes. Shading marks the largest connected "
        "plateau found by the stated algorithm (adjacent nodes compatible when "
        "$|\\Delta Y|\\le 3\\sigma + $ Cauchy--Green bound). Error bars are the "
        "standard error over independent named seeds.")


def fig_convergence(outdir, figdir):
    df = load(outdir, "predarboux_convergence_seed_rows.csv")
    fits = load(outdir, "predarboux_convergence_fits.csv")
    if df is None:
        return
    panels = [("pairing_leakage", "predarboux_pairing_vs_width.pdf",
               r"$\epsilon_{\rm leak}$ (operational two-radius)"),
              ("raw_virasoro_residual", "raw_virasoro_vs_width.pdf",
               r"raw Sugawara residual $R_{\rm raw}$ (filtered-Fock)"),
              ("pairing_sampling_error", "predarboux_sampling_error_vs_width.pdf",
               r"$\|\omega^{(N)}-\omega^{\rm ana}\|_F/\|\omega_{\rm can}\|_F$")]
    for metric, fname, ylabel in panels:
        classes = sorted(df["regulator_class"].unique())
        fig, axes = PS.new_figure(1, len(classes),
                                  figsize=(3.4 * len(classes), 3.6),
                                  sharey=True, squeeze=False)
        for ax, rc in zip(axes[0], classes):
            sub = df[df["regulator_class"] == rc]
            for i, (mc, s) in enumerate(sub.groupby("mode_cutoff")):
                g = (s.groupby("n_features")[metric]
                     .agg(["mean", "std", "size"]).reset_index())
                _errbar(ax, g["n_features"], g["mean"],
                        g["std"] / np.sqrt(g["size"]), rf"$\Lambda={mc}$",
                        PS.COLORS[i % len(PS.COLORS)],
                        PS.MARKERS["deterministic_finite_feature"])
            g = (sub.groupby("n_features")[metric].mean().reset_index())
            PS.annotate_slope(ax, g["n_features"], g[metric], -0.5,
                              r"$N^{-1/2}$")
            if fits is not None:
                f = fits[(fits["regulator_class"] == rc)
                         & (fits["metric"] == metric)]
                if len(f):
                    b = float(f["n_features_exponent"].iloc[0])
                    PS.annotate_slope(ax, g["n_features"], g[metric], b,
                                      rf"fit $b={b:.3f}$", color=PS.COLORS[1],
                                      ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"feature width $N$")
            ax.set_title(rc.replace("_", " "), fontsize=9)
            ax.legend(ncol=2)
        axes[0][0].set_ylabel(ylabel)
        PS.save(fig, fname, outdir=figdir)
        cap(fname, "predarboux_convergence_seed_rows.csv",
            "deterministic_finite_feature",
            f"{ylabel} versus feature width, grouped by Laurent cutoff, for both "
            "regulator classes. Points and bars are the mean and standard error "
            "over ten independent named seeds. Dashed grey: the $N^{-1/2}$ "
            "random-feature guide; dotted: the fitted width exponent.")


def fig_controls(outdir, figdir):
    df = load(outdir, "architecture_controls_seed_rows.csv")
    if df is None:
        return
    specs = [("architecture_controls_predarboux.pdf",
              ["pre_darboux_leakage", "raw_virasoro_residual"],
              "(a) radial / pre-Darboux"),
             ("architecture_controls_ward.pdf",
              ["ward_current_max_absolute_error", "ward_stress_max_absolute_error"],
              "(b) chiral Ward test"),
             ("architecture_controls_postdarboux.pdf",
              ["post_darboux_pairing_error", "post_darboux_virasoro_residual",
               "star_compatibility_defect"],
              "(c) post-Darboux implementation check")]
    for fname, fields, title in specs:
        classes = sorted(df["regulator_class"].unique())
        fig, axes = PS.new_figure(1, len(classes),
                                  figsize=(4.2 * len(classes), 4.4),
                                  sharey=True, squeeze=False)
        for ax, rc in zip(axes[0], classes):
            sub = df[(df["regulator_class"] == rc)
                     & (df["n_features"] == df["n_features"].max())]
            archs = sorted(sub["architecture"].unique())
            x = np.arange(len(archs))
            width = 0.8 / len(fields)
            floor = None
            for j, f in enumerate(fields):
                if f not in sub:
                    continue
                m = sub.groupby("architecture")[f].mean().reindex(archs).to_numpy()
                e = sub.groupby("architecture")[f].std().reindex(archs).to_numpy()
                pos = m > 0
                # never plot a floating-point zero on a log axis
                if floor is None:
                    allv = sub[[c for c in fields if c in sub]].to_numpy(dtype=float)
                    allv = allv[np.isfinite(allv) & (allv > 0)]
                    floor = float(allv.min()) / 30 if allv.size else 1e-18
                mm = np.where(pos, m, floor)
                ax.bar(x + j * width, mm, width, yerr=np.where(pos, e, 0),
                       color=PS.COLORS[j % len(PS.COLORS)],
                       label=f.replace("_", " "), capsize=2,
                       hatch="" if pos.all() else "//")
                for xi, p in zip(x + j * width, pos):
                    if not p:
                        ax.text(xi, floor * 1.4, "0", ha="center", fontsize=6,
                                rotation=90)
            ax.set_yscale("log")
            ax.set_xticks(x + 0.4 - width / 2)
            ax.set_xticklabels([a.replace("_", "\n") for a in archs], fontsize=6.5,
                               rotation=0)
            ax.set_title(f"{title} - {rc.replace('_',' ')}", fontsize=8.5)
            ax.legend(fontsize=7)
        PS.save(fig, fname, outdir=figdir)
        cap(fname, "architecture_controls_seed_rows.csv",
            "deterministic_finite_feature",
            f"Architecture controls, panel {title}. Bars are means over "
            "independent named seeds with standard-deviation whiskers; hatched "
            "bars marked '0' are exact floating-point zeros drawn at an "
            "off-scale floor rather than on the log axis. Panel (c) is a "
            "non-discriminating implementation check (Proposition 3.5).")


def fig_admissibility(outdir, figdir):
    """Appendix C: physical admissibility versus feature width."""
    df = load(outdir, "darboux_physical_admissibility.csv")
    if df is None or df["n_features"].nunique() < 3:
        return
    classes = sorted(df["regulator_class"].unique())
    fig, axes = PS.new_figure(2, len(classes),
                              figsize=(4.4 * len(classes), 6.6), squeeze=False)
    for col, rc in enumerate(classes):
        sub = df[df["regulator_class"] == rc]
        for i, (arch, g) in enumerate(sub.groupby("architecture")):
            p = (g.groupby("n_features")
                 .agg(star=("star_compatibility_defect", "mean"),
                      star_sd=("star_compatibility_defect", "std"),
                      eig=("polarization_min_eigenvalue", "mean"),
                      eig_sd=("polarization_min_eigenvalue", "std"),
                      n=("seed_name", "size")).reset_index())
            c = PS.COLORS[i % len(PS.COLORS)]
            _errbar(axes[0][col], p["n_features"], p["star"],
                    p["star_sd"] / np.sqrt(p["n"].clip(lower=1)),
                    arch.replace("_", " "), c,
                    PS.MARKERS["exact_symbolic"])
            # min eigenvalue is signed: a linear axis, with zero marked
            axes[1][col].errorbar(p["n_features"], p["eig"],
                                  yerr=p["eig_sd"] / np.sqrt(p["n"].clip(lower=1)),
                                  fmt=PS.MARKERS["exact_symbolic"] + "-", color=c,
                                  capsize=2, ms=4.5, label=arch.replace("_", " "))
        axes[0][col].set_xscale("log")
        axes[0][col].set_yscale("log")
        axes[0][col].set_xlabel(r"feature width $N$")
        axes[0][col].set_title(f"(a) involution defect - {rc.replace('_',' ')}",
                               fontsize=8.5)
        axes[1][col].set_xscale("log")
        axes[1][col].axhline(0.0, color="0.3", ls="--", lw=1.0)
        axes[1][col].annotate("normalizable above", xy=(0.02, 0.06),
                              xycoords="axes fraction", fontsize=7, color="0.3")
        axes[1][col].set_xlabel(r"feature width $N$")
        axes[1][col].set_title(
            f"(b) polarization min eigenvalue - {rc.replace('_',' ')}",
            fontsize=8.5)
    axes[0][0].set_ylabel(r"$\|\bar S C - C S\|_F/\|S\|_F$")
    axes[1][0].set_ylabel(r"$\min\,\mathrm{eig}\;G$")
    axes[0][-1].legend(fontsize=6, ncol=1)
    PS.save(fig, "darboux_admissibility_vs_width.pdf", outdir=figdir)
    cap("darboux_admissibility_vs_width.pdf",
        "darboux_physical_admissibility.csv", "exact_symbolic",
        "Physical admissibility of the canonicalizing map versus feature width "
        "(appendix C). Panel (a): the involution defect of eq. (3.32) on a log "
        "axis. Panel (b): the minimum eigenvalue of the induced "
        "positive-frequency Gram form on a LINEAR axis, since it is signed; the "
        "dashed line is zero, above which a normalizable Fock vacuum exists. "
        "Bars are standard errors over independent named seeds.")


def fig_tt(outdir, figdir):
    df = load(outdir, "sampled_tt_central_charge.csv")
    if df is None:
        return
    sub = df[(df["architecture"] == "selected_log_kernel")
             & (df["fit_window_index"] == 0)]
    fig, ax = PS.new_figure(figsize=(6.4, 4.0))
    labels, vals, errs, colors = [], [], [], []
    for i, (_, r) in enumerate(sub.iterrows()):
        labels.append(f"{r['regulator_class'][:6]}\n{r['estimator_variant'][:6]}\n"
                      f"N={int(r['n_features'])}")
        vals.append(r["c_exp"])
        errs.append(r["c_exp_stderr"])
        colors.append(PS.COLORS[0] if r["estimator_variant"] == "wick_factorized"
                      else PS.COLORS[1])
    x = np.arange(len(vals))
    ax.errorbar(x, vals, yerr=errs, fmt=PS.MARKERS["sampled_finite_network"],
                ls="none", capsize=3, color="k")
    for xi, v, e, c in zip(x, vals, errs, colors):
        ax.plot([xi], [v], PS.MARKERS["sampled_finite_network"], color=c, ms=7)
    ax.axhline(1.0, color=PS.COLORS[2], ls="--", lw=1.0)
    ax.annotate("$c=1$", xy=(x[-1], 1.0), xytext=(-4, 5),
                textcoords="offset points", ha="right", color=PS.COLORS[2],
                fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(r"$c_{\rm exp}$")
    ax.set_title("Sampled neural-ensemble $\\langle TT\\rangle$ central charge",
                 fontsize=9)
    PS.save(fig, "sampled_tt_central_charge.pdf", outdir=figdir)
    cap("sampled_tt_central_charge.pdf", "sampled_tt_central_charge.csv",
        "sampled_finite_network",
        "Sampled $\\langle TT\\rangle$ central charge with block standard errors "
        "over independent named seeds (blue: Wick-factorized, with the Gaussian "
        "factorization disclosed as an analytic input; red: fully direct "
        "four-field Monte Carlo). Dashed line: the free-boson value $c=1$.")


def fig_ward(outdir, figdir):
    s = load(outdir, "sampled_ward_summary.csv")
    if s is not None:
        fig, ax = PS.new_figure(figsize=(6.4, 4.0))
        sub = s[s["architecture"] == "selected_log_kernel"]
        for i, ((rc, cname), g) in enumerate(sub.groupby(["regulator_class",
                                                          "configuration"])):
            g = g[g["n_features"] == g["n_features"].max()].sort_values("n_networks")
            _errbar(ax, g["n_networks"], g["stress_absolute_error_mean"],
                    g["stress_absolute_error_stderr"],
                    f"{cname} / {rc[:6]}", PS.COLORS[i % len(PS.COLORS)],
                    PS.MARKERS["sampled_finite_network"])
        gg = sub.groupby("n_networks")["stress_absolute_error_mean"].mean()
        PS.annotate_slope(ax, gg.index.to_numpy(), gg.to_numpy(), -0.5,
                          r"$M^{-1/2}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"number of network realizations $M$")
        ax.set_ylabel("stress Ward absolute error")
        ax.legend(fontsize=7, ncol=2)
        PS.save(fig, "sampled_ward_convergence.pdf", outdir=figdir)
        cap("sampled_ward_convergence.pdf", "sampled_ward_summary.csv",
            "sampled_finite_network",
            "Sampled finite-network stress Ward error versus the number of "
            "independent realizations $M$, for the neutral 2-, 3- and 4-vertex "
            "configurations. Bars are block standard errors over independent "
            "named seeds; the guide is the $M^{-1/2}$ Monte Carlo rate.")

        fig, ax = PS.new_figure(figsize=(6.4, 4.0))
        top = s[(s["n_networks"] == s["n_networks"].max())
                & (s["n_features"] == s["n_features"].max())]
        archs = sorted(top["architecture"].unique())
        x = np.arange(len(archs))
        for j, (rc, g) in enumerate(top.groupby("regulator_class")):
            m = g.groupby("architecture")["stress_absolute_error_mean"].mean()
            e = g.groupby("architecture")["stress_absolute_error_stderr"].mean()
            ax.bar(x + 0.4 * j, m.reindex(archs), 0.38, yerr=e.reindex(archs),
                   color=PS.COLORS[j], label=rc.replace("_", " "), capsize=2)
        ax.set_yscale("log")
        ax.set_xticks(x + 0.2)
        ax.set_xticklabels([a.replace("_", "\n") for a in archs], fontsize=6.5)
        ax.set_ylabel("stress Ward absolute error")
        ax.legend(fontsize=7)
        PS.save(fig, "ward_controls.pdf", outdir=figdir)
        cap("ward_controls.pdf", "sampled_ward_summary.csv",
            "sampled_finite_network",
            "Ward controls at the largest $N$ and $M$: the chiral/angular test "
            "separates the selected architecture from the radial and angular "
            "controls. Bars are block standard errors over independent seeds.")


def fig_ns(outdir, figdir):
    df = load(outdir, "ns_frozen_metrics.csv")
    if df is None:
        return
    sub = df[df["architecture"] == "selected_log_kernel"]
    fig, ax = PS.new_figure(figsize=(6.2, 4.0))
    for i, (rc, g) in enumerate(sub.groupby("regulator_class")):
        g = g.sort_values("n_features")
        _errbar(ax, g["n_features"], g["raw_car_defect_mean"],
                g["raw_car_defect_stderr"], f"raw CAR defect / {rc[:6]}",
                PS.COLORS[i], PS.MARKERS["deterministic_finite_feature"])
        _errbar(ax, g["n_features"], g["fermionic_pairing_leakage_mean"],
                g["fermionic_pairing_leakage_stderr"],
                f"fermionic leakage / {rc[:6]}", PS.COLORS[i + 2],
                PS.MARKERS["analytic_closed_kernel"])
    gg = sub.groupby("n_features")["raw_car_defect_mean"].mean()
    PS.annotate_slope(ax, gg.index.to_numpy(), gg.to_numpy(), -0.5, r"$N^{-1/2}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"feature width $N$")
    ax.set_ylabel("NS pairing defect")
    ax.legend(fontsize=7)
    PS.save(fig, "ns_pairing_convergence.pdf", outdir=figdir)
    cap("ns_pairing_convergence.pdf", "ns_frozen_metrics.csv",
        "deterministic_finite_feature",
        "Pre-canonicalization Neveu--Schwarz pairing convergence: raw CAR defect "
        "and fermionic leakage versus feature width, with block standard errors "
        "over independent named seeds and the $N^{-1/2}$ guide.")

    fig, ax = PS.new_figure(figsize=(6.2, 4.0))
    fields = [("raw_LG_residual_mean", "raw $[L,G]$"),
              ("raw_GG_residual_mean", "raw $\\{G,G\\}$"),
              ("canonical_LG_residual_mean", "canonical $[L,G]$"),
              ("canonical_GG_residual_mean", "canonical $\\{G,G\\}$")]
    allv = sub[[f for f, _ in fields]].to_numpy(dtype=float)
    allv = allv[np.isfinite(allv) & (allv > 0)]
    floor = float(allv.min()) / 30 if allv.size else 1e-18
    archs = sorted(sub["regulator_class"].unique())
    x = np.arange(len(archs))
    for j, (f, lab) in enumerate(fields):
        m = sub.groupby("regulator_class")[f].mean().reindex(archs).to_numpy()
        pos = m > 0
        ax.bar(x + j * 0.2, np.where(pos, m, floor), 0.19,
               color=PS.COLORS[j], label=lab)
        for xi, p in zip(x + j * 0.2, pos):
            if not p:
                ax.text(xi, floor * 1.5, "0", ha="center", fontsize=6, rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels([a.replace("_", "\n") for a in archs], fontsize=7)
    ax.set_ylabel("super-Virasoro residual")
    ax.legend(fontsize=7)
    PS.save(fig, "ns_super_virasoro_residuals.pdf", outdir=figdir)
    cap("ns_super_virasoro_residuals.pdf", "ns_frozen_metrics.csv",
        "deterministic_finite_feature",
        "Raw and canonicalized $\\mathcal N=1$ super-Virasoro residuals. Exact "
        "floating-point zeros are drawn at an off-scale floor and labelled '0'.")


def fig_central_offset(outdir, figdir):
    df = load(outdir, "central_offset_mode_extension.csv")
    if df is None:
        return
    fig, ax = PS.new_figure(figsize=(6.4, 4.2))
    for i, (src, g) in enumerate(df.groupby("source")):
        gg = g.groupby("m")["modewise_c"].agg(["mean", "std", "size"]).reset_index()
        ax.errorbar(gg["m"], gg["mean"],
                    yerr=gg["std"] / np.sqrt(gg["size"].clip(lower=1)),
                    fmt=(PS.MARKERS["exact_symbolic"] if src == "canonical_truncated"
                         else PS.MARKERS["deterministic_finite_feature"]),
                    color=PS.COLORS[i], label=src.replace("_", " "), capsize=2)
    mm = np.array(sorted(df["m"].unique()), dtype=float)
    A = float(df["model_prediction"].notna().sum() and 0.1821971014211161)
    ax.plot(mm, 1.0 + ((-1.0) ** mm) * A / (mm * (mm**2 - 1) / 12.0),
            "--", color="0.4", lw=1.0,
            label=r"legacy offset model $1+(-1)^mA/[m(m^2-1)/12]$")
    ax.axhline(1.0, color=PS.COLORS[2], ls=":", lw=1.0)
    ax.set_xlabel(r"interior mode $m$")
    ax.set_ylabel(r"modewise anomaly ratio $c_m$")
    ax.legend(fontsize=7)
    PS.save(fig, "raw_modewise_central_charge_offset_model.pdf", outdir=figdir)
    cap("raw_modewise_central_charge_offset_model.pdf",
        "central_offset_mode_extension.csv",
        "exact_symbolic + deterministic_finite_feature",
        "Extended modewise anomaly ratio $c_m$. Squares: the exact_symbolic null "
        "from the canonical truncated Sugawara algebra. Circles: the raw "
        "empirical finite-feature measurement with seed standard errors. Dashed: "
        "the legacy one-parameter alternating offset model.")


def fig_vertex_cumulant(outdir, figdir):
    vw = load(outdir, "sampled_vertex_weights.csv")
    if vw is not None:
        fig, ax = PS.new_figure(figsize=(6.2, 4.0))
        sub = vw[vw["fit_window_index"] == 0]
        for i, (rc, g) in enumerate(sub.groupby("regulator_class")):
            gg = g[g["n_features"] == g["n_features"].max()].sort_values("charge")
            ax.errorbar(gg["charge"], gg["h_fit"], yerr=gg["h_stderr"],
                        fmt=PS.MARKERS["sampled_finite_network"],
                        color=PS.COLORS[i], capsize=3, ls="none",
                        label=rc.replace("_", " "))
        a = np.linspace(float(sub["charge"].min()) * 0.9,
                        float(sub["charge"].max()) * 1.1, 100)
        ax.plot(a, a**2 / 2, "--", color="0.4", lw=1.0, label=r"$h_a=a^2/2$")
        ax.set_xlabel(r"charge $a$")
        ax.set_ylabel(r"$h_a$")
        ax.legend(fontsize=7)
        PS.save(fig, "sampled_vertex_weights.pdf", outdir=figdir)
        cap("sampled_vertex_weights.pdf", "sampled_vertex_weights.csv",
            "sampled_finite_network",
            "Sampled vertex-operator weights with block standard errors over "
            "independent named seeds, against the theory curve $h_a=a^2/2$.")

    cu = load(outdir, "finite_width_cumulant.csv")
    if cu is not None:
        fig, ax = PS.new_figure(figsize=(6.2, 4.0))
        for i, (rc, g) in enumerate(cu.groupby("regulator_class")):
            g = g.sort_values("n_features")
            _errbar(ax, g["n_features"], np.abs(g["cumulant4"]),
                    g["cumulant4_stderr"], rf"$|\kappa_4|$ / {rc[:6]}",
                    PS.COLORS[i], PS.MARKERS["sampled_finite_network"])
            ax.plot(g["n_features"], np.abs(g["kappa4_prediction"]), ":",
                    color=PS.COLORS[i], lw=1.0,
                    label=rf"theory $3\sigma_\phi^4/8N$ / {rc[:6]}")
            p = float(g["fit_exponent"].iloc[0])
            PS.annotate_slope(ax, g["n_features"], np.abs(g["cumulant4"]), p,
                              rf"fit $p={p:.3f}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"feature width $N$")
        ax.set_ylabel(r"$|\kappa_4(\phi_N)|$")
        ax.legend(fontsize=7)
        PS.save(fig, "finite_width_cumulant.pdf", outdir=figdir)
        cap("finite_width_cumulant.pdf", "finite_width_cumulant.csv",
            "sampled_finite_network",
            "Connected fourth cumulant of the neural field versus width, with "
            "block standard errors, the fitted exponent, and the exact "
            "finite-width prediction $\\kappa_4=-3\\sigma_\\phi^4/(8N)$. "
            "Interpreted only as finite-width non-Gaussianity.")


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    PS.use_style()
    for fn in (fig_radius_plateau, fig_convergence, fig_controls,
               fig_admissibility, fig_tt,
               fig_ward, fig_ns, fig_central_offset, fig_vertex_cumulant):
        try:
            fn(outdir, figdir)
        except Exception as exc:  # pragma: no cover
            print(f"  [warn] {fn.__name__}: {type(exc).__name__}: {exc}")
    write_json(Path(figdir) / "figure_captions.json", CAPTIONS)
    print(f"  wrote {len(CAPTIONS)} figures to {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
