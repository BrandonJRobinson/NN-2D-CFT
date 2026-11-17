#!/usr/bin/env python3
"""Redraw manuscript figures 1, 2, 3, 4 and 6 from saved evidence.

Run from any directory with Python, matplotlib, numpy and pandas installed.
No simulations or statistical decisions are rerun. Historical CSV column
names are translated to the notation of the revised manuscript.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "virasoro-figure-font-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=HERE.parent / "figures")
parser.add_argument("--preview-dir", type=Path)
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
PREVIEW = args.preview_dir
if PREVIEW:
    PREVIEW.mkdir(parents=True, exist_ok=True)

WIDTH = 5.6
TARGET_WIDTH_PT = 210 / 25.4 * 72 * 0.72 * 0.94
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "axes.labelpad": 5.0,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.columnspacing": 1.4,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid": True,
    "grid.alpha": 0.26,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": None,
    "figure.constrained_layout.w_pad": 0.035,
    "figure.constrained_layout.h_pad": 0.035,
    "figure.constrained_layout.wspace": 0.045,
    "figure.constrained_layout.hspace": 0.055,
})
for record in json.loads((HERE / "INPUTS.json").read_text()):
    p = HERE / record["path"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == record["sha256"], p

PLOTTED = []
REPORT = []
FIG_NUMBERS = {
    "regulator_and_pairing": 1,
    "predarboux_width": 2,
    "architecture_symmetries": 3,
    "tt_geometry": 4,
    "ns_two_constructions": 6,
}


def load(name):
    return pd.read_csv(HERE / "data" / name)


def line(ax, x, y, series, *, panel, figure, error=None, color=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    error = np.asarray(error, dtype=float) if error is not None else None
    if error is None:
        ax.plot(x, y, "o-", color=color, label=series)
    else:
        ax.errorbar(x, y, yerr=error, fmt="o-", color=color,
                    label=series, capsize=2.4, elinewidth=1.0)
    for i in range(len(x)):
        PLOTTED.append(dict(figure=figure, panel=panel, series=series,
                            x=float(x[i]), y=float(y[i]),
                            yerr=float(error[i]) if error is not None else np.nan))


def logaxes(ax, x=True, y=True):
    if x:
        ax.set_xscale("log")
    if y:
        ax.set_yscale("log")
    ax.set_axisbelow(True)


def finish(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for ax in fig.axes:
        texts = [ax.xaxis.label, ax.yaxis.label,
                 ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(limits)
            texts.extend(label for value, label in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
                         if lo <= value <= hi)
        for text in texts:
            if not text.get_visible() or not text.get_text():
                continue
            b = text.get_window_extent(renderer)
            assert b.x0 >= bounds.x0 - 1 and b.y0 >= bounds.y0 - 1, (name, text.get_text(), b)
            assert b.x1 <= bounds.x1 + 1 and b.y1 <= bounds.y1 + 1, (name, text.get_text(), b)
    scale = TARGET_WIDTH_PT / (WIDTH * 72)
    report = dict(
        figure=FIG_NUMBERS[name], file=name + ".pdf",
        width_inches=WIDTH, height_inches=float(fig.get_figheight()),
        manuscript_width_inches=TARGET_WIDTH_PT / 72,
        effective_axis_label_points=10.5 * scale,
        effective_tick_label_points=9.5 * scale,
        effective_legend_points=9.5 * scale,
        panel_count=len(fig.axes),
        text_bounds_checked=True,
    )
    REPORT.append(report)
    fig.savefig(OUT / (name + ".pdf"), metadata={
        "Title": "Virasoro neural fields: figure " + str(FIG_NUMBERS[name]),
        "Subject": "Typography revision from preserved numerical evidence",
        "Creator": "redraw_figures.py",
    })
    if PREVIEW:
        fig.savefig(PREVIEW / (name + ".png"), dpi=210)
    plt.close(fig)


def regulator_and_pairing():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1), layout="constrained")
    d = load("gate03_regulator_convergence.csv").sort_values("lambda_uv")
    line(axes[0], d.lambda_uv, d.max_abs_deviation_from_one,
         "Analytic regulator", panel=0, figure=1, color=COLORS[0])
    logaxes(axes[0])
    axes[0].set_xlabel(r"UV cutoff $\Lambda_{\rm UV}$")
    axes[0].set_ylabel(r"$\max_r\,|c_{\rm reg}(r)-1|$")

    d = load("gate01_intrinsic_admissibility_seed_rows.csv")
    d = d[d.architecture == "selected_log_kernel"]
    for color, (key, label) in zip(COLORS, [
        ("sharp_annulus", "Sharp annulus"),
        ("tapered_log_window", "Smooth taper"),
    ]):
        g = d[d.spectral_regulator_class == key].groupby("n_features").operational_pairing_error.mean()
        line(axes[1], g.index, g.values, label, panel=1, figure=1, color=color)
    logaxes(axes[1])
    axes[1].set_xlabel(r"Feature width $N$")
    axes[1].set_ylabel(r"Pairing error $\epsilon_{\rm pair}$")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.025),
                   borderaxespad=0, fontsize=9.5)
    finish(fig, "regulator_and_pairing")


def predarboux_width():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), layout="constrained")
    d = load("gate02_predarboux_virasoro_seed_rows.csv")
    d = d[(d.architecture == "selected_log_kernel") &
          (d.spectral_regulator_class == "tapered_log_window")]
    for i, (column, label) in enumerate([
        ("pairing_error_frobenius", r"Pairing error $\|E_N\|_F$"),
        ("excess_virasoro_residual_pbw", r"Virasoro residual $\|\Delta\mathcal{R}\|_{\rm PBW}$"),
    ]):
        g = d.groupby("n_features")[column].agg(["mean", "sem"])
        line(axes[i], g.index, g["mean"], "Log-Kernel", error=g["sem"],
             panel=i, figure=2, color=COLORS[0])
        logaxes(axes[i])
        axes[i].set_xlabel(r"Feature width $N$")
        axes[i].set_ylabel(label)
    finish(fig, "predarboux_width")


def architecture_symmetries():
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.35), layout="constrained")
    stationarity = load("intrinsic_stationarity_seed_rows.csv")
    pairing = load("gate01_intrinsic_admissibility_seed_rows.csv")
    pairing = pairing[pairing.spectral_regulator_class == "tapered_log_window"]
    mapping = [
        ("anisotropic_angular_prior", "Anisotropic prior"),
        ("phase_correlated_features", "Correlated phases"),
        ("rbf_gaussian_prior", "RBF prior"),
        ("selected_log_kernel", "Log-Kernel"),
    ]
    specs = [
        (stationarity, "scalar_translation_defect", r"Stationarity $\delta_{\rm TI}$"),
        (stationarity, "scalar_rotation_defect", r"Rotation $\delta_{\rm RI}$"),
        (pairing, "delta_A", r"Self-pairing $\delta_{\rm iso}$"),
        # The old code calls this column delta_iso. It is the normalized
        # cross-block error denoted delta_G in the revised manuscript.
        (pairing, "delta_iso", r"Cross-block error $\delta_G$"),
    ]
    for panel, (ax, (data, column, label)) in enumerate(zip(axes.flat, specs)):
        for color, (key, legend) in zip(COLORS, mapping):
            g = data[data.architecture == key].groupby("n_features")[column].mean()
            line(ax, g.index, g.values, legend, panel=panel, figure=3, color=color)
        logaxes(ax)
        ax.set_xlabel(r"Feature width $N$")
        ax.set_ylabel(label)
    # One readable legend serves all panels.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    axes[0, 0].yaxis.set_major_locator(FixedLocator([1e-16, 1e-12, 1e-8, 1e-4, 1]))
    axes[0, 0].yaxis.set_major_formatter(LogFormatterMathtext())
    finish(fig, "architecture_symmetries")


def tt_geometry():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), layout="constrained")
    data = load("gate03_q_ratio_blocks.csv")
    for color, (key, label) in zip(COLORS, [
        ("direct_TT", "Direct $TT$"),
        ("wick_factorized", "Wick estimate"),
    ]):
        d = data[data.estimator_variant == key]
        X = d.pivot(index="seed_name", columns="separation_index", values="q").to_numpy()
        assert X.shape == (12, 4)
        mean = X.mean(axis=0)
        S = np.cov(X, rowvar=False, ddof=1)
        se = np.sqrt(np.diag(S) / len(X))
        one = np.ones(4)
        a = np.linalg.solve(S, one)
        qhat = a @ mean / (a @ one)
        x = np.arange(4)
        line(axes[0], x, mean, label, error=se, panel=0, figure=4, color=color)
        line(axes[1], x, (mean - qhat) / se, label, panel=1, figure=4, color=color)
    axes[0].axhline(1, ls="--", color="0.35", lw=1)
    axes[1].axhline(0, ls="--", color="0.35", lw=1)
    axes[0].set_ylabel(r"Amplitude ratio $q_g$")
    axes[1].set_ylabel("Standardized residual")
    for ax in axes:
        ax.set_xticks(np.arange(4), [r"$r_1$", r"$r_2$", r"$r_3$", r"$r_4$"])
        ax.set_xlabel("Insertion geometry")
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    finish(fig, "tt_geometry")


def ns_two_constructions():
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.0), layout="constrained")
    pairing = load("ns_pairing_seed_rows.csv")
    algebra = load("ns_super_virasoro_seed_rows.csv")
    specs = [
        (pairing, "raw_car_defect", r"CAR error $\epsilon_{\rm CAR}$"),
        (algebra, "raw_LG_residual", r"$[L,G]$ residual"),
        (algebra, "raw_GG_residual", r"$\{G,G\}$ residual"),
    ]
    for panel, (ax, (data, column, label)) in enumerate(zip(axes, specs)):
        data = data[data.architecture == "selected_log_kernel"]
        for color, (key, legend) in zip(COLORS, [
            ("native_derived_from_boson", "Boson-derived"),
            ("randomized_cauchy", "Independent Cauchy"),
        ]):
            g = data[data.ns_construction == key].groupby("n_features")[column].mean()
            line(ax, g.index, g.values, legend, panel=panel, figure=6, color=color)
        logaxes(ax)
        ax.xaxis.set_major_locator(FixedLocator([1e3, 1e4, 1e5, 1e6]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_ylabel(label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    fig.supxlabel(r"Fermionic feature width $N$", fontsize=10.5)
    finish(fig, "ns_two_constructions")


regulator_and_pairing()
predarboux_width()
architecture_symmetries()
tt_geometry()
ns_two_constructions()
pd.DataFrame(PLOTTED).to_csv(OUT / "plotted_data.csv", index=False, float_format="%.17g")
(OUT / "typography_report.json").write_text(
    json.dumps(dict(matplotlib_version=matplotlib.__version__, figures=REPORT), indent=2) + "\n")
print(json.dumps(REPORT, indent=2))
