#!/usr/bin/env python3
"""Draw the historical absolute Ward errors and new exact-phase benchmark."""
from pathlib import Path
import argparse
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "virasoro-figure-font-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "figures")
OUT = parser.parse_args().output_dir
OUT.mkdir(exist_ok=True, parents=True)
WIDTH = 5.6
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "legend.frameon": False, "lines.linewidth": 1.4, "lines.markersize": 3.8,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
    "axes.linewidth": .7, "pdf.fonttype": 42,
    "figure.constrained_layout.w_pad": .035,
    "figure.constrained_layout.h_pad": .035,
    "figure.constrained_layout.wspace": .055,
    "savefig.bbox": None,
})
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts = [x for leg in fig.legends for x in leg.get_texts()]
    for ax in fig.axes:
        texts.extend([ax.xaxis.label, ax.yaxis.label, ax.title])
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(limits)
            texts.extend(label for value, label in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
                         if lo <= value <= hi)
    for label in texts:
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, (name, label.get_text(), box)
            assert box.x1 <= canvas.x1 + 1 and box.y1 <= canvas.y1 + 1, (name, label.get_text(), box)
    fig.savefig(OUT / (name + ".pdf"), metadata={
        "Title": name.replace("_", " "), "Creator": "plot_ward_revision.py",
        "Subject": "Ward accuracy and exact phase integration, 12 November 2026"})
    plt.close(fig)


d = pd.read_csv(ROOT / "data/deterministic_ward_normalized_summary.csv")
d = d[(d.architecture == "selected_log_kernel") & (d.regulator_class == "tapered_log_window")]
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.45), layout="constrained")
for color, config, label in zip(colors, ["two_vertex", "three_vertex", "four_vertex"],
                                ["Two vertices", "Three vertices", "Four vertices"]):
    g = d[d.configuration == config].sort_values("n_features")
    axes[0].errorbar(g.n_features, g.normalized_mean, yerr=g.normalized_stderr,
                     fmt="o-", capsize=2.2, color=color, label=label)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel(r"Feature width $N$")
axes[0].set_ylabel("Relative RMS Ward error")
axes[0].set_title("(a) Gaussian covariance proxy", fontsize=9.5)
b = pd.read_csv(ROOT / "data/deterministic_ward_normalized_blocks.csv")
b = b[(b.regulator_class == "tapered_log_window") & (b.n_features == 200000)]
p = b.pivot(index=["seed_name", "configuration"], columns="architecture",
            values="normalized_l2_pooled")
logs = np.log(p.rbf_gaussian_prior / p.selected_log_kernel).groupby("seed_name").mean()
mean, se = logs.mean(), logs.std(ddof=1) / np.sqrt(len(logs))
lo, hi = np.exp(mean + np.array([-1, 1]) * t.ppf(.975, 7) * se)
axes[1].axhspan(lo, hi, color="0.9", zorder=0)
axes[1].axhline(np.exp(mean), color="0.35", lw=1.2)
axes[1].plot(np.arange(1, 9), np.exp(logs.to_numpy()), "o", color="0.1")
axes[1].set_xticks(np.arange(1, 9))
axes[1].set_xlabel("Feature seed")
axes[1].set_ylabel("RBF / Log-Kernel error")
axes[1].set_title(r"(b) Paired ratio at $N=2\times10^5$", fontsize=9.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=3,
           columnspacing=.9, handlelength=1.25)
save(fig, "ward_independent_blocks")

conditional = pd.read_csv(ROOT / "ward_exact_summary.csv")
ensemble = pd.read_csv(ROOT / "ward_full_ensemble_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5), sharey=True, layout="constrained")
handles = []
for color, cutoff in zip(colors, [30, 100, 300]):
    c = conditional[conditional.lambda_uv == cutoff].sort_values("n_features")
    e = ensemble[ensemble.lambda_uv == cutoff].sort_values("n_features")
    axes[0].errorbar(c.n_features, c.exact_total_error_mean, yerr=c.exact_total_error_se,
                     fmt="o-", capsize=2.2, color=color)
    axes[1].errorbar(e.n_features, e.full_ratio_relative_rms, yerr=e.rms_error_jackknife_se,
                     fmt="o-", capsize=2.2, color=color)
    for ax in axes:
        ax.axhline(c.regulator_error_mean.iloc[0], color=color, ls="--", lw=1, alpha=.8)
    handles.append(Line2D([], [], color=color, marker="o",
                          label=r"$\Lambda_{\rm UV}=" + str(cutoff) + "$"))
handles.append(Line2D([], [], color="0.35", ls="--", label="Regulator bias"))
for ax in axes:
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Feature width $N$")
    ax.set_ylim(.004, 2.5)
axes[0].set_ylabel("Relative RMS Ward error")
axes[0].set_title("(a) Conditioned fields", fontsize=10.0)
axes[1].set_title("(b) Ensemble ratio", fontsize=10.0)
fig.legend(handles=handles, loc="outside upper center", ncol=2,
           columnspacing=1.2, handlelength=1.8)
save(fig, "exact_phase_ward")
print("Wrote two vector Ward figures at manuscript width.")
