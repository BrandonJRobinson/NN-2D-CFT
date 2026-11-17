"""Plot boundary kernels and coefficient stability from saved numerical data."""
from pathlib import Path
import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "virasoro-figure-font-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11,
                         "legend.fontsize": 9, "pdf.fonttype": 42})

    data = pd.read_csv(ROOT / "verification/boundary/evidence/boundary/scalar_image.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), layout="constrained")
    log_height = np.log(data.height / .4)
    axes[0].errorbar(log_height, data["mean"], yerr=data.standard_error,
                    fmt="o", capsize=3, label="Fourier features")
    axes[0].plot(log_height, data.regulated, label="Regulated kernel")
    axes[0].plot(log_height, data.continuum, "--", label="Continuum")
    axes[0].set(xlabel=r"$\log(y/y_0)$", ylabel=r"$V_+(y;y_0)$")
    axes[0].legend()
    x = np.geomspace(.1, 3, 60)
    height = .05
    u, v = x + 1j * height, -1j * height
    truncated = (1 - (v / u)**32) / (u - v)
    axes[1].loglog(x, np.abs(truncated), "o", ms=3, label="32 Cauchy modes")
    axes[1].loglog(x, 1 / np.sqrt(x*x + 4*height*height), label="Image kernel")
    axes[1].loglog(x, 1 / x, "--", label=r"$1/|x|$")
    axes[1].set(xlabel=r"$x$", ylabel=r"$|S_{LR}|$")
    axes[1].legend()
    fig.savefig(output / "boundary_kernels.pdf")
    plt.close(fig)

    data = pd.read_csv(ROOT / "results/jhep_submission_hardening/gate02_stability_epsilon_sweep.csv")
    reference = data[data.direction_index == 0].sort_values("pairing_error_frobenius")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), layout="constrained")
    for _, direction in data.groupby("direction_index"):
        direction = direction.sort_values("pairing_error_frobenius")
        axes[0].loglog(direction.pairing_error_frobenius,
                       direction.excess_virasoro_residual_pbw, color=".75", lw=.8)
        axes[1].semilogx(direction.pairing_error_frobenius,
                        direction.residual_over_pairing_plus_square, color=".75", lw=.8)
    axes[0].loglog(reference.pairing_error_frobenius,
                   reference.excess_virasoro_residual_pbw, "o-", label="Reference direction")
    axes[1].semilogx(reference.pairing_error_frobenius,
                    reference.residual_over_pairing_plus_square, "o-", label="Reference direction")
    axes[0].set(xlabel=r"$\|E\|_F$", ylabel=r"$\|\Delta\mathcal{R}\|_{\mathrm{PBW}}$")
    axes[1].set(xlabel=r"$\|E\|_F$", ylabel=r"$\|\Delta\mathcal{R}\|/(\|E\|_F+\|E\|_F^2)$")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend()
    fig.savefig(output / "pbw_epsilon_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
