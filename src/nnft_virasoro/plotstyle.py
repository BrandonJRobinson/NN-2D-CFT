"""One shared publication figure style for the frozen suite.

Figure rules (mission section 4): log axes only on strictly positive data,
fitted lines and theory guides shown, slopes labelled in-panel, uncertainty
bars present, deterministic values visually distinguished from sampled ones,
never a floating-point zero on a log axis, and every caption naming the
estimator class and the source CSV.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIGDIR = Path(__file__).resolve().parents[2] / "figures_jhep_frozen"

STYLE = {
    "figure.figsize": (6.2, 4.2),
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "mathtext.fontset": "cm",
    "text.usetex": False,
}

#: Colour-blind-safe qualitative palette.
COLORS = ["#0b6fa4", "#d1495b", "#3f7d20", "#8c5fc7", "#c98a00", "#0f8b8d",
          "#7a5230", "#555555", "#b5179e"]

#: Marker convention distinguishing estimator classes.
MARKERS = {
    "analytic_closed_kernel": "s",
    "deterministic_finite_feature": "o",
    "sampled_finite_network": "^",
    "exact_symbolic": "D",
}


def use_style() -> None:
    plt.rcParams.update(STYLE)


def new_figure(*args, **kwargs):
    use_style()
    return plt.subplots(*args, **kwargs)


def positive_only(y, floor: float | None = None):
    """Mask non-positive / non-finite values so nothing lands on a log axis."""
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & (y > 0)
    if floor is not None:
        mask &= y > floor
    return mask


def annotate_slope(ax, x, y, slope, label, *, color="0.35", ls="--"):
    """Draw a reference power law through the geometric centre and label it."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = positive_only(x) & positive_only(y)
    if ok.sum() < 2:
        return
    xg = np.exp(np.mean(np.log(x[ok])))
    yg = np.exp(np.mean(np.log(y[ok])))
    xs = np.array([x[ok].min(), x[ok].max()])
    ys = yg * (xs / xg) ** slope
    ax.plot(xs, ys, ls=ls, color=color, lw=1.0, zorder=1)
    ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8, color=color)


def save(fig, name: str, *, outdir: Path | None = None) -> Path:
    outdir = Path(outdir) if outdir is not None else FIGDIR
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path)
    if path.suffix == ".pdf":
        fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path
