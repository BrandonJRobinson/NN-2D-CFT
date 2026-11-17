FIGURE TYPOGRAPHY SOURCES - 12 NOVEMBER 2026

Run with Python 3, matplotlib, numpy and pandas:

    python3 evidence/figure_sources/redraw_figures.py

Run from the manuscript directory, or use an absolute path to the script.
The default output is evidence/figures/. Optional --preview-dir DIRECTORY
writes PNG previews. No LaTeX or network simulations are run.

The seven CSV files in data/ are byte-identical saved outputs from the
September evidence release. INPUTS.json records their original relative
locations and SHA-256 hashes. The script verifies these inputs before use.
Original generation commit: 2900ae4aa10ff33ecf88a527e9cf633f0185f98c
Configuration hash: f51ad38719b59543

Figures 1, 2, 3, 4 and 6 use a 5.6-inch canvas, matched to their manuscript
insertion width. Axis labels are 10.5 points, and ticks and legends 9.5
points before the negligible final scaling. Legends are moved outside the
plots and shared across panels where appropriate. Internal code identifiers
are replaced by readable legend labels. The plotted values, error bars,
scales, panel order and sampling units are preserved.

In the historical gate01 CSV, delta_A is the positive-frequency self-pairing
error, denoted delta_iso in the manuscript. The historical delta_iso column
is the normalized cross-block error, denoted delta_G in the manuscript.
The script maps those column names to the manuscript's notation explicitly.

The direct and Wick TT curves use all twelve independent four-geometry
block vectors. Standard errors and the fit-centered residuals are computed
from the same block covariance as before. No significance tests are altered.
