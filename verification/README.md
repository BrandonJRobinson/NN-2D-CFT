# Numerical verification

From the repository root:

    python verification/independent_audit.py
    python verification/check_current.py
    python verification/width_bootstrap.py

The independent audit recalculates the TT covariance and finite-block tests,
checks the regulated Bessel kernels and bounds, and exercises the Fock
canonicalizer. The current checks verify saved pairing projections, exact
phase quadrature, the four-point width fit and sector identities. Recalculated
outputs go to verification/rerun/.

The width-bootstrap command recomputes the equal-weight average of eight
width exponents, resampling shared seeds within the control and NS families.
It verifies the original individual fits first, then writes the summary and
all 1000 replicates beside the script; --output-dir selects another location.

Boundary data, the generation script and its manifest are in boundary/.
The sector-check script and its reference outputs are in sectors/.

For the TT ratios, the joint four-geometry Hotelling tests and the
three-dimensional geometry-contrast F test use all twelve independent blocks.
The direct contrast-test p-value is 0.125043. Geometries within a block are
correlated. Numerical outputs are in tt_finite_block_audit.json.
