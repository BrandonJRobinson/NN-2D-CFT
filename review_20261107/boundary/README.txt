VIRASORO BOUNDARY EVIDENCE SUPPLEMENT - REVIEW CANDIDATE
7 November 2026

This dated supplement supplies the boundary calculations in section 8 and
appendix G of the restored JHEP manuscript. The September bulk archive is
separate. This supplement does not reproduce or validate the older boundary
figures and high-precision assertions; they are not used for its results.

Contents
  scripts/boundary_checks.py        Complete generation and checking script.
  evidence/boundary/boundary_batches.npz
                                   All 32 scalar batches and reference arrays.
  evidence/boundary/scalar_image.csv
                                   Human-readable scalar means and references.
  evidence/boundary/boundary_checks.json
                                   Numerical results and declared scope.
  evidence/boundary/boundary_kernels.png
  manuscript/evidence/figures/boundary_kernels.pdf
                                   The two-panel manuscript figure.
  manuscript/*.tex                 Boundary text and methods, requiring the
                                   separate complete manuscript to compile.
  requirements.txt                 Versions used for this calculation.
  manifest.json                    SHA256 hashes, sizes, environment, status.

To check the supplied files, use the Python standard library:
  python3 verify_boundary_evidence.py

To regenerate the calculation from the supplement root, install the
requirements in a suitable Python environment and run:
  python3 scripts/boundary_checks.py
The calculation uses seed 20261107. It overwrites the generated outputs.
Figure metadata can change on regeneration; numerical arrays and results
are the scientific comparison. Check the distributed hashes before rerunning.

The scalar statistic uses exact phase averaging and independently sampled
frequency/angle batches. Common momenta correlate the heights within each
batch. Its errors are frequency-sampling errors, not finite-ensemble
stress-tensor or Wick errors. The fermion curve is a deterministic 32-term
Cauchy expansion. No independent supersymmetry Ward measurement, annulus
sewing test, or finite-width QFT reconstruction is asserted.

The supplement is prepared for author review. Public deposition and the
license must be approved together with the accompanying main release.
