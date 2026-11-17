# Neural Network Models of 2D Conformal Field Theory

Code and computational evidence for Brandon Robinson,
*Virasoro Symmetry in Neural Network Field Theories*
([arXiv:2512.24420](https://arxiv.org/abs/2512.24420)).

The November 2026 update accompanies the revised JHEP manuscript. Its central
result is the construction of the chiral current pairing and Virasoro
representation from specified neural architectures and parameter laws.
The April scripts, notebooks and figures remain in their original locations.
Their original documentation is preserved in [legacy/april_2026](legacy/april_2026/).

## Contents

- [src/nnft_virasoro](src/nnft_virasoro/), scripts, configs and tests:
  the complete September code export from generation commit
  2900ae4aa10ff33ecf88a527e9cf633f0185f98c. PROVENANCE.json records
  all 78 exact source hashes and original Git blob identifiers.
- [September results](results/jhep_submission_hardening/):
  all 377 original manifested artifacts, with configuration hash
  f51ad38719b59543, preserved byte-for-byte.
- [Earlier frozen results](results/jhep_frozen/) and the historical frozen
  manuscript: the earlier baseline, retained with its original provenance.
- [November 7 review](review_20261107/): independent regulator and finite-block
  statistical checks. CORRECTIONS.md records the TT calibration correction
  separately from the original data. The boundary and recovered_physics
  subdirectories contain boundary evidence and exact sector checks.
- [November Fock and Ward evidence](evidence/major_revision/):
  twenty saved pairing matrices, finite Fock-projection diagnostics,
  historical absolute Ward errors and the exact phase-integrated benchmark.
  Conditional-field and full-ensemble statistics are saved separately.
- [Four-point width evidence](evidence/four_point_width/): saved cumulant
  data and source code. The connected scalar four-point function scales
  as 1/N at fixed regulator; the full function approaches Wick contractions.
- [Current figures](evidence/figures/) and
  [figure sources](evidence/figure_sources/): vector figures, plotted data,
  and redraw scripts, preserving the enlarged labels.

Historical reports and Draft-JHEP-Frozen.tex describe earlier stages and
contain claims superseded by the revised manuscript. Interpret them with
the dated corrections and November evidence.

## Verify without simulation

    git clone https://github.com/BrandonJRobinson/NN-2D-CFT.git
    cd NN-2D-CFT
    python3 verify_release.py
    python3 review_20261107/boundary/verify_boundary_evidence.py
    python3 review_20261107/recovered_physics/verify_files.py

The root verifier checks the current payload, all 377 original September
artifacts and all 78 exact exported sources using only the standard library.
Original data retain their generation provenance.

## Installation and scientific tests

Use Python 3.10 or newer for the current package.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python verify_release.py --restore-archive-times
    python -m pytest tests -q

Timestamp normalization follows checksum verification, accommodates the
legacy figure-freshness test after checkout or extraction, and changes no
content. The original suite contains 215 tests. Archive-only instructions
are in INSTALLATION.md.

## Reproduce analyses

    python review_20261107/independent_audit.py

For the current Fock, phase-quadrature, exact symbolic and four-point checks, run:

    python review_20261117/verify_current.py

This writes only beneath review_20261117/rerun/. The full phase benchmark is separate.
The completed checks are recorded in [VALIDATION.json](review_20261117/VALIDATION.json).

The November 7 audit recalculates finite-block TT inference and regulator checks from saved
inputs and writes only beneath review_20261107/rerun/.

The saved tapered-window fourth-cumulant exponent is -0.9951 +/- 0.0092
over widths 8, 16, 32, 64 and 128, consistent with the exact
kappa_4 = -3 sigma_phi^4/(8 N). Standardized excess kurtosis is -3/(2 N).

The [November analysis instructions](evidence/major_revision/README.txt)
describe the Fock and exact-phase calculations and record their package
versions. The complete prescribed exact-phase benchmark may take time;
every saved seed/probe and the independent phase-quadrature checks are
included. Regenerate outputs in a fresh copy to retain the frozen evidence.

The full September production runner is:

    python scripts/run_jhep_submission_hardening.py --config configs/jhep_submission_hardening.yaml

Production campaigns can take substantial compute and overwrite configured
outputs. Run them in a fresh copy. The quick option is a smoke run.

## Interpretation

Finite Fock readouts use the explicit pairing projection and require a
positive Hermitian mixed block. Nineteen of the twenty saved projected
readouts are positive; the remaining sample is retained as a failure.
Raw kernel and Ward comparisons use the original neural ensemble.

The historical Gaussian Ward proxy has appreciable absolute errors.
The exact-phase benchmark distinguishes conditional errors from the
weighted full-ensemble ratio, with its prescribed probes and regulator.

Finite-width cumulants measure non-Gaussianity; a local-interaction
interpretation requires additional physical input. Exact operator statements
refer to the limiting chiral construction and stated finite-level support.

## Citation and provenance

Use CITATION.cff and cite the full repository commit used together with the
manuscript. No archive DOI is asserted. The WZW extension is Capuozzo,
Robinson and Suzzoni, [Learning from Duality in Neural Network Field
Theories, arXiv:2611.21503](https://arxiv.org/abs/2611.21503).

The provenance directory retains the dated local November 7 deposit-candidate
metadata, including its historical licensing proposal. This update introduces
no new license grant.
