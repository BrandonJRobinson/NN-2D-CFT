# Numerical reproduction for neural 2D CFT

Code, configurations, saved data and figure-generation scripts for
[Virasoro Symmetry in Neural Network Field Theories](https://arxiv.org/abs/2512.24420).

## Install and verify

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python verify_release.py --restore-archive-times
    python -m pytest tests -q

Python 3.10 or newer is recommended. requirements-historical.txt records the
original production environment. Verification checks content hashes before
optionally normalizing file times for the figure-freshness test.

## Reproduce numerical results

Run production campaigns in a fresh checkout: they write the configured
result and figure directories and can take substantial compute.

    python scripts/run_jhep_frozen_suite.py --config configs/jhep_frozen.yaml
    python scripts/run_jhep_submission_hardening.py --config configs/jhep_submission_hardening.yaml

The first campaign contains the regulator, current, vertex, four-point and
NS calculations. The second contains the intrinsic pairing diagnostics,
coefficient residuals, direct stress-tensor estimates, Ward comparisons and
fermionic checks. The quick option runs a reduced smoke calculation.

For the finite Fock projection and exact phase-integrated Ward benchmark:

    python evidence/major_revision/analyze_revision.py
    python evidence/major_revision/summarize_full_ward.py

The benchmark configuration and saved seed/probe results are in that same
folder. Conditional-field statistics and weighted ensemble ratios are distinct.

## Verify reported numerical claims from saved inputs

    python verification/independent_audit.py
    python verification/check_current.py
    python verification/boundary/verify_boundary_evidence.py
    python verification/sectors/verify_files.py

The first two commands write their recalculations to verification/rerun/.
The boundary generation script and its seed, batches and regulated references
are under verification/boundary/. Sector checks are under verification/sectors/.

The connected scalar four-point function scales as 1/N at fixed regulator.
Its exact equal-point cumulant is -3 sigma_phi^4/(8 N), with normalized
excess kurtosis -3/(2 N). Saved measurements and the fitted exponent
-0.9951 +/- 0.0092 are in results/jhep_frozen/finite_width_cumulant*.csv.

The TT calculation uses twelve independent four-geometry blocks and the
finite-block covariance calibration in verification/tt_finite_block_audit.json.
The twenty saved bosonic pairings produce nineteen positive projected Fock
readouts; the remaining sample is retained as a failure.

## Regenerate figures from saved data

    python scripts/reproduce_figures.py --output-dir reproduction/figures

This regenerates all nine current figure assets using the saved numerical
inputs. It does not rerun a production campaign. Individual plot recipes
are in evidence/figure_sources/ and evidence/major_revision/.
Original notebook experiments and their figures remain in experiments/ and
figures/. Baseline figure recipes are also available in scripts/make_figures.py.

## File layout

- src/, scripts/, configs/, tests/: numerical implementation and checks.
- results/: production data and numerical manifests.
- evidence/: current figure inputs, Fock/Ward data and plot recipes.
- verification/: independent numerical and sector checks.
- PROVENANCE.json and RELEASE_MANIFEST.json: generation identifiers and hashes.

Use CITATION.cff and the exact repository commit when citing these calculations.
