VIRASORO RECOVERED-PHYSICS CONSISTENCY CHECKS
Review candidate, 7 November 2026

This supplement checks the local formulas restored after comparison with
the two original manuscripts. It complements the unchanged bulk archive
and the separate 7 November boundary supplement. It contains no additional
Monte Carlo data and does not claim full bosonization, modular consistency,
BRST cohomology, or a complete Euclidean ghost measure.

The script independently derives the bc and beta-gamma conformal weights
and central charges by graded Wick contractions. It checks the Majorana
stress and Dirac current coefficients, the native momentum feature
identity, the Ramond geometric kernel and point-split ground-state weight,
the Clifford CAR and parity on a finite mode space, and 24 angular Bessel
identities by quadrature.

Verify the distributed file hashes first:
  python3 verify_files.py

Install the listed requirements in a suitable environment, then run from
the supplement root:
  python3 scripts/check_recovered_physics.py

The script overwrites audit/recovered_physics_checks.json with its output.
Mathematical expressions in the JSON are exact symbolic results; the
remaining values are finite-mode or quadrature residuals. The complete
manuscript source is distributed separately.

Public deposition and licensing remain subject to author approval together
with the main evidence release. This package has not been deposited.
