SECTOR CONSISTENCY CHECKS

Verify files:
    python3 verify_files.py

Recalculate:
    python3 scripts/check_recovered_physics.py

The script checks graded contractions, conformal weights, central charges,
Clifford CAR and parity, Ramond kernels, and angular Bessel quadrature.
Results are written to audit/recovered_physics_checks.json. Numerical
package versions are recorded in requirements.txt.
