BOUNDARY NUMERICAL REPRODUCTION

Verify files from this directory:
    python3 verify_boundary_evidence.py

Regenerate the calculation:
    python3 scripts/boundary_checks.py

The script uses seed 20261107, 32 independent batches and 16,384 momenta
per batch. The saved NPZ contains batches and references; scalar_image.csv
contains their means and standard errors. The fermion curve uses 32 Cauchy
modes. Outputs are written beneath evidence/boundary/ and figures/.
Install the recorded numerical package versions from requirements.txt.
