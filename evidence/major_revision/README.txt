FOCK AND WARD NUMERICAL REPRODUCTION

Run from the repository root:
    python3 evidence/major_revision/analyze_revision.py
    python3 evidence/major_revision/summarize_full_ward.py
    python3 evidence/major_revision/plot_ward_revision.py

The first script verifies the INPUTS.json hashes, checks the twenty saved
pairings, and runs the prescribed exact-phase Ward benchmark. The benchmark
uses eight independent momentum samples per spectral pair and five widths
up to 2^21. All seeds, probes, regulators and uncertainties are saved.
The second script forms characteristic-function-weighted ensemble ratios.
The third regenerates the two Ward figures and accepts --output-dir.

For fast checks without the full benchmark, run verification/check_current.py
from the repository root. Requirements and the benchmark configuration are
recorded here. The data/ inputs are preserved production data. Conditional
field errors and weighted ensemble-ratio errors are distinct statistics.

Finite Fock readouts require a positive Hermitian projected mixed block.
The failed sample is retained without clipping. Raw kernel and Ward
calculations use the original neural ensemble.
