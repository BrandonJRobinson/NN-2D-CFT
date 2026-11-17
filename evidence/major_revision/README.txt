MAJOR REVISION EVIDENCE - 12 NOVEMBER 2026

From any working directory, run these scripts in this order:

    python3 analyze_revision.py
    python3 summarize_full_ward.py
    python3 plot_ward_revision.py

Use the script paths if running from elsewhere. The first script verifies
all INPUTS.json hashes, checks the twenty saved pairing matrices, and
runs the exact-phase Ward benchmark. It uses eight independent momentum
samples per spectral pair, with shared prefixes across five widths up to
2^21. No explicit network-phase Monte Carlo is needed. The second script
forms the full-ensemble ratios using the vertex characteristic functions
as weights. The third writes the two Ward figure PDFs under ../figures/.
No manuscript compilation is performed.

The requirements file records the package versions used for this analysis.
The benchmark configuration records all geometries, spectral pairs, widths
and seed-derivation rules. Per-probe and per-seed results are retained.
Eight independent momentum samples provide the uncertainty units; probes
and widths within each sample are correlated.

The files under data/ are unchanged September outputs and matrices, with
original relative locations and checksums in INPUTS.json. New analysis
outputs are separate. The old deterministic Ward campaign evaluates a
Gaussian covariance proxy. Its absolute errors and its paired ratios are
retained. The interval on the historical paired log ratios now uses the
Student t distribution with seven degrees of freedom.

The finite Fock projection changes the input pairing for an auxiliary
oscillator readout. Raw kernel and Ward calculations use the original
ensemble. A projected matrix is used only when its Hermitian mixed block
is positive. The one failure is recorded without eigenvalue clipping.

New exact-phase formulas are checked by direct phase quadrature before
the benchmark runs. The full ensemble Ward ratio is a ratio of weighted
block means. Mean conditional errors and errors of the full ensemble
ratio are distinct statistics, each reported in the saved summaries.
