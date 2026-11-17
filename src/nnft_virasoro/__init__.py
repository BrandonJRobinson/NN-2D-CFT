"""Frozen evidence suite for *Virasoro Symmetry in Neural Network Field Theories*.

All conventions are taken from ``Draft-JHEP-Active-v6.tex``; the mapping from
code symbol to manuscript equation is tabulated in
``results/jhep_frozen/REPOSITORY_MAP.md`` section 3.

Estimator classes (never mixed inside one uncertainty calculation):

``analytic_closed_kernel``
    Exact regulated Bessel kernel, eq. (A.1)/(A.2). Deterministic, no seed.
``deterministic_finite_feature``
    Fixed momentum sample, phase-averaged kernel, eq. (3.6). Seeded but
    exact given the sample.
``sampled_finite_network``
    Monte Carlo over independent network realizations. Carries sampling
    uncertainty.
``exact_symbolic``
    Closed-form algebra on the ordered-word / filtered-Fock representation.
"""
from __future__ import annotations

__version__ = "1.0.0"

ESTIMATOR_CLASSES = (
    "analytic_closed_kernel",
    "deterministic_finite_feature",
    "sampled_finite_network",
    "exact_symbolic",
)
