"""Gate 2: pre-Darboux Virasoro stability in a fixed PBW basis."""
from __future__ import annotations

import numpy as np
import pytest

from nnft_virasoro import ordered_pairing as OP
from nnft_virasoro import pbw_residual as PB

L = 4
MODES = OP.mode_window(L)
CAN = OP.canonical_pairing(MODES)
PAIRS = [(1, -1), (2, -2), (2, -1), (3, -2), (-2, 1), (2, 3)]


def test_pbw_basis_is_fixed_and_independent_of_omega():
    """The basis must depend only on the mode window (mission requirement)."""
    b1 = PB.pbw_basis(MODES)
    b2 = PB.pbw_basis(list(MODES)[::-1])       # different input order
    assert b1 == b2
    # includes the zero mode, excluded from the measured pairing window
    idx = {int(k[1]) for w in b1 for k in w}
    assert 0 in idx
    assert idx == set(range(-L, L + 1))
    assert len(b1) == 1 + (2 * L + 1) * (2 * L + 2) // 2
    assert b1[0] == ()


def test_residual_is_a_polynomial_of_degree_at_most_two():
    """Every word in the residual must fit the degree-<=2 basis."""
    for om in (CAN, CAN + 0.05 * np.triu(np.ones_like(CAN), 1)
               - 0.05 * np.triu(np.ones_like(CAN), 1).T):
        d = PB.raw_pbw_residual(om, MODES, L, PAIRS)
        assert d["n_basis"] == len(PB.pbw_basis(MODES))
    basis = PB.pbw_basis(MODES)
    assert all(len(w) <= 2 for w in basis)


def test_canonical_pairing_leaves_only_the_edge_remainder():
    """R_raw does NOT vanish at Omega_can; the excess does."""
    d = PB.raw_pbw_residual(CAN, MODES, L, PAIRS)
    assert d["combined_pbw_norm_raw"] > 1.0          # truncation edge floor
    assert d["combined_pbw_norm_excess"] < 1e-12     # excess vanishes exactly
    assert d["edge_baseline_combined_norm"] == pytest.approx(
        d["combined_pbw_norm_raw"])


def test_scalar_and_quadratic_parts_are_stored_separately():
    d = PB.raw_pbw_residual(CAN, MODES, L, PAIRS)
    for k in ("scalar_norm_raw", "quadratic_norm_raw", "combined_pbw_norm_raw",
              "scalar_norm_excess", "quadratic_norm_excess",
              "combined_pbw_norm_excess"):
        assert k in d
    # combined is the Pythagorean sum of the two parts
    assert d["combined_pbw_norm_raw"] == pytest.approx(
        np.hypot(d["scalar_norm_raw"], d["quadratic_norm_raw"]), rel=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_residual_excess_is_O_eps_plus_eps_squared(seed):
    """The stability theorem: R/(eps + eps^2) stays bounded as eps -> 0."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(CAN.shape)
    A = X - X.T
    A = A / np.linalg.norm(A)
    ratios = []
    for e in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        om = CAN + e * A
        eps = PB.pairing_error_frobenius(om, MODES)
        r = PB.raw_pbw_residual(om, MODES, L, PAIRS)["combined_pbw_norm_excess"]
        ratios.append(r / (eps + eps**2))
    ratios = np.array(ratios)
    assert np.all(np.isfinite(ratios))
    # bounded, and stable across five decades (no divergence as eps -> 0)
    assert ratios.max() / ratios.min() < 2.0, ratios
    assert ratios.max() < 50.0, ratios


def test_excess_is_linear_in_eps_at_small_eps():
    """At small eps the O(eps) term dominates, so R_excess ~ eps."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal(CAN.shape)
    A = (X - X.T) / np.linalg.norm(X - X.T)
    r1 = PB.raw_pbw_residual(CAN + 1e-5 * A, MODES, L,
                             PAIRS)["combined_pbw_norm_excess"]
    r2 = PB.raw_pbw_residual(CAN + 1e-4 * A, MODES, L,
                             PAIRS)["combined_pbw_norm_excess"]
    assert r2 / r1 == pytest.approx(10.0, rel=0.02)


def test_unknown_word_is_rejected_loudly():
    from nnft_virasoro.superalgebra import boson

    basis = PB.pbw_basis(MODES)
    with pytest.raises(KeyError):
        PB.pbw_vector({(boson(99), boson(1)): 1.0 + 0j}, basis)


def test_stability_theorem_plateau_is_direction_stable():
    """The quoted 1.65-1.82 band: five decades of EPSILON, reference direction.

    This is a different sweep from the architecture grid's five decades of
    WIDTH, and the two must not be conflated: the epsilon sweep sits on the
    plateau, the width sweep is still climbing toward it.
    """
    import numpy as np
    eps_grid = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    spans = []
    for direction in range(12):
        rng = np.random.default_rng(direction)
        X = rng.standard_normal(CAN.shape)
        A = (X - X.T) / np.linalg.norm(X - X.T)
        r = []
        for e in eps_grid:
            om = CAN + e * A
            eps = PB.pairing_error_frobenius(om, MODES)
            r.append(PB.raw_pbw_residual(om, MODES, L,
                                         PAIRS)["combined_pbw_norm_excess"]
                     / (eps + eps**2))
        r = np.array(r)
        spans.append(r.max() / r.min())
        if direction == 0:                     # the quoted reference band
            assert r.min() == pytest.approx(1.6502, abs=5e-3)
            assert r.max() == pytest.approx(1.8151, abs=5e-3)
    # bounded in every direction, with the same span
    assert max(spans) < 1.2
