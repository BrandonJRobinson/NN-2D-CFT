"""Gate 1: intrinsic physical admissibility of the pre-Darboux pairing."""
from __future__ import annotations

import numpy as np
import pytest

from nnft_virasoro import intrinsic_admissibility as IA

L = 4
D = np.diag(np.arange(1, L + 1).astype(float))


def canonical():
    om = np.zeros((2 * L, 2 * L), dtype=complex)
    om[:L, L:] = D
    om[L:, :L] = -D
    return om


def star_completed(G, A=None, B=None):
    """Build Omega from a cross block, completing B by the star condition."""
    om = np.zeros((2 * L, 2 * L), dtype=complex)
    om[:L, :L] = np.zeros((L, L)) if A is None else A
    om[:L, L:] = G
    om[L:, :L] = -G.T
    om[L:, L:] = (np.conj(om[:L, :L]) * -1) if B is None else B
    return om


# --------------------------------------------------------------- test matrix
def test_canonical_physical_pairing():
    r = IA.intrinsic_diagnostics(canonical())
    assert r.delta_star < 1e-14
    assert r.delta_A < 1e-14 and r.delta_B < 1e-14
    assert r.delta_H < 1e-14
    assert r.delta_iso < 1e-12
    assert r.lambda_min_normalized == pytest.approx(1.0, abs=1e-12)
    assert r.physically_admissible
    assert r.principal_symplectic_defect < 1e-12
    assert r.principal_involution_defect < 1e-12


def test_star_breaking_pairing():
    """A non-zero positive-frequency block breaks the involution condition."""
    om = canonical()
    om[0, 1] += 0.4
    om[1, 0] -= 0.4          # keep antisymmetry, break A = 0
    r = IA.intrinsic_diagnostics(om)
    assert r.delta_A > 1e-3
    assert r.delta_star > 1e-3
    assert not r.physically_admissible


def test_non_isotropic_positive_frequency_block():
    """G Hermitian positive but not proportional to D: admissible, delta_iso>0."""
    G = D @ np.diag([1.0, 1.6, 0.7, 1.2])
    G = 0.5 * (G + G.conj().T)
    r = IA.intrinsic_diagnostics(star_completed(G))
    assert r.delta_H < 1e-12
    assert r.lambda_min_normalized > 0
    assert r.delta_iso > 1e-2
    assert r.physically_admissible          # isotropy is not required
    assert r.principal_symplectic_defect < 1e-10


def test_hermiticity_breaking_cross_block():
    """delta_H must expose a non-Hermitian G that a symmetrised eigenvalue hides."""
    G = D.astype(complex).copy()
    G[0, 1] += 0.5j
    G[1, 0] += 0.5j                          # anti-Hermitian part
    r = IA.intrinsic_diagnostics(star_completed(G))
    assert r.delta_H > 1e-3
    assert not r.physically_admissible
    # the symmetrised eigenvalue alone would have looked fine
    assert r.lambda_min_normalized > 0


def test_indefinite_cross_block():
    G = D @ np.diag([1.0, -1.0, 1.0, 1.0])
    G = 0.5 * (G + G.conj().T)
    r = IA.intrinsic_diagnostics(star_completed(G))
    assert r.delta_H < 1e-12
    assert r.lambda_min_normalized < 0
    assert not r.physically_admissible
    assert not r.principal_available         # G^{-1/2} does not exist


def test_positive_physical_cross_block():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L))
    G = X @ X.conj().T + L * np.eye(L)       # Hermitian positive definite
    r = IA.intrinsic_diagnostics(star_completed(G))
    assert r.delta_H < 1e-12
    assert r.lambda_min_normalized > 0
    assert r.physically_admissible
    assert r.principal_symplectic_defect < 1e-10
    assert r.principal_involution_defect < 1e-10


# --------------------------------------------------------------- iff theorem
@pytest.mark.parametrize("seed", range(12))
def test_physical_darboux_iff_theorem(seed):
    """admissible  <=>  the principal canonicalizer exists and works.

    Forward: when the criterion holds, S_0 must realise Omega_can while
    respecting the involution.  Converse: when it fails, no S_0 may do both.
    """
    rng = np.random.default_rng(seed)
    kind = seed % 4
    if kind == 0:
        X = rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L))
        om = star_completed(X @ X.conj().T + L * np.eye(L))
    elif kind == 1:
        G = D.astype(complex).copy()
        G[0, 2] += 0.6j
        G[2, 0] += 0.6j
        om = star_completed(G)
    elif kind == 2:
        s = np.ones(L)
        s[rng.integers(0, L)] = -1.0
        om = star_completed(D @ np.diag(s))
    else:
        om = canonical()
        om[0, 1] += 0.5
        om[1, 0] -= 0.5

    ok, cond = IA.physical_darboux_admissible(om)
    r = IA.intrinsic_diagnostics(om)
    assert ok == r.physically_admissible

    if ok:
        S0, info = IA.principal_canonicalizer(om)
        assert info["symplectic_defect"] < 1e-9
        assert info["involution_defect"] < 1e-9
    else:
        # either the witness cannot be built, or it fails one of the two checks
        try:
            _, info = IA.principal_canonicalizer(om)
            assert (info["symplectic_defect"] > 1e-9
                    or info["involution_defect"] > 1e-9), (cond, info)
        except (np.linalg.LinAlgError, ValueError):
            pass


def test_reorder_matches_frozen_mode_convention():
    """The Gate-1 ordering is a permutation of the frozen ascending ordering."""
    from nnft_virasoro import ordered_pairing as OP

    modes = OP.mode_window(L)                      # (-L..-1, 1..L)
    om_asc = OP.canonical_pairing(modes)
    om_phys = IA.reorder_to_physical(om_asc, modes, L)
    assert np.allclose(om_phys, canonical())
    r = IA.intrinsic_diagnostics(om_phys)
    assert r.physically_admissible
