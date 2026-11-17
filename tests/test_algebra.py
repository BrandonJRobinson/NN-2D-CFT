"""Filtered Fock theorem, charged weights, edge remainder, Darboux, NS."""
from __future__ import annotations

import numpy as np
import pytest

from nnft_virasoro import algebra as AL
from nnft_virasoro import darboux as DB
from nnft_virasoro import ns_fermion as NS
from nnft_virasoro import ordered_pairing as OP
from nnft_virasoro.superalgebra import Pairings

CAN = Pairings.canonical()


@pytest.mark.parametrize("lam,K", [(6, 1), (8, 2), (10, 3)])
@pytest.mark.parametrize("charge", [0.0, 0.7])
def test_filtered_fock_exact_virasoro_identity(lam, K, charge):
    """Theorem 4.6: exact closure when K + m_- + n_- <= Lambda."""
    fock = AL.FockSpace(level=K, mode_cutoff=lam, charge=charge)
    checked = 0
    for m in range(-3, 4):
        for n in range(-3, 4):
            if m == 0 or n == 0 or m == n:
                continue
            if K + AL.m_minus(m) + AL.m_minus(n) > lam:
                continue
            el = AL.virasoro_residual(m, n, lam, CAN)
            assert fock.frobenius(el) < 1e-10, (m, n)
            checked += 1
    assert checked > 0


def test_identity_fails_just_outside_the_guaranteed_region():
    lam, K = 4, 4
    fock = AL.FockSpace(level=K, mode_cutoff=lam, charge=0.0)
    violations = 0
    for m in range(-3, 4):
        for n in range(-3, 4):
            if m == 0 or n == 0 or m == n:
                continue
            if K + AL.m_minus(m) + AL.m_minus(n) <= lam:
                continue
            if fock.frobenius(AL.virasoro_residual(m, n, lam, CAN)) > 1e-10:
                violations += 1
    assert violations > 0, "the filtered bound must be sharp, not vacuous"


@pytest.mark.parametrize("a", [0.0, 0.5, 1.0, 1.7])
def test_charged_sector_L0_weight(a):
    """Eq. (4.11): L_0|a> = (a^2/2)|a>."""
    fock = AL.FockSpace(level=0, mode_cutoff=8, charge=a)
    v = fock.apply_element(AL.sugawara_L(0, 8, CAN), {fock.basis[0]: 1.0 + 0j})
    assert complex(v.get(fock.basis[0], 0j)).real == pytest.approx(a * a / 2,
                                                                  abs=1e-12)


@pytest.mark.parametrize("m", [2, 3, 4, 8])
def test_canonical_central_scalar_is_exact(m):
    """scalar([L_m,L_-m] - 2mL_0) = m(m^2-1)/12 for the canonical algebra."""
    lam = m + 4
    got = complex(AL.central_scalar(m, lam, CAN)).real
    assert got == pytest.approx(m * (m * m - 1) / 12.0, abs=1e-12)


def test_raw_residual_vanishes_for_the_canonical_pairing():
    modes = OP.mode_window(8)
    rep = AL.raw_virasoro_residual(OP.canonical_pairing(modes), modes, 8,
                                   level=2, m_alg=3)
    assert rep.value < 1e-12


def test_raw_residual_is_positive_for_a_perturbed_pairing():
    modes = OP.mode_window(8)
    om = OP.canonical_pairing(modes).copy()
    i, j = list(modes).index(-1), list(modes).index(1)
    om[i, j] += 0.3
    om[j, i] -= 0.3
    rep = AL.raw_virasoro_residual(om, modes, 8, level=2, m_alg=3)
    assert rep.value > 1e-6


def test_edge_remainder_is_machine_readable():
    rec = AL.edge_remainder_terms(2, -2, 4)
    assert rec["n_terms"] > 0
    assert rec["coefficient_l2"] > 0
    for t in rec["terms"]:
        assert set(t) >= {"normal_ordered_monomial", "coefficient",
                          "oscillator_indices", "edge_support_reason",
                          "coefficient_abs"}


# --------------------------------------------------------------------------
def test_darboux_canonicalizes_and_is_star_compatible_for_the_canonical_form():
    modes = OP.mode_window(4)
    rep = DB.admissibility_report(OP.canonical_pairing(modes), modes)
    assert rep.symplectic_defect < 1e-10
    assert rep.star_compatibility_defect < 1e-10
    assert rep.polarization_min_eigenvalue > 0
    assert rep.vacuum_normalizable


def test_darboux_closes_a_non_symplectic_control_but_may_break_admissibility():
    """Proposition 3.5: closure is universal; admissibility is not."""
    rng = np.random.default_rng(2)
    modes = OP.mode_window(4)
    T = np.linalg.qr(rng.standard_normal((modes.size, modes.size)))[0] @ np.diag(
        np.geomspace(1.0, 6.0, modes.size))
    om = T @ OP.canonical_pairing(modes) @ T.T
    rep = DB.admissibility_report(om, modes)
    assert rep.symplectic_defect < 1e-8, "unrestricted Darboux must still close"


def test_star_compatible_darboux_condition_definition():
    """||conj(S) C - C S||_F / ||S||_F, C_{mn} = delta_{m,-n}."""
    modes = OP.mode_window(3)
    C = DB.involution_matrix(modes)
    assert np.allclose(C, C.T)
    assert np.allclose(C @ C, np.eye(modes.size))
    S = np.eye(modes.size, dtype=complex)
    assert np.linalg.norm(np.conj(S) @ C - C @ S) < 1e-14
    S2 = S * 1j
    assert np.linalg.norm(np.conj(S2) @ C - C @ S2) > 1e-6


# --------------------------------------------------------------------------
def test_ideal_fermionic_ordered_pairing_eq_7_5():
    tt = NS.fermion_mode_window(7)
    res = NS.fermionic_pairing_ideal_quadrature(tt, 1 / np.sqrt(2), np.sqrt(2),
                                                q=128, r0=1.0)
    ig_t = np.where((tt[:, None] + tt[None, :] == 0) & (tt[:, None] > 0), 1.0, 0.0)
    il_t = np.where((tt[:, None] + tt[None, :] == 0) & (tt[:, None] < 0), -1.0, 0.0)
    assert np.abs(res.i_greater - ig_t).max() < 1e-12
    assert np.abs(res.i_less - il_t).max() < 1e-12
    assert np.abs(res.omega - NS.canonical_car(tt)).max() < 1e-12


def test_car_pairing_is_symmetric():
    tt = NS.fermion_mode_window(7)
    res = NS.fermionic_pairing_ideal_quadrature(tt, 1 / np.sqrt(2), np.sqrt(2),
                                                q=128, r0=1.0)
    met = NS.car_metrics(res.omega, tt)
    assert met["car_symmetry_defect"] < 1e-12


def test_ns_super_virasoro_canonical_c_is_three_halves():
    lam, tmax = 8, 9
    mm = np.arange(-lam, lam + 1)
    om = np.array([[float(i) if i + j == 0 else 0.0 for j in mm] for i in mm])
    ft = np.array([t for t in range(-tmax, tmax + 1) if t % 2])
    et = np.array([[1.0 if s + t == 0 else 0.0 for t in ft] for s in ft])
    r = AL.super_virasoro_residuals(om, mm, et, ft, lam, tmax, level=1, r_alg=3)
    assert r["LG_residual"] < 1e-10
    assert r["GG_residual"] < 1e-10
    assert r["central_charge_fit"] == pytest.approx(1.5, abs=1e-10)


def test_isotropic_symmetric_canonicalization():
    tt = NS.fermion_mode_window(5)
    rng = np.random.default_rng(9)
    T = np.eye(tt.size) + 0.1 * rng.standard_normal((tt.size, tt.size))
    eta = T @ NS.canonical_car(tt) @ T.T
    rep = DB.car_canonicalization_report(eta, tt)
    assert rep["post_canonicalization_car_defect"] < 1e-8
