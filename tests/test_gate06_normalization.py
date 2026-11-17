"""Gate 6: kernel normalization and convention checks.

These pin the conventions every later gate depends on. Each test that asserts a
convention is paired, where it matters, with a control asserting that the
obvious wrong convention does NOT satisfy it.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

from nnft_virasoro import kernels as K
from nnft_virasoro import ordered_pairing as OP

WINDOWS = [(1e-2, 1e2), (1e-3, 1e3), (1e-4, 1e4)]


@pytest.mark.parametrize("lir,luv", WINDOWS)
def test_sharp_window_log_measure_equals_log_ratio(lir, luv):
    assert K.tapered_log_measure(lir, luv, 0.0) == pytest.approx(
        K.log_ratio(lir, luv), rel=0, abs=1e-12)


@pytest.mark.parametrize("lir,luv", WINDOWS)
@pytest.mark.parametrize("taper", [0.05, 0.15, 0.3])
def test_raised_cosine_taper_removes_half_of_each_rolloff(lir, luv, taper):
    """A two-sided raised-cosine window has L_w/L = 1 - taper exactly."""
    L = K.log_ratio(lir, luv)
    Lw = K.tapered_log_measure(lir, luv, taper)
    assert Lw / L == pytest.approx(1.0 - taper, abs=1e-6)


@pytest.mark.parametrize("lir,luv", WINDOWS)
@pytest.mark.parametrize("taper", [0.0, 0.15])
def test_sigma_phi_squared_is_four_L_w_and_matches_Z(lir, luv, taper):
    """sigma_phi^2 = 4 L_w, equivalently 2Z/pi with Z = 2 pi L_w."""
    Lw = K.tapered_log_measure(lir, luv, taper)
    s2 = K.sigma_phi_squared(lir, luv, log_measure=Lw)
    assert s2 == pytest.approx(4.0 * Lw, rel=1e-14)
    Z = 2.0 * np.pi * Lw
    assert 2.0 * Z / np.pi == pytest.approx(s2, rel=1e-14)


@pytest.mark.parametrize("lir,luv", WINDOWS)
def test_using_L_instead_of_L_w_would_be_wrong_when_tapered(lir, luv):
    """Control: the sharp-window measure must NOT serve for a tapered prior."""
    L = K.log_ratio(lir, luv)
    Lw = K.tapered_log_measure(lir, luv, 0.15)
    assert abs(4.0 * L - 4.0 * Lw) / (4.0 * Lw) > 0.15


@pytest.mark.parametrize("lir,luv", WINDOWS)
def test_normal_ordering_constant_is_half_sigma_phi_squared(lir, luv):
    assert K.scalar_kernel_zero(lir, luv) == pytest.approx(
        0.5 * K.sigma_phi_squared(lir, luv), rel=1e-14)


@pytest.mark.parametrize("lir,luv", WINDOWS)
def test_scalar_kernel_log_slope_is_minus_two(lir, luv):
    """dK/dlog r = -2 inside the scaling window.

    NOTE the sign trap: (K(r1) - K(r2))/log(r2/r1) is MINUS the derivative.
    """
    r1 = 10.0 / luv * np.sqrt(10.0)
    r2 = r1 * 10.0
    assert r2 < 0.1 / lir, "probe radii must sit inside the scaling window"
    k1 = float(np.asarray(K.scalar_kernel(r1, lir, luv)).reshape(()))
    k2 = float(np.asarray(K.scalar_kernel(r2, lir, luv)).reshape(()))
    slope = (k2 - k1) / np.log(r2 / r1)
    assert slope == pytest.approx(-2.0, abs=5e-3)


@pytest.mark.parametrize("theta", [0.0, 0.4, 0.7, 1.3, 2.3])
def test_current_channel_carries_spin_two(theta):
    """K_JJ(theta)/K_JJ(0) = exp(-2 i theta): spin two, not spin zero."""
    lir, luv = 1e-2, 1e2
    r = 0.3
    xi = r * np.exp(1j * theta)
    got = (complex(K.current_kernel_analytic(xi, 0j, lir, luv))
           / complex(K.current_kernel_analytic(r + 0j, 0j, lir, luv)))
    assert got == pytest.approx(np.exp(-2j * theta), abs=1e-9)
    if abs(theta % np.pi) > 1e-9:            # a spin-zero channel is excluded
        assert abs(got - 1.0) > 1e-3


@pytest.mark.parametrize("r", [0.05, 0.1, 0.3, 0.7])
def test_radial_moment_is_the_J2_moment_not_J0(r):
    """The closed form equals the J_2 spectral quadrature, and not the J_0 one."""
    lir, luv = 1e-2, 1e2
    pts, wts = OP._gauss_legendre_panels(lir, luv, np.pi / (8 * r), nodes=8)
    q2 = float(np.sum(wts * pts * jv(2, pts * r)) / 2.0)
    q0 = float(np.sum(wts * pts * jv(0, pts * r)) / 2.0)
    exact = float(np.real(K.current_kernel_analytic(r + 0j, 0j, lir, luv)))
    assert q2 == pytest.approx(exact, rel=1e-8, abs=1e-10)
    assert abs(q0 - exact) / abs(exact) > 0.1


@pytest.mark.parametrize("r", [0.5, 1.0, 2.0])
def test_uv_tail_envelope_grows_like_sqrt_lambda(r):
    """The pointwise kernel does not converge: its envelope is ~ sqrt(Lambda).

    Amplitude of the stationary-phase boundary term of
    K_JJ^Lambda(r) = (1/2) int^Lambda dk k J_2(kr).
    """
    lir, luv = 1e-2, 1e5
    half = 20.0 * np.pi / (luv * r)
    rr = np.linspace(r - half, r + half, 40001)
    env = float(np.max(np.abs(np.real(
        K.current_kernel_analytic(rr + 0j, 0j, lir, luv)))))
    predicted = np.sqrt(luv) * np.sqrt(2.0 / (np.pi * r)) / (2.0 * r)
    assert env == pytest.approx(predicted, rel=0.05)


def test_pointwise_ideal_limit_does_not_hold():
    """Control for the above: widening the window does not approach 1/(z-w)^2."""
    errs = [abs(complex(K.current_kernel_analytic(1 + 0j, 0j, lir, luv)) - 1.0)
            for lir, luv in WINDOWS]
    assert min(errs) > 1.0            # never close
    assert errs[-1] > errs[0]         # and it gets worse, not better


def test_ordered_pairing_of_ideal_kernel_is_canonical():
    """The ideal limit IS meaningful after ordered contour pairing."""
    modes = OP.mode_window(4)
    r0 = 1.0
    _, r_in, r_out = OP.radii_from_window(r0 / 2.0, r0 * 2.0, 2.0, r0=r0)
    res = OP.pairing_ideal_quadrature(modes, r_in, r_out, q=512, r0=r0)
    can = OP.canonical_pairing(modes)
    assert np.max(np.abs(res.omega - can)) < 1e-8
    # control: a single ordering alone is NOT canonical
    assert np.max(np.abs(res.i_greater - can)) > 0.5
