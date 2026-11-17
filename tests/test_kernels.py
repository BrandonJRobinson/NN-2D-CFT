"""Kernel conventions and the finite-cutoff non-holomorphy formulas.

The tolerances below are not slack: they encode the *measured* convergence rates
of the sharp annular prior of eq. (2.2).  Writing them down is part of the WP-A
and WP-F findings, so they are asserted rather than hidden:

* ``K_Lambda``  converges cleanly (the integrand is ``J_0(kr)/k``);
* ``k'``        residual is ``J_0(b)``            -- decays like ``b^{-1/2}``;
* ``k''``       residual is ``2J_0(b) + b J_1(b)`` -- *grows* like ``b^{+1/2}``,
  so the current kernel's relative error decays only like ``b^{-1/2}``;
* ``2k'' + u k'''`` equals ``[b^2 J_2(b) - a^2 J_2(a)]/(4u^2)`` and grows like
  ``b^{3/2}``, which is exactly why the Cauchy--Green bound of prop. 3.6 is
  large for eq. (2.2).

Here ``b = Lambda_UV r`` and ``a = Lambda_IR r``.
"""
from __future__ import annotations

import numpy as np
import pytest

from nnft_virasoro import kernels as K


def _f(x):
    return float(np.asarray(x, dtype=float).reshape(()))


def test_sigma_calibration_gives_minus_two_log_r():
    """kappa = 1 calibration of eq. (2.6): K_Lambda(r) ~ -2 log r + const."""
    lir, luv = 1e-6, 1e6
    r1, r2 = 0.5, 2.0
    k1 = _f(K.scalar_kernel(r1, lir, luv))
    k2 = _f(K.scalar_kernel(r2, lir, luv))
    assert (k1 - k2) == pytest.approx(-2.0 * np.log(r1 / r2), rel=1e-4)


def test_sigma_phi_squared_matches_the_log_measure():
    assert K.sigma_phi_squared(1e-2, 1e2) == pytest.approx(4 * np.log(1e4))
    lw = K.tapered_log_measure(1e-2, 1e2, 0.15)
    assert lw < K.log_ratio(1e-2, 1e2)
    assert K.sigma_phi_squared(1e-2, 1e2, lw) == pytest.approx(4 * lw)
    assert K.tapered_log_measure(1e-2, 1e2, 0.0) == K.log_ratio(1e-2, 1e2)


def test_kd_conformal_limits_at_the_measured_rates():
    """k' converges like b^{-1/2}; k'' does NOT converge at all.

    k'' carries the endpoint term b J_1(b), whose amplitude grows like b^{1/2},
    so the sharp annulus of eq. (2.2) has no conformal limit for the current
    kernel. Both statements are asserted.
    """
    u = 0.7
    for luv in (1e6, 1e12):
        b = luv * np.sqrt(u)
        assert _f(K.k_d1(u, 1e-14, luv)) == pytest.approx(
            -1.0 / u, rel=20.0 / np.sqrt(b))
    err6 = abs(_f(K.k_d2(u, 1e-14, 1e6)) - 1.0 / u**2)
    err12 = abs(_f(K.k_d2(u, 1e-14, 1e12)) - 1.0 / u**2)
    assert err12 > err6 > 1.0, "k'' must diverge with the sharp cutoff"


def test_closed_form_derivatives_match_numerical_differentiation():
    """Acid test of the analytic Bessel derivative formulas."""
    lir, luv = 1e-2, 1e2
    for u0 in (0.09, 0.25, 1.0, 4.0):
        h = u0 * 1e-4
        f = lambda uu: _f(K.scalar_kernel(np.sqrt(uu), lir, luv))
        num = (f(u0 + h) - 2 * f(u0) + f(u0 - h)) / h**2
        assert _f(K.k_d2(u0, lir, luv)) == pytest.approx(num, rel=1e-5)


def test_current_kernel_has_no_sharp_cutoff_conformal_limit():
    """Widening the sharp annulus makes the current kernel *worse*, not better."""
    z, w = 1.3 + 0.4j, 0.2 - 0.5j
    ideal = complex(K.current_kernel_ideal(z, w))
    e1 = abs(complex(K.current_kernel_analytic(z, w, 1e-14, 1e4)) - ideal)
    e2 = abs(complex(K.current_kernel_analytic(z, w, 1e-14, 1e8)) - ideal)
    assert e2 > e1 > abs(ideal)


def test_sharp_cutoff_current_kernel_is_wrong_inside_the_nominal_window():
    """The WP-A / WP-F finding, asserted: eq. (2.2) does not deliver 1/(z-w)^2."""
    lir, luv = 1e-2, 1e2
    r_min, r_max = K.physical_window(lir, luv, margin_uv=30.0, margin_ir=30.0)
    r = np.exp(np.linspace(np.log(r_min), np.log(r_max), 64))
    kjj = np.real(K.current_kernel_analytic(r.astype(complex), 0j, lir, luv))
    basis = r**-2.0
    amp = float(np.sum(kjj * basis) / np.sum(basis * basis))
    shape = float(np.linalg.norm(kjj - amp * basis) / np.linalg.norm(amp * basis))
    assert shape > 0.5, "sharp-annulus shape error must be O(1), not small"


def test_fermion_kernel_reduces_to_cauchy_and_is_antisymmetric():
    """One derivative only, so this converges cleanly: the residual is J_0(b)."""
    lir, luv = 1e-12, 1e12
    z, w = 1.1 + 0.2j, -0.3 + 0.7j
    got = complex(K.fermion_kernel_analytic(z, w, lir, luv))
    assert got == pytest.approx(1.0 / (z - w), rel=1e-5)
    rev = complex(K.fermion_kernel_analytic(w, z, lir, luv))
    assert rev == pytest.approx(-got, rel=1e-12)


def test_dbar_formula_matches_finite_difference():
    """Eq. (3.15) verified against a centred complex finite difference."""
    lir, luv = 1e-2, 1e2
    z, w = 1.0 + 0.3j, 0.2 - 0.1j
    h = 1e-6
    f = lambda zz: complex(K.current_kernel_analytic(zz, w, lir, luv))
    dx = (f(z + h) - f(z - h)) / (2 * h)
    dy = (f(z + 1j * h) - f(z - 1j * h)) / (2 * h)
    num = 0.5 * (dx + 1j * dy)
    ana = complex(K.dbar_current_kernel_analytic(z, w, lir, luv))
    assert ana == pytest.approx(num, rel=2e-5, abs=1e-8)


def test_dbar_is_a_sharp_cutoff_artifact():
    """A 15% raised-cosine taper removes seven orders of magnitude of dbar."""
    from nnft_virasoro import nonholomorphy as NH

    z, w = 1.0 + 0.3j, 0.2 - 0.1j
    sharp = abs(complex(K.dbar_current_kernel_analytic(z, w, 1e-2, 1e4)))
    sharp_alt = abs(complex(NH.dbar_current_kernel_tapered(z, w, 1e-2, 1e4, 0.0)))
    tapered = abs(complex(NH.dbar_current_kernel_tapered(z, w, 1e-2, 1e4, 0.15)))
    assert sharp_alt == pytest.approx(sharp, rel=1e-4)   # cross-validation
    assert sharp > 1e3
    assert tapered < 1e-1
    assert tapered < 1e-4 * sharp


def test_empirical_current_kernel_converges_to_analytic():
    lir, luv = 1e-2, 1e2
    z, w = 1.0 + 0.0j, 0.35 + 0.15j
    ana = complex(K.current_kernel_analytic(z, w, lir, luv))
    errs = {}
    for nn in (25_000, 400_000):
        d = []
        for s in range(8):
            r2 = np.random.default_rng(1000 + s)
            ka = np.exp(r2.uniform(np.log(lir), np.log(luv), size=nn))
            kk = ka * np.exp(1j * r2.uniform(0, 2 * np.pi, size=nn))
            d.append(abs(complex(K.empirical_current_kernel(z, w, kk, lir, luv))
                         - ana))
        errs[nn] = float(np.sqrt(np.mean(np.square(d))))
    # Monte Carlo rate: a 16x increase in N must cut the RMS error ~4x.
    ratio = errs[25_000] / errs[400_000]
    assert 2.0 < ratio < 8.0, errs


def test_empirical_scalar_kernel_converges_to_analytic():
    lir, luv = 1e-2, 1e2
    rng = np.random.default_rng(4242)
    n = 400_000
    kabs = np.exp(rng.uniform(np.log(lir), np.log(luv), size=n))
    kc = kabs * np.exp(1j * rng.uniform(0, 2 * np.pi, size=n))
    r = 0.8
    emp = _f(K.empirical_scalar_kernel(r + 0j, 0j, kc, lir, luv))
    ana = _f(K.scalar_kernel(r, lir, luv))
    assert emp == pytest.approx(ana, rel=0.05, abs=0.1)


def test_physical_window_is_strict():
    r_min, r_max = K.physical_window(1e-2, 1e2, margin_uv=30.0, margin_ir=30.0)
    assert 1e-2 < r_min < r_max < 1e2
    assert r_min == pytest.approx(0.3)
    assert r_max == pytest.approx(1.0 / 0.3)
    with pytest.raises(ValueError):
        K.physical_window(1e-2, 1e2, margin_uv=1.0, margin_ir=30.0)


def test_dphi_self_contraction_is_the_normal_ordering_subtraction():
    rng = np.random.default_rng(1)
    n = 20000
    kabs = np.exp(rng.uniform(np.log(1e-2), np.log(1e2), size=n))
    kc = kabs * np.exp(1j * rng.uniform(0, 2 * np.pi, size=n))
    direct = complex(K.empirical_dphi_self_contraction(kc, 1e-2, 1e2))
    at_zero = -complex(K.empirical_current_kernel(0j, 0j, kc, 1e-2, 1e2))
    assert direct == pytest.approx(at_zero, rel=1e-12)
