r"""Regulated Log-Kernel covariances, current kernels, and non-holomorphy.

Manuscript conventions (``Draft-JHEP-Active-v6.tex``)
----------------------------------------------------
Field, eq. (2.1)::

    phi_N(x) = (sigma_phi/sqrt(N)) sum_j cos(k_j.x + gamma_j),  gamma_j ~ U[0,2pi)

Spectral prior, eq. (2.2)::

    p_Lambda(k) = 1_{Lambda_IR<=|k|<=Lambda_UV} / (Z_Lambda |k|^2),
    Z_Lambda = 2 pi log(Lambda_UV/Lambda_IR)

Exact regulated scalar kernel, eq. (A.1)::

    K_Lambda(r) = sigma_phi^2/(2 log(Lambda_UV/Lambda_IR)) int dk/k J_0(k r)

Calibration.  Eq. (2.9) fixes ``E[phi phi] = -log|z-w|^2`` in the conformal
window, i.e. ``kappa = 1`` in eq. (2.6).  Since
``int_{a}^{infty} J_0(t) dt/t = -log(a/2) - gamma + O(a^2)``, the coefficient of
``log r`` in ``K_Lambda`` is ``-sigma_phi^2/(2L)`` with ``L =
log(Lambda_UV/Lambda_IR)``; matching ``-2 log r`` gives

    sigma_phi^2 = 4 L .

Radial derivatives.  Writing ``u = (z-w)(zbar-wbar) = r^2`` and
``K_Lambda = k_Lambda(u)``, the identity
``d/ds [s^{-nu/2} J_nu(sqrt s)] = -(1/2) s^{-(nu+1)/2} J_{nu+1}(sqrt s)``
gives the *closed form*

    d^n k_Lambda/du^n = 2 (-1/2)^n u^{-n} M_n(Lambda_IR r, Lambda_UV r),
    M_n(a,b) = int_a^b t^{n-1} J_n(t) dt .

``M_1``, ``M_2``, ``M_3`` have elementary antiderivatives (see ``_M1/_M2/_M3``),
so ``k'``, ``k''``, ``k'''`` are analytic - no numerical differentiation is
used anywhere in the suite (mission WP-A2: "Prefer analytic formulas").

Current kernel, eqs. (2.10), (A.2), (3.14), (3.15)::

    J = i d_z phi
    K_JJ^Lambda(z,w) = -d_z d_w K_Lambda(z-w) = (zbar-wbar)^2 k''(u)
    d_zbar K_JJ^Lambda = (zbar-wbar) [2 k''(u) + u k'''(u)]

Fermionic (NS) seed, section 7:  ``K_psipsi = -d_z K_Lambda``, which in the
conformal window is ``1/(z-w)``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0, j1, jv

__all__ = [
    "log_ratio",
    "sigma_phi_squared",
    "scalar_kernel",
    "k_d1",
    "k_d2",
    "k_d3",
    "current_kernel_analytic",
    "current_kernel_ideal",
    "dbar_current_kernel_analytic",
    "fermion_kernel_analytic",
    "fermion_kernel_ideal",
    "empirical_scalar_kernel",
    "empirical_current_kernel",
    "empirical_fermion_kernel",
    "physical_window",
    "tapered_log_measure",
]

EULER_GAMMA = 0.5772156649015328606


def log_ratio(lambda_ir: float, lambda_uv: float) -> float:
    """``L = log(Lambda_UV / Lambda_IR)``; the prior normalization exponent."""
    if not (lambda_uv > lambda_ir > 0):
        raise ValueError(
            f"require 0 < Lambda_IR < Lambda_UV, got {lambda_ir!r}, {lambda_uv!r}"
        )
    return float(np.log(lambda_uv / lambda_ir))


def tapered_log_measure(lambda_ir: float, lambda_uv: float,
                        taper: float = 0.0) -> float:
    r"""``L_w = int_{log Lambda_IR}^{log Lambda_UV} w(t) dt``.

    ``L_w = L`` for the manuscript's sharp annulus (``taper = 0``).  For the
    labelled tapered diagnostic the prior is ``p ~ w(k)/|k|^2`` with
    ``Z = 2 pi L_w``, so the ``kappa = 1`` calibration of eq. (2.6) becomes
    ``sigma_phi^2 = 4 L_w`` -- the coefficient of ``log r`` is set by the
    *plateau* of the window, where ``w = 1``.  Using ``L`` instead of ``L_w``
    would leave a spurious ``L/L_w`` amplitude error in every tapered row.
    """
    L = log_ratio(lambda_ir, lambda_uv)
    if taper <= 0:
        return L
    from .architectures import _tukey_log_window

    t = np.linspace(np.log(lambda_ir), np.log(lambda_uv), 20001)
    return float(np.trapz(_tukey_log_window(t, t[0], t[-1], taper), t))


def sigma_phi_squared(lambda_ir: float, lambda_uv: float,
                      log_measure: float | None = None) -> float:
    """``sigma_phi^2 = 4 L_w`` -- the kappa = 1 calibration of eq. (2.6)."""
    L = log_ratio(lambda_ir, lambda_uv) if log_measure is None else float(log_measure)
    return 4.0 * L


# --------------------------------------------------------------------------
# M_n(a,b) = int_a^b t^{n-1} J_n(t) dt   (elementary for n = 1,2,3)
# --------------------------------------------------------------------------
def _M1(t: np.ndarray) -> np.ndarray:
    """Antiderivative of ``J_1``: ``-J_0``."""
    return -j0(t)


def _M2(t: np.ndarray) -> np.ndarray:
    """Antiderivative of ``t J_2``: ``-2 J_0 - t J_1``."""
    return -2.0 * j0(t) - t * j1(t)


def _M3(t: np.ndarray) -> np.ndarray:
    """Antiderivative of ``t^2 J_3``: ``-8 J_0 - 4 t J_1 - t^2 J_2``."""
    return -8.0 * j0(t) - 4.0 * t * j1(t) - t * t * jv(2, t)


def _M0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``int_a^b J_0(t) dt / t``.

    Split into three pieces:

    * ``[a, 1]``   : ``int (J_0-1)/t dt + log(1/a)``  -- regular at the lower end;
    * ``[1, B]``   : panelized Gauss-Legendre with quarter-period panels;
    * ``[B, b]``   : the leading large-argument asymptotic
      ``J_0 ~ sqrt(2/(pi t)) cos(t - pi/4)``, whose primitive is
      ``sqrt(2/pi) t^{-3/2} sin(t - pi/4) + O(t^{-5/2})``.

    A single adaptive ``quad`` across ten decades of an oscillatory integrand is
    badly inaccurate, which is why the split is used.
    """
    from scipy.integrate import quad

    from .ordered_pairing import _gauss_legendre_panels

    a_arr = np.atleast_1d(np.asarray(a, dtype=float))
    b_arr = np.atleast_1d(np.asarray(b, dtype=float))
    shape = np.broadcast(a_arr, b_arr).shape
    out = np.empty(shape, dtype=float)
    T0, B = 1.0, 400.0
    it = np.nditer([np.broadcast_to(a_arr, shape), np.broadcast_to(b_arr, shape),
                    out], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
    with it:
        for lo, hi, res in it:
            lo_f, hi_f = float(lo), float(hi)
            total = 0.0
            mid = lo_f
            if lo_f < T0:
                mid = min(hi_f, T0)
                reg, _ = quad(lambda t: (j0(t) - 1.0) / t, lo_f, mid,
                              limit=200, epsabs=1e-14, epsrel=1e-12)
                total += reg + np.log(mid / lo_f)
            if hi_f > mid:
                upper = min(hi_f, B)
                if upper > mid:
                    pts, wts = _gauss_legendre_panels(mid, upper, np.pi / 4.0,
                                                      nodes=8)
                    total += float(np.sum(wts * j0(pts) / pts))
                if hi_f > B:
                    c = np.sqrt(2.0 / np.pi)
                    total += float(
                        c * (hi_f**-1.5 * np.sin(hi_f - np.pi / 4)
                             - B**-1.5 * np.sin(B - np.pi / 4)))
            res[...] = total
    return out if out.shape else out.reshape(())


def scalar_kernel(r, lambda_ir: float, lambda_uv: float) -> np.ndarray:
    """Exact regulated scalar covariance ``K_Lambda(r)``, eq. (A.1).

    Normalized so that ``K_Lambda(r) ~ -2 log r + const`` inside the window.
    """
    r = np.asarray(r, dtype=float)
    scale = sigma_phi_squared(lambda_ir, lambda_uv) / (
        2.0 * log_ratio(lambda_ir, lambda_uv)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        val = scale * _M0(lambda_ir * r, lambda_uv * r)
    # r = 0 limit: int dk/k over the annulus = L.
    val = np.where(r > 0, val, scale * log_ratio(lambda_ir, lambda_uv))
    return val


def scalar_kernel_zero(lambda_ir: float, lambda_uv: float) -> float:
    """``K_Lambda(0) = sigma_phi^2/2 = 2 L`` -- the normal-ordering constant."""
    return 0.5 * sigma_phi_squared(lambda_ir, lambda_uv)


def _kd(n: int, u, lambda_ir: float, lambda_uv: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    r = np.sqrt(u)
    a = lambda_ir * r
    b = lambda_uv * r
    if n == 1:
        prim = _M1
    elif n == 2:
        prim = _M2
    elif n == 3:
        prim = _M3
    else:  # pragma: no cover
        raise ValueError("only n = 1,2,3 are implemented in closed form")
    m_n = prim(b) - prim(a)
    return 2.0 * (-0.5) ** n * u ** (-n) * m_n


def k_d1(u, lambda_ir: float, lambda_uv: float) -> np.ndarray:
    """``k_Lambda'(u) = -(J_0(a)-J_0(b))/u``; tends to ``-1/u``."""
    return _kd(1, u, lambda_ir, lambda_uv)


def k_d2(u, lambda_ir: float, lambda_uv: float) -> np.ndarray:
    """``k_Lambda''(u) = M_2(a,b)/(2 u^2)``; tends to ``1/u^2``."""
    return _kd(2, u, lambda_ir, lambda_uv)


def k_d3(u, lambda_ir: float, lambda_uv: float) -> np.ndarray:
    """``k_Lambda'''(u) = -M_3(a,b)/(4 u^3)``; tends to ``-2/u^3``."""
    return _kd(3, u, lambda_ir, lambda_uv)


# --------------------------------------------------------------------------
# Analytic (closed-kernel) current and fermion kernels
# --------------------------------------------------------------------------
def current_kernel_analytic(z, w, lambda_ir: float, lambda_uv: float) -> np.ndarray:
    """``K_JJ^Lambda(z,w) = (zbar-wbar)^2 k''(u)``, eq. (3.14)."""
    xi = np.asarray(z) - np.asarray(w)
    u = np.abs(xi) ** 2
    return np.conj(xi) ** 2 * k_d2(u, lambda_ir, lambda_uv)


def current_kernel_ideal(z, w) -> np.ndarray:
    """``K_JJ(z,w) = 1/(z-w)^2``, eq. (2.14)."""
    xi = np.asarray(z) - np.asarray(w)
    return 1.0 / xi**2


def dbar_current_kernel_analytic(z, w, lambda_ir: float, lambda_uv: float) -> np.ndarray:
    """``d_zbar K_JJ^Lambda = (zbar-wbar)[2k''(u) + u k'''(u)]``, eq. (3.15)."""
    xi = np.asarray(z) - np.asarray(w)
    u = np.abs(xi) ** 2
    return np.conj(xi) * (
        2.0 * k_d2(u, lambda_ir, lambda_uv) + u * k_d3(u, lambda_ir, lambda_uv)
    )


def fermion_kernel_analytic(z, w, lambda_ir: float, lambda_uv: float) -> np.ndarray:
    """``K_psipsi^Lambda = -d_z K_Lambda = (J_0(a)-J_0(b))/(z-w)``, section 7."""
    xi = np.asarray(z) - np.asarray(w)
    u = np.abs(xi) ** 2
    return -np.conj(xi) * k_d1(u, lambda_ir, lambda_uv)


def fermion_kernel_ideal(z, w) -> np.ndarray:
    """``K_psipsi = 1/(z-w)``, eq. (7.1)."""
    return 1.0 / (np.asarray(z) - np.asarray(w))


# --------------------------------------------------------------------------
# Empirical (finite-feature, phase-averaged) kernels -- eq. (3.6)
# --------------------------------------------------------------------------
def empirical_scalar_kernel(z, w, kc: np.ndarray, lambda_ir: float, lambda_uv: float,
                            *, chunk: int = 4096,
                            log_measure: float | None = None) -> np.ndarray:
    r"""``K^{(N)}(z,w) = (2L_w/N) sum_j cos(Re(conj(k_j)(z-w)))``.

    ``kc`` is the complex momentum array ``k_1 + i k_2``; ``k.x = Re(conj(k) z)``.
    Streamed over features in ``chunk``-sized blocks (mission engineering rule:
    never materialize the full feature matrix).
    """
    xi = np.asarray(z) - np.asarray(w)
    n = kc.size
    acc = np.zeros(np.shape(xi), dtype=float)
    for start in range(0, n, chunk):
        block = kc[start : start + chunk]
        ph = np.real(np.conj(block) * xi[..., None])
        acc += np.cos(ph).sum(axis=-1)
    L = log_ratio(lambda_ir, lambda_uv) if log_measure is None else log_measure
    return (2.0 * L / n) * acc


def empirical_current_kernel(z, w, kc: np.ndarray, lambda_ir: float, lambda_uv: float,
                             *, chunk: int = 4096,
                             prefactor: np.ndarray | None = None,
                             log_measure: float | None = None) -> np.ndarray:
    r"""Phase-averaged empirical current kernel, eq. (3.6).

    ``K_JJ^{(N,Lambda)}(z,w) = -(L/(2N)) sum_j P_j cos(Re(conj(k_j)(z-w)))``
    with ``P_j = conj(k_j)^2`` for the selected architecture.  ``prefactor``
    overrides ``P_j`` for angular-scramble controls (see
    :mod:`nnft_virasoro.architectures`).
    """
    xi = np.asarray(z) - np.asarray(w)
    n = kc.size
    pref = np.conj(kc) ** 2 if prefactor is None else np.asarray(prefactor)
    acc = np.zeros(np.shape(xi), dtype=complex)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        ph = np.real(np.conj(kc[sl]) * xi[..., None])
        acc += (np.cos(ph) * pref[sl]).sum(axis=-1)
    L = log_ratio(lambda_ir, lambda_uv) if log_measure is None else log_measure
    return -(L / (2.0 * n)) * acc


def empirical_fermion_kernel(z, w, kc: np.ndarray, lambda_ir: float, lambda_uv: float,
                             *, chunk: int = 4096,
                             log_measure: float | None = None) -> np.ndarray:
    r"""``K_psipsi^{(N,Lambda)} = -d_z K^{(N)} = (L/N) sum_j conj(k_j) sin(Re(conj(k_j)(z-w)))``.

    Manifestly antisymmetric under ``z <-> w``, as required by section 7.
    """
    xi = np.asarray(z) - np.asarray(w)
    n = kc.size
    acc = np.zeros(np.shape(xi), dtype=complex)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        ph = np.real(np.conj(kc[sl]) * xi[..., None])
        acc += (np.sin(ph) * np.conj(kc[sl])).sum(axis=-1)
    L = log_ratio(lambda_ir, lambda_uv) if log_measure is None else log_measure
    return (L / n) * acc


def empirical_dphi_self_contraction(kc: np.ndarray, lambda_ir: float,
                                    lambda_uv: float,
                                    log_measure: float | None = None) -> complex:
    r"""``E_gamma[(d_z phi_N)^2] = (L/(2N)) sum_j conj(k_j)^2``.

    This is the exact normal-ordering subtraction for ``:d phi d phi:`` at a
    fixed momentum sample.  It vanishes in the ensemble average by isotropy and
    is *identically* zero for the closed regulated kernel, because
    ``K_JJ^Lambda`` carries the prefactor ``(zbar-wbar)^2``.
    """
    n = kc.size
    L = log_ratio(lambda_ir, lambda_uv) if log_measure is None else log_measure
    return complex((L / (2.0 * n)) * np.sum(np.conj(kc) ** 2))


# --------------------------------------------------------------------------
# Physical window
# --------------------------------------------------------------------------
def physical_window(lambda_ir: float, lambda_uv: float, *, margin_uv: float,
                    margin_ir: float) -> tuple[float, float]:
    """Strict evaluation interval inside eq. (2.5).

    ``r_min = margin_uv / Lambda_UV`` and ``r_max = 1/(margin_ir * Lambda_IR)``
    so that ``Lambda_UV^{-1} << r << Lambda_IR^{-1}`` holds with an explicit,
    reported margin factor.  An empty window is returned as-is (``r_min >=
    r_max``); callers must retain and label such rows rather than drop them.
    """
    if margin_uv <= 1.0 or margin_ir <= 1.0:
        raise ValueError("window margin factors must exceed 1 for a strict window")
    return margin_uv / lambda_uv, 1.0 / (margin_ir * lambda_ir)
