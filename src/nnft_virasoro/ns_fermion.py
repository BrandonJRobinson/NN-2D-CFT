r"""Neveu-Schwarz Cauchy-kernel architecture and its ordered pairing (WP-D).

Section 7.  The Grassmann architecture has ``K_psipsi(z,w) = 1/(z-w)`` in the
conformal window, and for half-integer ``r,s``

    J^>_{rs} = oint_{Rout} dz/(2pi i) z^{r-1/2} oint_{Rin} dw/(2pi i) w^{s-1/2} K
    J^<_{rs} = (radii exchanged)
    eta_{rs} = J^>_{rs} - J^<_{rs} = delta_{r+s,0} .

Finite-feature realization.  The manuscript fixes the fermion kernel only by its
conformal form, so the suite derives it from the *same* Log-Kernel feature map by
one holomorphic derivative of the scalar seed:

    K_psipsi = -d_z K            (exactly ``1/(z-w)`` when ``K = -log|z-w|^2``)
    K_psipsi^{(N,Lambda)}(z,w) = (L/N) sum_j conj(k_j) sin(Re(conj(k_j)(z-w)))
    K_psipsi^{Lambda}(z,w)     = (J_0(Lambda_IR r) - J_0(Lambda_UV r))/(z-w)

Both are manifestly antisymmetric under ``z <-> w``, which is the property
section 7 uses to convert the ordering discontinuity into the **symmetric** CAR
pairing ``{beta_r, beta_s} = delta_{r+s,0}``.  The CAR is therefore
architecture-derived, not postulated.

Ideal-limit proof (unit-tested): for ``|z|>|w|``, ``1/(z-w) = sum_k w^k z^{-k-1}``
forces ``k = r-1/2 >= 0``, giving ``J^> = 1_{r>0} delta_{r+s,0}``; the opposite
ordering gives ``J^< = -1_{r<0} delta_{r+s,0}``; their difference is
``delta_{r+s,0}`` for every half-integer ``r``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv

from .kernels import log_ratio
from .ordered_pairing import PairingResult, _gauss_legendre_panels

__all__ = [
    "fermion_mode_window",
    "canonical_car",
    "car_metrics",
    "fermionic_pairing_ideal_quadrature",
    "fermionic_pairing_empirical_exact",
    "fermionic_pairing_analytic_spectral",
    "sampled_fermion_correlator",
]


def fermion_mode_window(fermion_cutoff: int) -> np.ndarray:
    """Odd integers ``2r`` with ``|2r| <= fermion_cutoff`` (NS half-integers)."""
    t = np.array([x for x in range(1, int(fermion_cutoff) + 1) if x % 2], dtype=int)
    if t.size == 0:
        raise ValueError("fermion_cutoff must admit at least 2r = 1")
    return np.concatenate([-t[::-1], t])


def canonical_car(two_r: np.ndarray) -> np.ndarray:
    """``eta_can[r,s] = delta_{r+s,0}``, eq. (7.5)."""
    t = np.asarray(two_r, dtype=int)
    return (t[:, None] + t[None, :] == 0).astype(complex)


def car_metrics(eta: np.ndarray, two_r: np.ndarray) -> dict:
    t = np.asarray(two_r, dtype=int)
    can = canonical_car(t)
    diag = (t[:, None] + t[None, :]) == 0
    on = float(np.sqrt(np.sum(np.abs(eta[diag]) ** 2)))
    off = float(np.sqrt(np.sum(np.abs(eta[~diag]) ** 2)))
    nrm = float(np.linalg.norm(eta))
    denom = float(np.vdot(eta, eta).real)
    s_star = float(np.vdot(eta, can).real / denom) if denom > 0 else 0.0
    return dict(
        fermionic_pairing_relative_error=float(
            np.linalg.norm(eta - can) / np.linalg.norm(can)),
        fermionic_pairing_relative_error_calibrated=float(
            np.linalg.norm(s_star * eta - can) / np.linalg.norm(can)),
        fermionic_amplitude_calibration=s_star,
        fermionic_pairing_leakage=float(off / on) if on > 0 else float("inf"),
        raw_car_defect=float(np.linalg.norm(eta - can) / np.linalg.norm(can)),
        car_symmetry_defect=float(np.linalg.norm(eta - eta.T) / nrm) if nrm else 0.0,
        car_normalization="R0_units",
    )


def _P_half(two_r: np.ndarray, r_contour: float, r0: float, kabs: np.ndarray,
            theta: np.ndarray) -> np.ndarray:
    r"""``oint dz/(2pi i) z^{r-1/2} e^{i Re(conj(k) z)}`` with ``mu = r-1/2 = (2r-1)/2``.

    ``z^{r-1/2}`` is an integer power for half-integer ``r``, so the integrand is
    single valued.  ``mu + 1 = (2r+1)/2`` -> Bessel order ``(t+1)/2`` with ``t=2r``.
    """
    t = np.asarray(two_r, dtype=int)
    order = ((t + 1) // 2)[:, None]
    bess = jv(order, r_contour * kabs[None, :])
    phase = np.exp(1j * order * theta[None, :])
    # (R/R_0)^order only; the single overall R_0 (dimension L^{r+s} with r+s=0
    # on the antidiagonal) is applied once by the caller.
    radial = (r_contour / r0) ** order
    return radial * (1j**order) * bess * phase


def _parity_sign_half(two_r: np.ndarray) -> np.ndarray:
    r"""``sin`` analogue of the bosonic parity factor.

    ``sin(X-Y) = (1/2i)[e^{iX}e^{-iY} - e^{-iX}e^{iY}]`` and
    ``Q_mu = (-1)^{mu+1} P_mu`` with ``mu+1 = (t+1)/2`` give the factor
    ``(1/2i)[(-1)^{nu+1} - (-1)^{mu+1}]``, nonzero only when ``mu`` and ``nu``
    have opposite parity -- automatic on the antidiagonal ``r+s=0``.
    """
    t = np.asarray(two_r, dtype=int)
    mu1 = ((t + 1) // 2)[:, None]
    nu1 = ((t + 1) // 2)[None, :]
    return ((-1.0) ** nu1 - (-1.0) ** mu1) / (2.0j)


def fermionic_pairing_ideal_quadrature(two_r: np.ndarray, r_in: float, r_out: float,
                                       *, q: int, r0: float = 1.0) -> PairingResult:
    """Eq. (7.5) check for ``K_psipsi = 1/(z-w)`` by torus quadrature."""
    t = np.asarray(two_r, dtype=int)
    theta = 2.0 * np.pi * np.arange(q) / q

    def functional(r_z, r_w):
        zs = r_z * np.exp(1j * theta)
        ws = r_w * np.exp(1j * theta)
        kmat = 1.0 / (zs[:, None] - ws[None, :])
        coeff = np.fft.fft2(kmat) / (q * q)
        order = (t + 1) // 2
        jz = (-order) % q
        jw = (-order) % q
        core = coeff[np.ix_(jz, jw)]
        wz = (r_z / r0) ** order * r0
        ww = (r_w / r0) ** order * r0
        return core * wz[:, None] * ww[None, :]

    ig = functional(r_out, r_in)
    il = functional(r_in, r_out)
    return PairingResult(omega=ig - il, modes=t, i_greater=ig, i_less=il,
                         meta=dict(path="fermionic_ideal_quadrature",
                                   quadrature_points=q))


def fermionic_pairing_empirical_exact(two_r: np.ndarray, r_in: float, r_out: float,
                                       r0: float, kabs: np.ndarray,
                                       theta: np.ndarray, lambda_ir: float,
                                       lambda_uv: float, *,
                                       chunk: int = 200_000,
                                       log_measure: float | None = None) -> PairingResult:
    r"""Exact ordered pairing of ``K_psipsi^{(N,Lambda)}`` (deterministic_finite_feature)."""
    t = np.asarray(two_r, dtype=int)
    kabs = np.asarray(kabs, dtype=float)
    theta = np.asarray(theta, dtype=float)
    n = kabs.size
    # J^>_{rs} has dimension L^{r+s}: exactly one power of R_0 overall.
    lam = log_ratio(lambda_ir, lambda_uv) if log_measure is None else float(log_measure)
    scale = r0 * lam / n
    sgn = _parity_sign_half(t)
    acc_g = np.zeros((t.size, t.size), dtype=complex)
    acc_l = np.zeros((t.size, t.size), dtype=complex)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        ka, th = kabs[sl], theta[sl]
        pref = ka * np.exp(-1j * th)          # conj(k_j)
        p_out = _P_half(t, r_out, r0, ka, th)
        p_in = _P_half(t, r_in, r0, ka, th)
        acc_g += (p_out * pref[None, :]) @ p_in.T
        acc_l += (p_in * pref[None, :]) @ p_out.T
    ig = scale * sgn * acc_g
    il = scale * sgn * acc_l
    return PairingResult(omega=ig - il, modes=t, i_greater=ig, i_less=il,
                         meta=dict(path="fermionic_empirical_exact", n_features=n,
                                   quadrature_points=0))


def fermionic_pairing_analytic_spectral(two_r: np.ndarray, r_in: float, r_out: float,
                                        r0: float, lambda_ir: float,
                                        lambda_uv: float, *, nodes_per_wave: int = 8,
                                        taper: float = 0.0) -> PairingResult:
    r"""Exact regulated fermionic pairing (``analytic_closed_kernel``).

    Isotropy again forces the antidiagonal: with ``t = 2r`` and ``s = -r``,
    the phase factor is ``e^{i(mu+1+nu+1-1)theta} = e^{i 0}``, and

        J^>_{r,-r} = sgn(r) * (R_z/R_0)^{(t+1)/2} (R_w/R_0)^{(1-t)/2} R_0^2
                     * int dk w(k) J_{(t+1)/2}(R_z k) J_{(1-t)/2}(R_w k)
    """
    from .architectures import _tukey_log_window

    t = np.asarray(two_r, dtype=int)
    sgn = _parity_sign_half(t)
    pts, wts = _gauss_legendre_panels(
        lambda_ir, lambda_uv, np.pi / (nodes_per_wave * max(r_in, r_out)), nodes=8)
    wgt = (_tukey_log_window(np.log(pts), np.log(lambda_ir), np.log(lambda_uv), taper)
           if taper > 0 else 1.0)

    def functional(r_z, r_w):
        out = np.zeros((t.size, t.size), dtype=complex)
        for i, ti in enumerate(t):
            tj = -int(ti)
            j = int(np.where(t == tj)[0][0])
            oz = (int(ti) + 1) // 2
            ow = (tj + 1) // 2
            integral = float(np.sum(wts * wgt * jv(oz, r_z * pts) * jv(ow, r_w * pts)))
            # i^{oz} i^{ow} come from the two Jacobi-Anger factors P_mu, P_nu;
            # combined with sgn they make the antidiagonal real.
            radial = ((r_z / r0) ** oz) * ((r_w / r0) ** ow) * r0
            out[i, j] = sgn[i, j] * (1j ** (oz + ow)) * radial * integral
        return out

    ig = functional(r_out, r_in)
    il = functional(r_in, r_out)
    return PairingResult(omega=ig - il, modes=t, i_greater=ig, i_less=il,
                         meta=dict(path="fermionic_analytic_spectral",
                                   quadrature_points=int(pts.size),
                                   regulator_taper=float(taper)))


def sampled_fermion_correlator(rng, *, n_features: int, n_networks: int,
                               separations: np.ndarray, lambda_ir: float,
                               lambda_uv: float,
                               architecture: str = "selected_log_kernel",
                               params: dict | None = None,
                               batch_size: int = 96) -> dict:
    r"""Sampled fermion / supercurrent correlator diagnostics (WP-D, appendix F).

    The Grassmann sector has no scalar sampler of its own, so the *sampled*
    diagnostic reported here is the finite-network estimate of the holomorphic
    seed that defines ``K_psipsi = -d_z K``:

        Khat_psipsi(r) = -(1/M) sum_Theta d_z phi(r;Theta) phi(0;Theta) * (a factor)

    Concretely we sample ``-<d phi(z) phi(0)>`` over independent realizations,
    which equals ``-d_z K`` in expectation, and compare with the closed
    regulated form ``(J_0(Lambda_IR r)-J_0(Lambda_UV r))/r`` and the conformal
    ``1/r``.  The supercurrent diagnostic is the same estimator contracted with
    the bosonic current, ``<J(z) d phi(0)>``.
    """
    from .kernels import fermion_kernel_analytic, fermion_kernel_ideal
    from .sampling import iter_batches

    seps = np.asarray(separations, dtype=float)
    points = np.concatenate([seps.astype(complex), [0.0 + 0.0j]])
    k = seps.size
    acc = np.zeros(k, dtype=complex)
    acc_sq = np.zeros(k, dtype=float)
    acc_super = np.zeros(k, dtype=complex)
    count = 0
    for batch in iter_batches(rng, n_features=n_features,
                              n_realizations=n_networks, points=points,
                              lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                              architecture=architecture, params=params,
                              batch_size=batch_size):
        val = -batch.dphi[:, :k] * batch.phi[:, k:]
        acc += val.sum(axis=0)
        acc_sq += (np.abs(val) ** 2).sum(axis=0)
        sup = (1j * batch.dphi[:, :k]) * batch.dphi[:, k:]
        acc_super += sup.sum(axis=0)
        count += batch.n_realizations
    mean = acc / count
    var = np.maximum(acc_sq / count - np.abs(mean) ** 2, 0.0)
    ana = np.array([complex(fermion_kernel_analytic(r + 0j, 0j, lambda_ir, lambda_uv))
                    for r in seps])
    ideal = np.array([complex(fermion_kernel_ideal(r + 0j, 0j)) for r in seps])
    return dict(
        separations=seps, fermion_correlator=mean,
        fermion_correlator_stderr=np.sqrt(var / max(count - 1, 1)),
        fermion_analytic=ana, fermion_ideal=ideal,
        supercurrent_correlator=acc_super / count,
        n_networks=int(count), n_features=int(n_features),
    )
