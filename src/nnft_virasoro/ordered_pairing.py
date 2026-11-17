r"""Operational two-radius ordered pairing (mission WP-A1/A2).

Definitions -- manuscript eqs. (3.2)-(3.4)::

    I^>_{mn}[K] = oint_{|z|=Rout} dz/(2pi i) z^m oint_{|w|=Rin} dw/(2pi i) w^n K(z,w)
    I^<_{mn}[K] = oint_{|z|=Rin}  dz/(2pi i) z^m oint_{|w|=Rout} dw/(2pi i) w^n K(z,w)
    omega_{mn}  = I^>_{mn} - I^<_{mn}

with ``Rout > Rin > 0``.  Proposition 3.1: for ``K = (z-w)^{-2}``,
``I^> = m 1_{m>0} delta_{m+n,0}``, ``I^< = (-m) 1_{m<0} delta_{m+n,0}``,
``omega = m delta_{m+n,0}``.

Mode normalization (declared convention)
----------------------------------------
The raw functionals scale as ``Rout^{m+1} Rin^{n+1}``.  Because the manuscript's
error metrics eq. (3.24) are unweighted Frobenius ratios, a bare
``R_0``-dependent normalization would make ``eps_leak`` depend on the arbitrary
overall contour scale (and would make Proposition 3.5's own bound diverge for
negative modes when ``R_0 != 1``).  The suite therefore reports the
**dimensionless ``R_0``-unit pairing**

    omega~_{mn} = R_0^{-(m+n)} omega_{mn} ,

which is exactly the ordered pairing of the rescaled coordinate
``zeta = z / R_0`` (contours at ``1/sqrt(rho)`` and ``sqrt(rho)``).  For the ideal
kernel ``omega~ = m delta_{m+n,0}`` identically in ``R_0`` and ``rho``, so the
convention is target-preserving; ``R_0~ = 1`` keeps the diagonal rescaling
compatible with the involution ``C`` of eq. (3.31).  Every emitted row carries
``pairing_normalization = "R0_units"``.

Three estimator paths, never mixed
----------------------------------
``pairing_ideal_quadrature``
    ``exact_symbolic`` check of Proposition 3.1 by torus quadrature.
``pairing_analytic_spectral``
    ``analytic_closed_kernel``: exact regulated spectral integral.  Rotational
    invariance of ``p_Lambda`` makes it **exactly antidiagonal**, so its leakage
    vanishes structurally; only the antidiagonal amplitude deviates.
``pairing_empirical_exact``
    ``deterministic_finite_feature``: closed-form Bessel/Jacobi-Anger
    evaluation of the contour integrals at a fixed momentum sample -- exact, no
    quadrature error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import jv

from .kernels import log_ratio

__all__ = [
    "mode_window",
    "canonical_pairing",
    "pairing_metrics",
    "ordered_functional_quadrature",
    "pairing_quadrature",
    "pairing_ideal_quadrature",
    "pairing_empirical_exact",
    "pairing_analytic_spectral",
    "radii_from_window",
    "PairingResult",
]


def mode_window(mode_cutoff: int) -> np.ndarray:
    """``M_Lambda^x = {-Lambda,...,-1,1,...,Lambda}``, eq. (3.1)."""
    if mode_cutoff < 1:
        raise ValueError("mode_cutoff must be >= 1")
    pos = np.arange(1, mode_cutoff + 1)
    return np.concatenate([-pos[::-1], pos])


def canonical_pairing(modes: np.ndarray) -> np.ndarray:
    """``(omega_can)_{mn} = m delta_{m+n,0}``, eq. (3.10)."""
    m = np.asarray(modes)
    return np.where(m[:, None] + m[None, :] == 0, m[:, None].astype(float), 0.0).astype(
        complex
    )


def radii_from_window(r_min: float, r_max: float, rho: float,
                      r0: float | None = None) -> tuple[float, float, float]:
    """Eq. (3.9): ``R_0 = sqrt(r_min r_max)``, ``R_in = R_0/sqrt(rho)``,
    ``R_out = R_0 sqrt(rho)``.  ``r0`` overrides the geometric centre for the
    radius-stability audit; the caller must check
    ``r_min < R_in < R_out < r_max``."""
    if rho <= 1.0:
        raise ValueError("rho must exceed 1 so that R_out > R_in")
    if r0 is None:
        r0 = float(np.sqrt(r_min * r_max))
    s = float(np.sqrt(rho))
    return float(r0), r0 / s, r0 * s


# --------------------------------------------------------------------------
# Metrics -- eq. (3.24)
# --------------------------------------------------------------------------
@dataclass
class PairingResult:
    omega: np.ndarray
    modes: np.ndarray
    i_greater: np.ndarray | None = None
    i_less: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def pairing_metrics(omega: np.ndarray, modes: np.ndarray) -> dict:
    """``eps_pair``, ``eps_leak`` and antidiagonal amplitude diagnostics.

    ``eps_pair``  = ||omega - omega_can||_F / ||omega_can||_F
    ``eps_leak``  = (sum_{m+n!=0}|omega|^2)^(1/2) / (sum_{m+n=0}|omega|^2)^(1/2)
    ``anti_diagonal_amplitude_error``
                  = ||diag_{m+n=0}(omega) - m||_2 / ||m||_2
    ``pairing_relative_error_calibrated``
                  = eps_pair after the single best-fit real amplitude rescaling
                    ``s* = Re<omega,omega_can>/||omega||_F^2``.  Reported so
                    architecture controls are not penalized for overall
                    normalization; leakage is already scale-free.
    ``antisymmetry_defect`` = ||omega + omega^T||_F / ||omega||_F
    """
    modes = np.asarray(modes)
    can = canonical_pairing(modes)
    diag_mask = (modes[:, None] + modes[None, :]) == 0
    num_can = np.linalg.norm(can)
    eps_pair = float(np.linalg.norm(omega - can) / num_can)

    on = float(np.sqrt(np.sum(np.abs(omega[diag_mask]) ** 2)))
    off = float(np.sqrt(np.sum(np.abs(omega[~diag_mask]) ** 2)))
    eps_leak = float(off / on) if on > 0 else float("inf")

    anti = omega[diag_mask].reshape(-1)
    # ordering of the boolean mask follows row-major (m ascending), partner -m
    anti_target = np.array([modes[i] for i in range(len(modes))], dtype=float)
    anti_err = float(
        np.linalg.norm(anti - anti_target) / np.linalg.norm(anti_target)
    )

    denom = float(np.vdot(omega, omega).real)
    s_star = float(np.vdot(omega, can).real / denom) if denom > 0 else 0.0
    eps_pair_cal = float(np.linalg.norm(s_star * omega - can) / num_can)

    nrm = float(np.linalg.norm(omega))
    anti_defect = float(np.linalg.norm(omega + omega.T) / nrm) if nrm > 0 else 0.0

    return dict(
        pairing_relative_error=eps_pair,
        pairing_leakage=eps_leak,
        anti_diagonal_amplitude_error=anti_err,
        pairing_relative_error_calibrated=eps_pair_cal,
        amplitude_calibration=s_star,
        antisymmetry_defect=anti_defect,
        pairing_frobenius=nrm,
        pairing_normalization="R0_units",
    )


# --------------------------------------------------------------------------
# General torus quadrature (works for any kernel, incl. non-translation-invariant)
# --------------------------------------------------------------------------
def ordered_functional_quadrature(kernel, modes: np.ndarray, r_z: float, r_w: float,
                                  *, q: int, r0: float) -> np.ndarray:
    r"""``I_{mn}`` by uniform periodic trapezoid + 2-D DFT, in ``R_0`` units.

    ``kernel(z, w)`` is called once on a ``q x q`` grid.  Uniform quadrature on
    a periodic analytic integrand is spectrally accurate; the ``q``-convergence
    is an explicit audit column (mission A1 test 3).
    """
    modes = np.asarray(modes, dtype=int)
    theta = 2.0 * np.pi * np.arange(q) / q
    zs = r_z * np.exp(1j * theta)
    ws = r_w * np.exp(1j * theta)
    kmat = np.asarray(kernel(zs[:, None], ws[None, :]), dtype=complex)
    if kmat.shape != (q, q):
        raise ValueError(f"kernel returned shape {kmat.shape}, expected {(q, q)}")
    coeff = np.fft.fft2(kmat) / (q * q)
    # coeff[j, l] = (1/q^2) sum K exp(-i j theta) exp(-i l psi);
    # we need frequency a = -(m+1) so index (-(m+1)) mod q.
    jz = (-(modes + 1)) % q
    jw = (-(modes + 1)) % q
    core = coeff[np.ix_(jz, jw)]
    wz = (r_z / r0) ** (modes + 1) * r0
    ww = (r_w / r0) ** (modes + 1) * r0
    return core * wz[:, None] * ww[None, :]


def pairing_quadrature(kernel, modes: np.ndarray, r_in: float, r_out: float,
                       *, q: int, r0: float) -> PairingResult:
    ig = ordered_functional_quadrature(kernel, modes, r_out, r_in, q=q, r0=r0)
    il = ordered_functional_quadrature(kernel, modes, r_in, r_out, q=q, r0=r0)
    return PairingResult(omega=ig - il, modes=np.asarray(modes), i_greater=ig,
                         i_less=il, meta=dict(quadrature_points=q, path="quadrature"))


def pairing_ideal_quadrature(modes: np.ndarray, r_in: float, r_out: float, *, q: int,
                             r0: float = 1.0) -> PairingResult:
    """Proposition 3.1 check for ``K_0 = (z-w)^{-2}``."""
    res = pairing_quadrature(lambda z, w: 1.0 / (z - w) ** 2, modes, r_in, r_out,
                             q=q, r0=r0)
    res.meta["path"] = "ideal_quadrature"
    return res


def local_residue_pairing(kernel, modes: np.ndarray, r_w: float, *, eps: float,
                          q: int, r0: float = 1.0) -> np.ndarray:
    r"""Local epsilon-residue prescription, eq. (3.13).

        oint_{|w|=R} dw/(2 pi i) w^n oint_{|z-w|=eps} dz/(2 pi i) z^m K(z,w)

    Proposition 3.2 makes this equal to the ordering discontinuity **only** when
    ``K`` is meromorphic in the swept annulus with the diagonal as its sole
    intervening singularity.  For the smooth finite-cutoff kernel the two
    prescriptions are not interchangeable; the suite reports the discrepancy
    rather than applying a residue theorem to a non-holomorphic kernel.
    """
    modes = np.asarray(modes, dtype=int)
    psi = 2.0 * np.pi * np.arange(q) / q
    ws = r_w * np.exp(1j * psi)
    zs = ws[:, None] + eps * np.exp(1j * psi)[None, :]
    kmat = np.asarray(kernel(zs, ws[:, None]), dtype=complex)
    # inner: (1/2pi i) oint dz z^m K = (eps/q) sum_l e^{i psi_l} z^m K
    inner = np.empty((modes.size, q), dtype=complex)
    for i, m in enumerate(modes):
        inner[i] = (eps / q) * np.sum(
            np.exp(1j * psi)[None, :] * zs**int(m) * kmat, axis=1
        )
    # outer: (1/2pi i) oint dw w^n f(w) = (R^{n+1}/q) sum e^{i(n+1)psi} f
    out = np.empty((modes.size, modes.size), dtype=complex)
    for j, n in enumerate(modes):
        wn = ((r_w / r0) ** (int(n) + 1)) * r0 / q
        out[:, j] = wn * (inner * np.exp(1j * (int(n) + 1) * psi)[None, :]).sum(axis=1)
    return out


# --------------------------------------------------------------------------
# Exact closed-form pairing for finite-feature (plane-wave) kernels
# --------------------------------------------------------------------------
def _P_matrix(modes: np.ndarray, r_contour: float, r0: float, kabs: np.ndarray,
              theta: np.ndarray) -> np.ndarray:
    r"""``P_m(R, k) = oint dz/(2pi i) z^m e^{i Re(conj(k) z)}``, in ``R_0`` units.

    Jacobi-Anger (appendix B) gives
    ``P_m = R^{m+1} i^{m+1} J_{m+1}(R|k|) e^{i(m+1)theta_k}``.  The radial factor
    is written as ``(R/R_0)^{m+1} R_0`` so nothing overflows for |m| ~ 16 and
    ``R_0`` far from 1.
    """
    orders = (np.asarray(modes, dtype=int) + 1)[:, None]
    bess = jv(orders, r_contour * kabs[None, :])
    phase = np.exp(1j * orders * theta[None, :])
    radial = ((r_contour / r0) ** orders) * r0
    return radial * (1j**orders) * bess * phase


def _parity_sign(modes_m: np.ndarray, modes_n: np.ndarray) -> np.ndarray:
    r"""``eps(m,n) = (-1)^{m+1}`` when ``m = n mod 2``, else 0.

    From ``cos(X-Y) = 1/2[e^{iX}e^{-iY} + e^{-iX}e^{iY}]`` together with
    ``Q_m(R,k) = (-1)^{m+1} P_m(R,k)``.
    """
    m = np.asarray(modes_m, dtype=int)[:, None]
    n = np.asarray(modes_n, dtype=int)[None, :]
    same = ((m - n) % 2) == 0
    return np.where(same, (-1.0) ** (m + 1), 0.0)


def pairing_empirical_exact(modes: np.ndarray, r_in: float, r_out: float, r0: float,
                            kabs: np.ndarray, theta: np.ndarray,
                            prefactor: np.ndarray, lambda_ir: float, lambda_uv: float,
                            *, idx_a: np.ndarray | None = None,
                            idx_b: np.ndarray | None = None,
                            chunk: int = 200_000,
                            log_measure: float | None = None) -> PairingResult:
    r"""Exact ordered pairing of the phase-averaged empirical kernel, eq. (3.6).

    The kernel is
    ``K^{(N,L)}_JJ(z,w) = -(L/(2N)) sum_t Pref_t cos(Re(conj(k_{a_t}) z) - Re(conj(k_{b_t}) w))``
    where ``t`` runs over feature terms.  For the selected architecture
    ``a_t = b_t = j`` and ``Pref_j = conj(k_j)^2``; correlated-phase controls add
    off-diagonal ``(a,b)`` terms, and the angular scramble replaces the
    prefactor.  Contour integration is then exact:

        I_{mn} = -(L/(2N)) eps(m,n) sum_t Pref_t P_m(R_z,k_{a_t}) P_n(R_w,k_{b_t}) .

    ``N`` is the feature count ``kabs.size`` (not the term count).
    """
    modes = np.asarray(modes, dtype=int)
    kabs = np.asarray(kabs, dtype=float)
    theta = np.asarray(theta, dtype=float)
    prefactor = np.asarray(prefactor, dtype=complex)
    n_features = kabs.size
    if idx_a is None:
        idx_a = np.arange(prefactor.size)
    if idx_b is None:
        idx_b = idx_a
    idx_a = np.asarray(idx_a, dtype=np.int64)
    idx_b = np.asarray(idx_b, dtype=np.int64)
    if not (idx_a.size == idx_b.size == prefactor.size):
        raise ValueError("idx_a, idx_b and prefactor must have equal length")

    lam = log_ratio(lambda_ir, lambda_uv) if log_measure is None else float(log_measure)
    scale = -(lam / (2.0 * n_features))
    eps_g = _parity_sign(modes, modes)

    # Both orderings need P at the same two radii, so each Bessel matrix is
    # built once per chunk and reused (halves the special-function cost).
    acc_g = np.zeros((modes.size, modes.size), dtype=complex)
    acc_l = np.zeros((modes.size, modes.size), dtype=complex)
    for start in range(0, idx_a.size, chunk):
        sl = slice(start, start + chunk)
        a, b, pf = idx_a[sl], idx_b[sl], prefactor[sl]
        p_out_a = _P_matrix(modes, r_out, r0, kabs[a], theta[a])
        p_in_a = _P_matrix(modes, r_in, r0, kabs[a], theta[a])
        if np.array_equal(a, b):
            p_out_b, p_in_b = p_out_a, p_in_a
        else:
            p_out_b = _P_matrix(modes, r_out, r0, kabs[b], theta[b])
            p_in_b = _P_matrix(modes, r_in, r0, kabs[b], theta[b])
        acc_g += (p_out_a * pf[None, :]) @ p_in_b.T
        acc_l += (p_in_a * pf[None, :]) @ p_out_b.T
    ig = scale * eps_g * acc_g
    il = scale * eps_g * acc_l
    return PairingResult(
        omega=ig - il, modes=modes, i_greater=ig, i_less=il,
        meta=dict(path="empirical_exact", n_features=n_features,
                  n_terms=int(idx_a.size), quadrature_points=0),
    )


# --------------------------------------------------------------------------
# Analytic closed-kernel spectral pairing
# --------------------------------------------------------------------------
def _gauss_legendre_panels(a: float, b: float, panel_width: float, nodes: int = 8):
    n_panels = max(1, int(np.ceil((b - a) / panel_width)))
    edges = np.linspace(a, b, n_panels + 1)
    x, w = np.polynomial.legendre.leggauss(nodes)
    lo = edges[:-1][:, None]
    hi = edges[1:][:, None]
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    pts = (mid + half * x[None, :]).ravel()
    wts = (half * w[None, :]).ravel()
    return pts, wts


def projection_proxy_covariance(modes: np.ndarray, r_proj: float, r0: float,
                                kabs: np.ndarray, theta: np.ndarray,
                                prefactor: np.ndarray, lambda_ir: float,
                                lambda_uv: float, *,
                                idx_a: np.ndarray | None = None,
                                idx_b: np.ndarray | None = None,
                                log_measure: float | None = None) -> np.ndarray:
    r"""Single-radius Laurent projection covariance ``C^proj_{mn}(R) = E_gamma[a_m a_n]``.

    Appendix B stores the mode map as ``a_m = sum_j F_{mj}(R)``.  Averaging the
    uniform phases of eq. (2.1) gives

        E_gamma[a_m a_n] = -(L/(2N)) eps(m,n) sum_j conj(k_j)^2 P_m(R,k_j) P_n(R,k_j),

    i.e. exactly the ordered functional evaluated at coincident radii.  It is
    therefore **symmetric**: its antisymmetrization vanishes identically, which
    is the precise sense in which "ordinary stochastic covariance is not a
    quantum commutator" (section 2.4).

    This is the suite's reproducible reconstruction of the released
    ``radial_ordered_projection_leakage`` proxy.  The legacy generating script
    is not present in the bundle, so no claim is made that the numerical value
    equals the legacy one; the proxy is carried only in clearly-separated
    comparison columns and is never identified with
    :func:`pairing_empirical_exact`.
    """
    res = pairing_empirical_exact(modes, r_proj, r_proj, r0, kabs, theta, prefactor,
                                 lambda_ir, lambda_uv, idx_a=idx_a, idx_b=idx_b,
                                 log_measure=log_measure)
    return res.i_greater


def projection_proxy_metrics(cov: np.ndarray, modes: np.ndarray) -> dict:
    """Leakage of the single-radius projection proxy (separate column family)."""
    modes = np.asarray(modes, dtype=int)
    diag_mask = (modes[:, None] + modes[None, :]) == 0
    on = float(np.sqrt(np.sum(np.abs(cov[diag_mask]) ** 2)))
    off = float(np.sqrt(np.sum(np.abs(cov[~diag_mask]) ** 2)))
    return dict(
        projection_pairing_leakage=float(off / on) if on > 0 else float("inf"),
        projection_symmetry_defect=float(
            np.linalg.norm(cov - cov.T) / max(np.linalg.norm(cov), 1e-300)
        ),
        projection_antisymmetrization_norm=float(np.linalg.norm(cov - cov.T) / 2.0),
    )


def bessel_product_moment(mu: int, nu: int, r_z: float, r_w: float,
                          k_lo: float, k_hi: float, *, nodes_per_wave: int = 6,
                          max_nodes: int = 400_000,
                          asymptotic_phase: float = 80.0,
                          taper: float = 0.0) -> tuple[float, dict]:
    r"""``int_{k_lo}^{k_hi} dk k J_mu(R_z k) J_nu(R_w k)``.

    Panelized Gauss-Legendre up to a crossover ``k_c``, then the two-term
    large-argument asymptotic expansion

        J_p(x) ~ sqrt(2/(pi x)) [cos(chi_p) - ((4p^2-1)/(8x)) sin(chi_p)],
        chi_p = x - p pi/2 - pi/4 ,

    integrated in closed form (elementary cosines plus ``Si``/``Ci``).  The
    integral is only conditionally convergent -- the integrand's amplitude is
    ``O(1)``, not decaying -- which is precisely the sharp-cutoff (Gibbs) tail
    that Proposition 3.4 predicts and that the audit must quantify.
    """
    from scipy.special import sici

    from .architectures import _tukey_log_window

    if taper > 0:
        # A tapered spectral density has no closed-form oscillatory tail; the
        # integrand decays smoothly, so the purely numerical path is used.
        pts, wts = _gauss_legendre_panels(
            k_lo, k_hi, np.pi / (nodes_per_wave * max(r_z, r_w)), nodes=8
        )
        wgt = _tukey_log_window(np.log(pts), np.log(k_lo), np.log(k_hi), taper)
        val = float(np.sum(wts * pts * wgt * jv(mu, r_z * pts) * jv(nu, r_w * pts)))
        return val, {"k_crossover": float(k_hi), "asymptotic_tail": False,
                     "numerical_nodes": int(pts.size), "taper": float(taper)}

    r_min_c = min(abs(r_z), abs(r_w))
    k_c = k_hi
    if r_min_c > 0:
        # numerical part must fit the node budget, and the asymptotic part must
        # start where both arguments are large.
        k_budget = k_lo + max_nodes * np.pi / (nodes_per_wave * 8 * max(r_z, r_w))
        k_asym = max(asymptotic_phase / r_min_c, k_lo)
        k_c = min(k_hi, max(k_asym, min(k_hi, k_budget)))

    info: dict = {"k_crossover": float(k_c), "asymptotic_tail": bool(k_c < k_hi)}
    total = 0.0
    if k_c > k_lo:
        pts, wts = _gauss_legendre_panels(
            k_lo, k_c, np.pi / (nodes_per_wave * max(r_z, r_w)), nodes=8
        )
        info["numerical_nodes"] = int(pts.size)
        total += float(np.sum(wts * pts * jv(mu, r_z * pts) * jv(nu, r_w * pts)))
    else:
        info["numerical_nodes"] = 0

    if k_c < k_hi:
        a_mu = mu * np.pi / 2.0 + np.pi / 4.0
        a_nu = nu * np.pi / 2.0 + np.pi / 4.0
        p_mu = (4.0 * mu * mu - 1.0) / 8.0
        p_nu = (4.0 * nu * nu - 1.0) / 8.0
        S, D = r_z + r_w, r_z - r_w
        phi_s, phi_d = -(a_mu + a_nu), -(a_mu - a_nu)
        pref = 1.0 / (np.pi * np.sqrt(r_z * r_w))

        def int_cos(alpha, phi):
            return (np.sin(alpha * k_hi + phi) - np.sin(alpha * k_c + phi)) / alpha

        def int_sin_over_k(alpha, phi):
            si_hi, ci_hi = sici(abs(alpha) * k_hi)
            si_lo, ci_lo = sici(abs(alpha) * k_c)
            sgn = np.sign(alpha)
            return np.cos(phi) * sgn * (si_hi - si_lo) + np.sin(phi) * (ci_hi - ci_lo)

        tail = pref * (int_cos(S, phi_s) + int_cos(D, phi_d))
        tail -= (p_mu * pref / r_z) * (
            int_sin_over_k(S, phi_s) + int_sin_over_k(D, phi_d)
        )
        tail -= (p_nu * pref / r_w) * (
            int_sin_over_k(S, phi_s) - int_sin_over_k(D, phi_d)
        )
        total += float(tail)
    return total, info


def pairing_analytic_spectral(modes: np.ndarray, r_in: float, r_out: float, r0: float,
                              lambda_ir: float, lambda_uv: float,
                              *, nodes_per_wave: int = 6,
                              max_nodes: int = 400_000,
                              taper: float = 0.0) -> PairingResult:
    r"""Exact regulated ordered pairing (``analytic_closed_kernel``).

    Averaging the empirical formula over the isotropic prior ``p_Lambda`` makes
    the angular integral ``int dtheta e^{i(m+n)theta} = 2 pi delta_{m+n,0}``, so

        I^>_{m,-m} = (1/2)(-1)^{m+1} R_out^{m+1} R_in^{1-m}
                     int_{Lambda_IR}^{Lambda_UV} dk k J_{m+1}(R_out k) J_{1-m}(R_in k)

    and **all off-antidiagonal entries vanish identically**.  Finite-``N``
    leakage is therefore purely an angular sampling effect; the regulator
    systematic shows up only in the antidiagonal amplitude.  Both statements are
    reported rather than conflated.
    """
    modes = np.asarray(modes, dtype=int)
    nodes_used = 0
    tail_used = False

    def functional(r_z: float, r_w: float) -> np.ndarray:
        nonlocal nodes_used, tail_used
        out = np.zeros((modes.size, modes.size), dtype=complex)
        for i, m in enumerate(modes):
            n = -int(m)
            j = int(np.where(modes == n)[0][0])
            integral, info = bessel_product_moment(
                int(m) + 1, n + 1, r_z, r_w, lambda_ir, lambda_uv,
                nodes_per_wave=nodes_per_wave, max_nodes=max_nodes, taper=taper,
            )
            nodes_used += info["numerical_nodes"]
            tail_used = tail_used or info["asymptotic_tail"]
            radial = ((r_z / r0) ** (int(m) + 1)) * ((r_w / r0) ** (n + 1)) * r0 * r0
            out[i, j] = 0.5 * ((-1.0) ** (int(m) + 1)) * radial * integral
        return out

    ig = functional(r_out, r_in)
    il = functional(r_in, r_out)
    return PairingResult(
        omega=ig - il, modes=modes, i_greater=ig, i_less=il,
        meta=dict(path="analytic_spectral", quadrature_points=int(nodes_used),
                  asymptotic_tail=bool(tail_used), regulator_taper=float(taper),
                  structurally_antidiagonal=True),
    )
