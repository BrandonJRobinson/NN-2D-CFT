r"""Explicit randomized Cauchy factorization for the NS sector (Gate 7).

The native NS construction in :mod:`ns_fermion` *derives* the fermion kernel
from the bosonic scalar seed, ``K_psipsi = -d_z K_phiphi``. That is a legitimate
cross-check but it is not an independent Grassmann architecture: its width and
its randomness are the boson's. This module supplies the independent one.

Architecture
------------
For ``|z| > |w|`` the Cauchy kernel has the NS mode expansion

    1/(z - w) = sum_{n>=0} w^n z^{-n-1} ,

so a *random subset of NS modes*, importance-weighted, is an unbiased
finite-width realization of it. Draw ``n_1..n_N`` i.i.d. from a proposal
``P(n)`` on the non-negative integers and define the Grassmann field

    psi_N(z) = (1/sqrt N) sum_j a_j z^{-n_j-1/2} / sqrt(P(n_j)) ,

with ``{a_j, a_l} = delta_{jl}``. Radially ordering the two-point function,

    K^(N)(z,w) = (1/N) sum_j w^{n_j} z^{-n_j-1} / P(n_j)        (|z| > |w|)
               = -(1/N) sum_j z^{n_j} w^{-n_j-1} / P(n_j)       (|w| > |z|)

Both branches are unbiased for ``1/(z-w)``, and the sign flip in the second is
the antisymmetry ``1/(z-w) = -1/(w-z)`` -- the same ordering discontinuity that
section 7 converts into the CAR.

The width is ``N`` (the number of drawn modes) and the randomness is the mode
draw. Neither is shared with the bosonic momentum sample.

Why the proposal matters
------------------------
The pointwise estimator has finite variance only if the proposal decays more
slowly than the expansion it is sampling:

    Var K^(N)(z,w) < infinity   <=>   q > (|w| / |z|)^2

for a geometric ``P(n) = (1-q) q^n``. On the ordered two-radius contours
``|w|/|z| = 1/rho``, so the condition is ``q > 1/rho^2``. Below it the estimator
is unbiased but has infinite variance, and its sample paths do not settle --
which is why the proposal is a declared, tested part of the architecture rather
than an implementation detail.

Exact pairing
-------------
The ordered pairing of ``K^(N)`` can be evaluated in closed form, because the
contour integrals simply select one Laurent mode:

    J^>_{rs} = delta_{r+s,0} 1_{r>0} * phat(r-1/2) / P(r-1/2)
    J^<_{rs} = -delta_{r+s,0} 1_{r<0} * phat(-r-1/2) / P(-r-1/2)
    eta_{rs} = J^> - J^< = delta_{r+s,0} * phat(|r|-1/2) / P(|r|-1/2)

where ``phat(n)`` is the empirical frequency of mode ``n``. So the CAR structure
``delta_{r+s,0}`` is exact at every width, and the only error is the importance
ratio ``phat/P``, which has mean 1 and standard deviation

    sqrt( (1 - P(n)) / (N P(n)) )   with n = |r| - 1/2 ,

i.e. exactly the ``N^{-1/2}`` law the architecture-to-operator argument needs.
This closed form is used to validate the numerical quadrature path, not to
replace it: :func:`cauchy_pairing_quadrature` evaluates the same object by the
same torus quadrature the ideal kernel uses.
"""
from __future__ import annotations

import numpy as np

from .ns_fermion import canonical_car
from .ordered_pairing import PairingResult

__all__ = [
    "geometric_proposal",
    "estimator_variance",
    "optimal_geometric_q",
    "sample_ns_modes",
    "finite_variance_condition",
    "cauchy_kernel_estimate",
    "cauchy_pairing_exact",
    "cauchy_pairing_quadrature",
    "car_error_prediction",
]


def geometric_proposal(q: float, n_max: int) -> np.ndarray:
    """``P(n) = (1-q) q^n`` on ``0..n_max``, renormalized over the truncation.

    The truncation is numerical only: at the values used here the discarded tail
    ``q^(n_max+1)`` is below machine epsilon, and renormalizing keeps the
    estimator exactly unbiased for the truncated law.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"geometric proposal needs 0 < q < 1, got {q!r}")
    n = np.arange(int(n_max) + 1)
    p = (1.0 - q) * q**n.astype(float)
    return p / p.sum()


def finite_variance_condition(q: float, r_in: float, r_out: float) -> dict:
    """Is ``q`` large enough for the pointwise estimator to have finite variance?

    The second moment carries ``sum_n (|w|^2/|z|^2)^n / P(n)``, a geometric
    series with ratio ``(|w|/|z|)^2 / q``.
    """
    ratio = (min(r_in, r_out) / max(r_in, r_out)) ** 2
    return dict(q=float(q), radius_ratio_squared=float(ratio),
                series_ratio=float(ratio / q),
                finite_variance=bool(q > ratio))


def estimator_variance(z, w, proposal: np.ndarray, n_features: int = 1) -> float:
    r"""Exact per-estimate variance of ``K^(N)(z,w)``.

        Var = (1/N) [ sum_n |f(n)|^2 / P(n)  -  |1/(z-w)|^2 ],
        f(n) = w^n z^{-n-1}   (for |z| > |w|)

    The sum is a geometric series with ratio ``(|w|/|z|)^2 / q``: it converges
    only above the threshold. Computing it directly is what makes the
    finite-variance condition a *demonstration* -- the sample standard deviation
    cannot show the divergence, because with finitely many draws the rare
    heavy-weight modes are essentially never drawn, so an unstable estimator
    still returns a modest number.
    """
    z = complex(z)
    w = complex(w)
    lo, hi = (w, z) if abs(z) > abs(w) else (z, w)
    n = np.arange(proposal.size)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f2 = abs(lo) ** (2 * n) / abs(hi) ** (2 * n + 2)
        second = float(np.sum(np.where(proposal > 0, f2 / proposal, np.inf)))
    return (second - abs(1.0 / (z - w)) ** 2) / float(n_features)


def optimal_geometric_q(n_highest: int) -> float:
    """``q`` maximizing the smallest proposal mass over modes ``0..n_highest``.

    The binding constraint on the CAR error is the *rarest* mode in the window,
    whose weight is ``P(n_highest) = (1-q) q^n``; maximizing it gives
    ``q = n/(n+1)``. Note this differs from the ``q`` that minimizes the
    pointwise kernel variance -- the pairing is what this gate measures.
    """
    n = int(n_highest)
    return n / (n + 1.0) if n > 0 else 0.5


def sample_ns_modes(rng, n_features: int, proposal: np.ndarray) -> np.ndarray:
    """Draw ``N`` NS modes i.i.d. from ``proposal``."""
    return rng.choice(proposal.size, size=int(n_features), p=proposal)


def cauchy_kernel_estimate(z, w, modes: np.ndarray,
                           proposal: np.ndarray) -> np.ndarray:
    r"""``K^(N)(z, w)``, radially ordered, unbiased for ``1/(z-w)``.

    Evaluated per point pair with the branch chosen by ``|z|`` vs ``|w|``; the
    two branches are the two radial orderings, not two approximations.
    """
    z = np.asarray(z, dtype=complex)
    w = np.asarray(w, dtype=complex)
    m = np.asarray(modes, dtype=int)
    n = m.size
    pw = proposal[m]                                    # P(n_j)
    zb, wb = np.broadcast_arrays(z, w)
    out = np.zeros(zb.shape, dtype=complex)
    it = np.nditer(zb, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        zz, ww = zb[i], wb[i]
        if abs(zz) > abs(ww):
            terms = ww**m / zz ** (m + 1)
        else:
            terms = -(zz**m) / ww ** (m + 1)
        out[i] = np.sum(terms / pw) / n
        it.iternext()
    return out


def cauchy_pairing_exact(two_r: np.ndarray, modes: np.ndarray,
                         proposal: np.ndarray) -> PairingResult:
    r"""Closed-form ordered pairing of ``K^(N)``.

    The contour integrals select a single Laurent mode, leaving the empirical
    importance ratio on the canonical antidiagonal.
    """
    t = np.asarray(two_r, dtype=int)
    counts = np.bincount(np.asarray(modes, dtype=int), minlength=proposal.size)
    phat = counts / max(int(np.asarray(modes).size), 1)
    ratio = np.divide(phat, proposal, out=np.zeros_like(phat),
                      where=proposal > 0)

    n_pos = (t - 1) // 2                       # n = r - 1/2 for r > 0
    n_neg = (-t - 1) // 2                      # n = -r - 1/2 for r < 0
    anti = (t[:, None] + t[None, :] == 0)
    ig = np.zeros((t.size, t.size), dtype=complex)
    il = np.zeros((t.size, t.size), dtype=complex)
    for a, rr in enumerate(t):
        if rr > 0 and 0 <= n_pos[a] < ratio.size:
            ig[a, anti[a]] = ratio[n_pos[a]]
        if rr < 0 and 0 <= n_neg[a] < ratio.size:
            il[a, anti[a]] = -ratio[n_neg[a]]
    return PairingResult(omega=ig - il, modes=t, i_greater=ig, i_less=il,
                         meta=dict(path="cauchy_pairing_exact",
                                   n_features=int(np.asarray(modes).size)))


def cauchy_pairing_quadrature(two_r: np.ndarray, modes: np.ndarray,
                              proposal: np.ndarray, r_in: float, r_out: float,
                              *, q: int, r0: float = 1.0) -> PairingResult:
    """The same pairing by torus quadrature, as an independent evaluation path.

    Mirrors :func:`ns_fermion.fermionic_pairing_ideal_quadrature` exactly, with
    the ideal kernel replaced by the finite-width estimate.
    """
    t = np.asarray(two_r, dtype=int)
    theta = 2.0 * np.pi * np.arange(q) / q

    def functional(r_z, r_w):
        zs = r_z * np.exp(1j * theta)
        ws = r_w * np.exp(1j * theta)
        kmat = cauchy_kernel_estimate(zs[:, None], ws[None, :], modes, proposal)
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
                         meta=dict(path="cauchy_pairing_quadrature",
                                   quadrature_points=int(q),
                                   n_features=int(np.asarray(modes).size)))


def car_error_prediction(two_r: np.ndarray, proposal: np.ndarray,
                         n_features: int) -> np.ndarray:
    """Predicted s.d. of each antidiagonal CAR entry: ``sqrt((1-P)/(N P))``."""
    t = np.asarray(two_r, dtype=int)
    n = (np.abs(t) - 1) // 2
    p = np.where((n >= 0) & (n < proposal.size), proposal[np.clip(n, 0,
                 proposal.size - 1)], np.nan)
    return np.sqrt((1.0 - p) / (float(n_features) * p))
