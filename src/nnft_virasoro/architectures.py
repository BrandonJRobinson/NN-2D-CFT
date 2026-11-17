r"""Architecture family and the frozen negative controls (mission WP-A6, WP-C3).

Every control is defined here from the manuscript only.  In particular the
local exponent-direction control is eq. (5.9)

    p_eps(k) = k^{-2} [1 + eps (log k - <log k>)] ,

with positivity enforced on the regulated annulus.  **No collaborator-owned
spectral-exponent sweep code or data is read anywhere in this module.**

A control may act at two levels:

*feature level*  -- it changes the sampled momenta, the current-kernel
prefactor, or the feature-term structure, and therefore changes the empirical
kernel itself;

*mode level*     -- it acts on the induced pairing by a linear change of mode
coordinates ``omega -> T omega T^T`` (``sign_breaking_mode_reassignment``,
``non_symplectic_dense_mixing``).  These are the controls that probe
Proposition 3.5: they remain nondegenerate and antisymmetric, hence must still
close after an unrestricted Darboux transformation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .kernels import log_ratio

__all__ = [
    "FeatureSample",
    "ARCHITECTURES",
    "sample_architecture",
    "mode_postmap",
    "local_exponent_positivity_bound",
    "control_description",
]


@dataclass
class FeatureSample:
    """A frozen momentum sample plus the data needed by every kernel path."""

    kabs: np.ndarray          # |k_j|
    theta: np.ndarray         # arg k_j
    prefactor: np.ndarray     # per-term current-kernel prefactor
    idx_a: np.ndarray         # left feature index of each term
    idx_b: np.ndarray         # right feature index of each term
    lambda_ir: float
    lambda_uv: float
    architecture: str
    translation_invariant: bool = True
    notes: str = ""
    extra: dict = field(default_factory=dict)
    #: ``L_w`` for this sample's regulator class (== L for the sharp annulus).
    log_measure: float = 0.0

    @property
    def n_features(self) -> int:
        return int(self.kabs.size)

    @property
    def kc(self) -> np.ndarray:
        """Complex momenta ``k_1 + i k_2`` (``k.x = Re(conj(k) z)``)."""
        return self.kabs * np.exp(1j * self.theta)

    def current_kernel(self, *, chunk: int = 256):
        """Callable ``K(z,w)`` for the *general* quadrature path.

        Streamed over feature terms so a ``q x q`` contour grid never
        materializes a ``q x q x n_terms`` array.
        """
        lam = self.log_measure or log_ratio(self.lambda_ir, self.lambda_uv)
        scale = -(lam / (2.0 * self.n_features))
        kc = self.kc
        ka = np.conj(kc[self.idx_a])
        kb = np.conj(kc[self.idx_b])
        pref = self.prefactor
        n_terms = pref.size

        def kernel(z, w):
            z = np.asarray(z)
            w = np.asarray(w)
            shape = np.broadcast(z, w).shape
            acc = np.zeros(shape, dtype=complex)
            zb = np.broadcast_to(z, shape)
            wb = np.broadcast_to(w, shape)
            for start in range(0, n_terms, chunk):
                sl = slice(start, start + chunk)
                phase = np.real(ka[sl] * zb[..., None]) - np.real(
                    kb[sl] * wb[..., None]
                )
                acc += (np.cos(phase) * pref[sl]).sum(axis=-1)
            return scale * acc

        return kernel


# --------------------------------------------------------------------------
# Radial samplers
# --------------------------------------------------------------------------
def _tapered_measure(lambda_ir, lambda_uv, taper):
    from .kernels import tapered_log_measure

    return tapered_log_measure(lambda_ir, lambda_uv, taper)


def _tukey_log_window(t: np.ndarray, t0: float, t1: float, frac: float) -> np.ndarray:
    """Raised-cosine taper in ``log k`` over a fraction ``frac`` of each edge."""
    if frac <= 0:
        return np.ones_like(t)
    width = frac * (t1 - t0)
    lo = np.clip((t - t0) / width, 0.0, 1.0)
    hi = np.clip((t1 - t) / width, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * lo)) * 0.5 * (1.0 - np.cos(np.pi * hi))


def _sample_logk_uniform(rng, n, lambda_ir, lambda_uv, taper: float = 0.0):
    """``p ~ |k|^-2`` on the annulus  <=>  ``log|k|`` uniform.

    ``taper > 0`` multiplies the density by a raised-cosine window in ``log k``
    (rejection sampling).  ``taper = 0`` is the manuscript's sharp annulus,
    eq. (2.2), and is the default everywhere; the tapered variant is a **labelled
    regulator diagnostic only** -- it is used to separate a sharp-cutoff (Gibbs)
    systematic from an architecture failure, never to make a manuscript claim.
    """
    t0, t1 = np.log(lambda_ir), np.log(lambda_uv)
    if taper <= 0:
        return np.exp(rng.uniform(t0, t1, size=n))
    out = np.empty(n)
    filled = 0
    while filled < n:
        cand = rng.uniform(t0, t1, size=2 * (n - filled) + 32)
        acc = rng.uniform(0.0, 1.0, size=cand.size) < _tukey_log_window(
            cand, t0, t1, taper
        )
        take = cand[acc][: n - filled]
        out[filled : filled + take.size] = take
        filled += take.size
    return np.exp(out)


def local_exponent_positivity_bound(lambda_ir: float, lambda_uv: float) -> float:
    """Largest ``|eps|`` keeping ``p_eps(k) > 0`` on the annulus.

    ``log k - <log k>`` ranges over ``[-L/2, +L/2]`` with ``L =
    log(Lambda_UV/Lambda_IR)`` (``log k`` is uniform under the base measure), so
    positivity requires ``|eps| < 2/L``.
    """
    return 2.0 / log_ratio(lambda_ir, lambda_uv)


def _sample_local_exponent(rng, n, lambda_ir, lambda_uv, eps):
    """Eq. (5.9) by inverse-CDF on ``t = log k``.

    Density on ``t in [t0, t1]``:  ``f(t) ~ 1 + eps (t - tbar)``, ``tbar`` the
    midpoint.  CDF: ``F(s) = (s + eps s^2/2)/L`` with ``s = t - t0 + ... ``;
    solved in closed form below.
    """
    bound = local_exponent_positivity_bound(lambda_ir, lambda_uv)
    if abs(eps) >= bound:
        raise ValueError(
            f"local_exponent_direction eps={eps} violates the annulus positivity "
            f"bound |eps| < {bound:.6g}"
        )
    t0, t1 = np.log(lambda_ir), np.log(lambda_uv)
    L = t1 - t0
    u = rng.uniform(0.0, 1.0, size=n)
    # x = t - tbar in [-L/2, L/2]; density (1 + eps x)/L
    # F(x) = (x + L/2 + eps (x^2 - L^2/4)/2)/L
    if abs(eps) < 1e-14:
        x = (u - 0.5) * L
    else:
        a = eps / 2.0
        b = 1.0
        c = L / 2.0 - a * L * L / 4.0 - u * L
        disc = b * b - 4.0 * a * c
        x = (-b + np.sqrt(disc)) / (2.0 * a)
    tbar = 0.5 * (t0 + t1)
    return np.exp(tbar + x)


def _sample_rbf(rng, n, lambda_ir, lambda_uv):
    """Gaussian/RBF prior: ``p(k) ~ exp(-l^2 k^2/2)`` in 2-D.

    The length scale is fixed to the geometric centre of the regulated annulus,
    ``l = 1/sqrt(Lambda_IR Lambda_UV)``, so the control lives on the same
    momentum scale as the selected architecture.  ``|k|`` is then Rayleigh.
    """
    k_typ = np.sqrt(lambda_ir * lambda_uv)
    return k_typ * np.sqrt(rng.chisquare(2, size=n))


# --------------------------------------------------------------------------
# Angular samplers
# --------------------------------------------------------------------------
def _sample_theta_isotropic(rng, n):
    return rng.uniform(0.0, 2.0 * np.pi, size=n)


def _sample_theta_anisotropic(rng, n, amp):
    """``p(theta) ~ 1 + amp cos(2 theta)``, ``|amp| < 1``, by rejection."""
    if not (0.0 <= amp < 1.0):
        raise ValueError("anisotropy amplitude must lie in [0,1)")
    out = np.empty(n)
    filled = 0
    while filled < n:
        cand = rng.uniform(0.0, 2.0 * np.pi, size=2 * (n - filled) + 16)
        acc = rng.uniform(0.0, 1.0 + amp, size=cand.size) < (
            1.0 + amp * np.cos(2.0 * cand)
        )
        take = cand[acc][: n - filled]
        out[filled : filled + take.size] = take
        filled += take.size
    return out


# --------------------------------------------------------------------------
# The frozen control set
# --------------------------------------------------------------------------
_DESCRIPTIONS = {
    "selected_log_kernel":
        "Selected architecture: p(k) ~ |k|^-2 on the regulated annulus, "
        "isotropic angles, i.i.d. uniform phases (eqs. 2.1-2.2).",
    "rbf_gaussian_prior":
        "Gaussian/RBF spectral prior p(k) ~ exp(-l^2k^2/2), l = "
        "(Lambda_IR Lambda_UV)^{-1/2}: wrong radial spectral weight, correct "
        "isotropy (app. E).",
    "sign_breaking_mode_reassignment":
        "Mode-level control: relabel the Laurent modes by a permutation that "
        "breaks the m <-> -m pairing structure (cyclic shift within each sign "
        "block). Pairing stays nondegenerate and antisymmetric (app. E).",
    "non_symplectic_dense_mixing":
        "Mode-level control: dense non-symplectic congruence omega -> T omega T^T "
        "with T a well-conditioned random real matrix. Tests Proposition 3.5 "
        "(app. E).",
    "anisotropic_angular_prior":
        "p(theta) ~ 1 + A cos 2theta with correct |k|^-2 radial weight: correct "
        "radial spectrum, broken rotational invariance (app. E).",
    "phase_correlated_features":
        "Features paired to share one uniform phase, so the phase-averaged "
        "kernel acquires cross terms cos(Re(conj(k_a)z) - Re(conj(k_b)w)) and is "
        "no longer translation invariant (app. E).",
    "local_exponent_direction":
        "Eq. (5.9): p_eps(k) = k^-2 [1 + eps(log k - <log k>)], positivity "
        "enforced on the annulus. Probes the power-law exponent direction "
        "locally, independently of any collaborator exponent sweep.",
    "angular_mode_scramble":
        "Angular/mode scramble: |k_j| is drawn from the selected |k|^-2 annular "
        "law but theta_{k_j} is drawn from a fixed random finite set of "
        "n_scramble_directions directions instead of the uniform circle. The "
        "radial spectral measure is preserved exactly while rotational "
        "invariance -- and hence the chiral/angular organization of the mode "
        "map -- is destroyed. Unlike a kernel-level prefactor surgery this is a "
        "genuine field-level architecture, so the same control drives the "
        "pairing, the algebra, and the Ward audits. Frozen replacement for the "
        "legacy weak 'scrambled_angular_projection'.",
}

ARCHITECTURES: tuple[str, ...] = tuple(_DESCRIPTIONS)


def control_description(name: str) -> str:
    try:
        return _DESCRIPTIONS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown architecture control {name!r}; known: {ARCHITECTURES}"
        ) from exc


def sample_architecture(name: str, rng: np.random.Generator, n_features: int,
                        lambda_ir: float, lambda_uv: float,
                        *, params: dict | None = None) -> FeatureSample:
    """Draw a frozen feature sample for one architecture control."""
    params = dict(params or {})
    control_description(name)  # validates
    taper = float(params.get("regulator_taper", 0.0))
    n = int(n_features)
    idx = np.arange(n)
    trans = True
    notes = ""
    extra: dict = {}

    if name in ("selected_log_kernel", "sign_breaking_mode_reassignment",
                "non_symplectic_dense_mixing"):
        kabs = _sample_logk_uniform(rng, n, lambda_ir, lambda_uv, taper)
        theta = _sample_theta_isotropic(rng, n)
    elif name == "rbf_gaussian_prior":
        kabs = _sample_rbf(rng, n, lambda_ir, lambda_uv)
        theta = _sample_theta_isotropic(rng, n)
    elif name == "anisotropic_angular_prior":
        kabs = _sample_logk_uniform(rng, n, lambda_ir, lambda_uv, taper)
        amp = float(params.get("anisotropy_amplitude", 0.6))
        theta = _sample_theta_anisotropic(rng, n, amp)
        extra["anisotropy_amplitude"] = amp
    elif name == "local_exponent_direction":
        eps = float(params.get("local_exponent_epsilon", 0.5))
        bound = local_exponent_positivity_bound(lambda_ir, lambda_uv)
        eps = float(np.clip(eps, -0.95 * bound, 0.95 * bound))
        kabs = _sample_local_exponent(rng, n, lambda_ir, lambda_uv, eps)
        theta = _sample_theta_isotropic(rng, n)
        extra.update(local_exponent_epsilon=eps,
                     local_exponent_positivity_bound=bound)
    elif name == "angular_mode_scramble":
        kabs = _sample_logk_uniform(rng, n, lambda_ir, lambda_uv, taper)
        n_dir = int(params.get("n_scramble_directions", 6))
        dirs = rng.uniform(0.0, 2.0 * np.pi, size=n_dir)
        theta = dirs[rng.integers(0, n_dir, size=n)]
        extra["n_scramble_directions"] = n_dir
        extra["scramble_directions"] = dirs.tolist()
    elif name == "phase_correlated_features":
        kabs = _sample_logk_uniform(rng, n, lambda_ir, lambda_uv, taper)
        theta = _sample_theta_isotropic(rng, n)
        trans = False
    else:  # pragma: no cover
        raise AssertionError(name)

    kc = kabs * np.exp(1j * theta)

    if name == "phase_correlated_features":
        if n % 2:
            raise ValueError("phase_correlated_features needs an even feature count")
        even = idx[0::2]
        odd = idx[1::2]
        # diagonal self terms plus the two cross terms of each shared-phase pair
        idx_a = np.concatenate([idx, even, odd])
        idx_b = np.concatenate([idx, odd, even])
        prefactor = np.conj(kc[idx_a]) * np.conj(kc[idx_b])
        notes = "features paired to share one uniform phase"
        extra["pair_block_size"] = 2
    else:
        prefactor = np.conj(kc) ** 2
        idx_a = idx_b = idx

    return FeatureSample(
        kabs=kabs, theta=theta, prefactor=prefactor, idx_a=idx_a, idx_b=idx_b,
        lambda_ir=lambda_ir, lambda_uv=lambda_uv, architecture=name,
        translation_invariant=trans, notes=notes,
        extra={**extra, "regulator_taper": taper},
        log_measure=_tapered_measure(lambda_ir, lambda_uv, taper),
    )


def mode_postmap(name: str, modes: np.ndarray, rng: np.random.Generator,
                 *, params: dict | None = None) -> np.ndarray | None:
    """Mode-level congruence ``T`` for the two mode-level controls, else ``None``.

    ``omega -> T omega T^T`` in the mode basis ``modes``.
    """
    params = dict(params or {})
    modes = np.asarray(modes, dtype=int)
    d = modes.size
    if name == "sign_breaking_mode_reassignment":
        # Cyclic shift inside each sign block: |m| -> |m|+1 (mod cutoff).
        pos = np.where(modes > 0)[0]
        neg = np.where(modes < 0)[0]
        perm = np.arange(d)
        perm[pos] = np.roll(pos, 1)
        perm[neg] = neg  # only the positive block is shifted -> m <-> -m broken
        T = np.zeros((d, d))
        T[np.arange(d), perm] = 1.0
        return T
    if name == "non_symplectic_dense_mixing":
        target = float(params.get("mixing_condition_number", 8.0))
        # Well-conditioned dense real matrix with prescribed condition number.
        q1 = np.linalg.qr(rng.standard_normal((d, d)))[0]
        q2 = np.linalg.qr(rng.standard_normal((d, d)))[0]
        s = np.geomspace(1.0, target, d)
        return q1 @ np.diag(s) @ q2
    return None
