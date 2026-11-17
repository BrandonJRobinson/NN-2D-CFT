r"""Streamed finite-network realizations and block resampling.

A *realization* is one draw of ``({k_j}, {gamma_j})`` from eqs. (2.1)-(2.2).
Realizations are generated in batches, fully vectorized over the batch, and
never all held in memory (mission engineering rule).

Field and current, eqs. (2.1) and (2.10)::

    phi_N(z)      = (sigma/sqrt N) sum_j cos(Re(conj(k_j) z) + gamma_j)
    d_z phi_N(z)  = -(sigma/(2 sqrt N)) sum_j conj(k_j) sin(Re(conj(k_j) z) + gamma_j)
    J_N(z)        = i d_z phi_N(z)

with ``sigma^2 = 4 L_w`` (the ``kappa = 1`` calibration for the sample's
regulator class).

Uncertainty policy (mission section 3): **independent named seeds define the
blocks**; the bootstrap resamples blocks, never individual realizations, and
regulator spread is never used as sampling uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import architectures as _arch
from .kernels import sigma_phi_squared, tapered_log_measure

__all__ = ["Batch", "iter_batches", "batch_momenta", "block_statistics",
           "bootstrap_ci", "jackknife"]


@dataclass
class Batch:
    """One batch of ``m`` independent realizations evaluated at ``points``."""

    phi: np.ndarray     # (m, n_points) real
    dphi: np.ndarray    # (m, n_points) complex
    n_realizations: int
    n_features: int
    self_contraction: np.ndarray  # (m,) complex; E_gamma[(d phi)^2] per realization


def batch_momenta(rng: np.random.Generator, m: int, n: int, lambda_ir: float,
                  lambda_uv: float, architecture: str,
                  params: dict | None = None) -> np.ndarray:
    """``(m, n)`` complex momenta, one independent network per row.

    Vectorized counterpart of :func:`architectures.sample_architecture` for the
    sampled audits.  Per-realization structure (e.g. the finite direction set of
    ``angular_mode_scramble``) is preserved: each row gets its own draw.
    """
    params = dict(params or {})
    taper = float(params.get("regulator_taper", 0.0))
    total = m * n
    if architecture in ("selected_log_kernel", "angular_mode_scramble",
                        "anisotropic_angular_prior",
                        "sign_breaking_mode_reassignment",
                        "non_symplectic_dense_mixing", "phase_correlated_features"):
        kabs = _arch._sample_logk_uniform(rng, total, lambda_ir, lambda_uv, taper)
    elif architecture == "rbf_gaussian_prior":
        kabs = _arch._sample_rbf(rng, total, lambda_ir, lambda_uv)
    elif architecture == "local_exponent_direction":
        eps = float(params.get("local_exponent_epsilon", 0.5))
        bound = _arch.local_exponent_positivity_bound(lambda_ir, lambda_uv)
        eps = float(np.clip(eps, -0.95 * bound, 0.95 * bound))
        kabs = _arch._sample_local_exponent(rng, total, lambda_ir, lambda_uv, eps)
    else:  # pragma: no cover
        raise KeyError(f"architecture {architecture!r} has no batch sampler")

    if architecture == "anisotropic_angular_prior":
        amp = float(params.get("anisotropy_amplitude", 0.6))
        theta = _arch._sample_theta_anisotropic(rng, total, amp)
    elif architecture == "angular_mode_scramble":
        n_dir = int(params.get("n_scramble_directions", 6))
        dirs = rng.uniform(0.0, 2.0 * np.pi, size=(m, n_dir))
        pick = rng.integers(0, n_dir, size=(m, n))
        theta = np.take_along_axis(dirs, pick, axis=1).reshape(total)
    else:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=total)
    return (kabs * np.exp(1j * theta)).reshape(m, n)


def iter_batches(rng: np.random.Generator, *, n_features: int, n_realizations: int,
                 points: np.ndarray, lambda_ir: float, lambda_uv: float,
                 architecture: str = "selected_log_kernel",
                 params: dict | None = None, batch_size: int = 256,
                 feature_chunk: int = 4096):
    """Yield :class:`Batch` objects covering ``n_realizations`` draws."""
    points = np.asarray(points, dtype=complex).reshape(-1)
    taper = float((params or {}).get("regulator_taper", 0.0))
    lam = tapered_log_measure(lambda_ir, lambda_uv, taper)
    sigma = float(np.sqrt(sigma_phi_squared(lambda_ir, lambda_uv, lam)))
    p = points.size
    done = 0
    while done < n_realizations:
        m = int(min(batch_size, n_realizations - done))
        kc = batch_momenta(rng, m, n_features, lambda_ir, lambda_uv, architecture,
                           params)
        gamma = rng.uniform(0.0, 2.0 * np.pi, size=(m, n_features))
        phi = np.zeros((m, p))
        dphi = np.zeros((m, p), dtype=complex)
        for start in range(0, n_features, feature_chunk):
            sl = slice(start, start + feature_chunk)
            kk = np.conj(kc[:, sl])                       # (m, c)
            ph = np.real(kk[:, None, :] * points[None, :, None]) + gamma[:, None, sl]
            phi += np.cos(ph).sum(axis=2)
            dphi += (np.sin(ph) * kk[:, None, :]).sum(axis=2)
        phi *= sigma / np.sqrt(n_features)
        dphi *= -(sigma / (2.0 * np.sqrt(n_features)))
        selfc = (lam / (2.0 * n_features)) * np.sum(np.conj(kc) ** 2, axis=1)
        done += m
        yield Batch(phi=phi, dphi=dphi, n_realizations=m, n_features=n_features,
                    self_contraction=selfc)


# --------------------------------------------------------------------------
# Block statistics
# --------------------------------------------------------------------------
def block_statistics(values: np.ndarray) -> dict:
    """Mean / sd / standard error over *independent blocks* (one per seed)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return dict(mean=float("nan"), sd=float("nan"), stderr=float("nan"), n_blocks=0)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    return dict(mean=float(v.mean()), sd=sd,
                stderr=float(sd / np.sqrt(n)) if n > 1 else 0.0, n_blocks=int(n))


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 20000, seed: int = 20260730,
                 alpha: float = 0.05, statistic=np.mean) -> tuple[float, float]:
    """Percentile bootstrap over blocks.  Deterministic given ``seed``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    stats = statistic(v[idx], axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def jackknife(values: np.ndarray, statistic=np.mean) -> dict:
    """Delete-one jackknife over blocks."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return dict(estimate=float(statistic(v)) if n else float("nan"),
                    stderr=float("nan"))
    full = float(statistic(v))
    loo = np.array([statistic(np.delete(v, i)) for i in range(n)])
    se = float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return dict(estimate=full, stderr=se)
