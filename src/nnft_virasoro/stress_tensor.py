r"""Sampled neural-ensemble ``<TT>`` central charge (WP-B).

Estimator definition (all items required by mission B1)
-------------------------------------------------------
neural field
    ``phi_N`` of eq. (2.1) with ``p_Lambda`` of eq. (2.2).
feature normalization
    ``sigma_phi^2 = 4 L_w``, the ``kappa = 1`` calibration of eq. (2.6).
stress tensor
    ``T(z) = -(1/2) :d phi d phi: (z)``, the abelian Sugawara readout
    (section 4.4), ``d = (1/2)(d_1 - i d_2)``.
normal-ordering subtraction
    per realization ``c_N = E_gamma[(d phi_N)^2] = (L_w/(2N)) sum_j conj(k_j)^2``
    -- exact at fixed momenta, vanishing in the ensemble mean by isotropy.
insertion geometry
    ``P`` probe points on a circle of radius ``R_c``:
    ``z_p = R_c exp(2 pi i p/P)``.  Chord separations
    ``d_k = 2 R_c sin(pi k/P)`` each occur ``P`` times, so every separation is
    averaged over ``P`` rotated pairs.  This is rotational averaging of a
    rotationally invariant ensemble: unbiased, and it is the only variance
    reduction used.
fit target
    ``<T(z_1)T(z_2)> = (c/2)(z_1-z_2)^{-4}``.  Rotation covariance is removed by
    forming ``U_ab = 2 T_a T_b (z_a-z_b)^4``, whose expectation is ``c``.
fit interval
    a chord sub-interval strictly inside the recorded evaluation window.
realizations / width / seeds
    from ``configs/jhep_frozen.yaml``.  **Independent named seeds are the
    blocks**; the bootstrap resamples blocks.

Two sampled estimators are reported side by side and never merged
--------------------------------------------------------------------
``direct_TT``
    ``chat = mean_{Theta, pairs} U_ab``.  Uses **no** analytic input: it is the
    raw four-field Monte Carlo average.  Its per-realization signal-to-noise is
    ``~ (1/2) corr^2`` with ``corr = |K_JJ(r)| / E|d phi|^2``, which is ``O(1e-2)``
    anywhere inside the conformal window; the achieved precision and the ``M``
    required for a target precision are both reported.
``wick_factorized``
    ``chat = (mean_{Theta,pairs} S_ab)^2`` with
    ``S_ab = -d phi_a d phi_b (z_a - z_b)^2``, i.e. ``c = (z^2 K_JJ)^2`` using the
    infinite-width Wick relation ``<TT> = (1/2)<d phi d phi>^2``.  The Gaussian
    factorization is an **explicitly disclosed analytic input**, exact at
    ``N = infinity`` with an ``O(1/N)`` correction bounded by the measured
    fourth cumulant of WP-H2.  It is far better conditioned because its noise is
    linear, not quartic, in the field.

Both are ``sampled_finite_network``; neither is a substitute for the symbolic
Sugawara theorem (Theorem 4.6).
"""
from __future__ import annotations

import numpy as np

from .sampling import iter_batches

__all__ = ["ESTIMATOR_PROVENANCE", "circle_geometry", "sampled_tt",
           "fit_central_charge"]

ESTIMATOR_PROVENANCE = {
    "neural_field": "phi_N = (sigma/sqrt N) sum_j cos(Re(conj(k_j) z) + gamma_j), eq. (2.1)",
    "spectral_prior": "p_Lambda(k) ~ 1_{annulus}/|k|^2, eq. (2.2)",
    "feature_normalization": "sigma_phi^2 = 4 L_w (kappa = 1, eq. 2.6)",
    "stress_tensor": "T = -(1/2):d phi d phi:, d = (1/2)(d_1 - i d_2), section 4.4",
    "normal_ordering_subtraction": "c_N = (L_w/(2N)) sum_j conj(k_j)^2 per realization",
    "insertion_geometry": "P points on a circle of radius R_c; chord separations "
                          "d_k = 2 R_c sin(pi k/P), each averaged over P rotated pairs",
    "fit_target": "<T(z1)T(z2)> = (c/2)(z1-z2)^{-4}; U_ab = 2 T_a T_b (z_a-z_b)^4",
    "estimators": "direct_TT (no analytic input) and wick_factorized "
                  "(Gaussian factorization disclosed as an analytic input)",
    "estimator_class": "sampled_finite_network",
    "blocking": "one block per independent named seed; bootstrap resamples blocks",
}


def circle_geometry(n_points: int, radius: float) -> dict:
    """Probe points and the pair index lists for every chord separation."""
    p = int(n_points)
    idx = np.arange(p)
    z = radius * np.exp(2j * np.pi * idx / p)
    seps, pairs = [], []
    for k in range(1, p // 2 + 1):
        a = idx
        b = (idx + k) % p
        if k == p // 2:  # each chord counted once
            a, b = idx[: p // 2], b[: p // 2]
        seps.append(float(2.0 * radius * np.sin(np.pi * k / p)))
        pairs.append((a, b))
    return dict(points=z, separations=np.array(seps), pairs=pairs,
                n_points=p, radius=float(radius))


def sampled_tt(rng, *, n_features: int, n_networks: int, geometry: dict,
               lambda_ir: float, lambda_uv: float,
               architecture: str = "selected_log_kernel", params: dict | None = None,
               batch_size: int = 512, feature_chunk: int = 4096) -> dict:
    """Both sampled estimators, accumulated in a single streaming pass."""
    z = geometry["points"]
    pairs = geometry["pairs"]
    seps = geometry["separations"]
    nk = len(pairs)

    s_sum = np.zeros(nk, dtype=complex)     # sum of S_ab
    s_sq = np.zeros(nk, dtype=float)
    u_sum = np.zeros(nk, dtype=complex)     # sum of U_ab
    u_sq = np.zeros(nk, dtype=float)
    counts = np.zeros(nk, dtype=float)
    n_real = 0
    selfc = 0.0 + 0.0j

    for batch in iter_batches(rng, n_features=n_features, n_realizations=n_networks,
                              points=z, lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                              architecture=architecture, params=params,
                              batch_size=batch_size, feature_chunk=feature_chunk):
        d = batch.dphi                       # (m, P)
        cn = batch.self_contraction[:, None]
        t = -0.5 * (d**2 - cn)               # (m, P)
        for k, (a, b) in enumerate(pairs):
            xi = z[a] - z[b]
            s = -d[:, a] * d[:, b] * xi**2
            u = 2.0 * t[:, a] * t[:, b] * xi**4
            s_mean = s.mean(axis=1)          # average over rotated pairs
            u_mean = u.mean(axis=1)
            s_sum[k] += s_mean.sum()
            s_sq[k] += (np.abs(s_mean) ** 2).sum()
            u_sum[k] += u_mean.sum()
            u_sq[k] += (np.abs(u_mean) ** 2).sum()
            counts[k] += s_mean.size
        selfc += batch.self_contraction.sum()
        n_real += batch.n_realizations

    s_bar = s_sum / counts
    u_bar = u_sum / counts
    s_err = np.sqrt(np.maximum(s_sq / counts - np.abs(s_bar) ** 2, 0.0)
                    / np.maximum(counts - 1, 1))
    u_err = np.sqrt(np.maximum(u_sq / counts - np.abs(u_bar) ** 2, 0.0)
                    / np.maximum(counts - 1, 1))
    return dict(
        separations=seps,
        kernel_scalar=s_bar, kernel_scalar_stderr=s_err,
        c_direct_per_separation=np.real(u_bar), c_direct_stderr=u_err,
        c_wick_per_separation=np.real(s_bar) ** 2,
        c_wick_stderr=2.0 * np.abs(np.real(s_bar)) * s_err,
        n_networks=int(n_real), n_features=int(n_features),
        n_pair_samples=counts,
        normal_ordering_subtraction=complex(selfc / max(n_real, 1)),
    )


def fit_central_charge(separations: np.ndarray, values: np.ndarray,
                       stderr: np.ndarray, window: tuple[float, float]) -> dict:
    """Inverse-variance weighted mean of the per-separation ``c`` estimates."""
    r = np.asarray(separations, dtype=float)
    y = np.real(np.asarray(values, dtype=float))
    e = np.asarray(stderr, dtype=float)
    sel = (r >= window[0] - 1e-12) & (r <= window[1] + 1e-12) & np.isfinite(y)
    n = int(sel.sum())
    if n == 0:
        raise ValueError(f"fit window {window} retains no separations")
    ee = np.where(e[sel] > 0, e[sel], np.nan)
    if np.all(~np.isfinite(ee)):
        w = np.ones(n)
    else:
        w = 1.0 / np.nan_to_num(ee, nan=np.nanmax(ee)) ** 2
    c = float(np.sum(w * y[sel]) / np.sum(w))
    resid = y[sel] - c
    chi2 = float(np.sum(w * resid**2))
    return dict(
        c_exp=c, n_fit_points=n, fit_window_low=float(window[0]),
        fit_window_high=float(window[1]),
        residual_rms=float(np.sqrt(np.mean(resid**2))),
        chi2=chi2, chi2_per_dof=float(chi2 / max(n - 1, 1)),
        spread_across_separations=float(np.std(y[sel], ddof=1)) if n > 1 else 0.0,
        propagated_stderr=float(np.sqrt(1.0 / np.sum(w))) if np.sum(w) > 0
        else float("nan"),
    )
