r"""Sampled vertex-operator weights (WP-H1) and finite-width cumulant (WP-H2).

Vertex weights.  For the neutral pair ``V_a(z) V_{-a}(0)`` the free-boson
two-point function is ``<V_a(z)V_{-a}(0)> = |z|^{2 a_1 a_2} = |z|^{-2a^2}``, and a
scalar primary of weights ``h = hbar`` scales as ``|z|^{-2(h+hbar)}``.  With
``h_a = hbar_a = a^2/2`` (eq. 6.1) the measured log-log slope ``s`` gives

    h_a = -s/4 ,   target  a^2/2 .

Sampled estimator (``sampled_finite_network``)::

    <V_a(z) V_{-a}(0)>hat = mean_Theta[ exp(i a (phi_N(z) - phi_N(0))) ]
                            * exp(a^2 K_Lambda(0))       <- normal ordering

The normal-ordering factor is ``exp(-sum_{i<j} ... )``-free here because the
configuration is a neutral pair; it is recorded explicitly.

Finite-width cumulant.  ``phi_N`` is a normalized sum of ``N`` i.i.d. bounded
terms, so its connected fourth cumulant obeys exactly

    kappa_4(N) = sigma_phi^4 kappa_4(cos) / N = -(3/8) sigma_phi^4 / N ,

i.e. ``kappa_4 ~ N^{-1}`` and standardized excess kurtosis ``-3/(2N)``.  The
measurement is interpreted **only** as finite-width non-Gaussianity, never as
evidence of a local interaction (appendix F).
"""
from __future__ import annotations

import numpy as np

from .architectures import sample_architecture
from .kernels import scalar_kernel_zero, sigma_phi_squared
from .sampling import iter_batches

__all__ = ["sampled_vertex_two_point", "fit_vertex_weight", "sampled_cumulant4",
           "cumulant4_prediction"]


def sampled_vertex_two_point(rng, *, charge: float, n_features: int,
                             n_networks: int, separations: np.ndarray,
                             lambda_ir: float, lambda_uv: float,
                             architecture: str = "selected_log_kernel",
                             params: dict | None = None, batch_size: int = 96,
                             feature_chunk: int = 8192) -> dict:
    seps = np.asarray(separations, dtype=float)
    points = np.concatenate([seps.astype(complex), [0.0 + 0.0j]])
    k = seps.size
    acc = np.zeros(k, dtype=complex)
    acc_sq = np.zeros(k, dtype=float)
    count = 0
    for batch in iter_batches(rng, n_features=n_features, n_realizations=n_networks,
                              points=points, lambda_ir=lambda_ir,
                              lambda_uv=lambda_uv, architecture=architecture,
                              params=params, batch_size=batch_size,
                              feature_chunk=feature_chunk):
        delta = batch.phi[:, :k] - batch.phi[:, k:]
        v = np.exp(1j * charge * delta)
        acc += v.sum(axis=0)
        acc_sq += (np.abs(v) ** 2).sum(axis=0)
        count += batch.n_realizations
    raw = acc / count
    norm = float(np.exp(charge * charge * scalar_kernel_zero(lambda_ir, lambda_uv)))
    var = np.maximum(acc_sq / count - np.abs(raw) ** 2, 0.0)
    return dict(
        separations=seps, correlator=raw * norm,
        correlator_stderr=np.sqrt(var / max(count - 1, 1)) * norm,
        normal_ordering_constant=norm, n_networks=int(count),
        n_features=int(n_features), charge=float(charge),
    )


def fit_vertex_weight(separations: np.ndarray, correlator: np.ndarray,
                      window: tuple[float, float], charge: float) -> dict:
    r = np.asarray(separations, dtype=float)
    y = np.real(np.asarray(correlator))
    sel = (r >= window[0] - 1e-12) & (r <= window[1] + 1e-12) & (y > 0)
    excluded = int(((r >= window[0] - 1e-12) & (r <= window[1] + 1e-12)).sum()
                   - sel.sum())
    if sel.sum() < 3:
        return dict(slope=float("nan"), h_fit=float("nan"), h_target=charge**2 / 2,
                    n_fit_points=int(sel.sum()), r_squared=float("nan"),
                    fit_window_low=float(window[0]), fit_window_high=float(window[1]),
                    excluded_nonpositive=excluded)
    A = np.column_stack([np.ones(sel.sum()), np.log(r[sel])])
    beta, *_ = np.linalg.lstsq(A, np.log(y[sel]), rcond=None)
    pred = A @ beta
    resid = np.log(y[sel]) - pred
    ss_tot = float(np.sum((np.log(y[sel]) - np.log(y[sel]).mean()) ** 2))
    return dict(
        slope=float(beta[1]), intercept=float(beta[0]),
        h_fit=float(-beta[1] / 4.0), h_target=float(charge**2 / 2.0),
        n_fit_points=int(sel.sum()),
        r_squared=float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan"),
        residual_rms=float(np.sqrt(np.mean(resid**2))),
        fit_window_low=float(window[0]), fit_window_high=float(window[1]),
        excluded_nonpositive=excluded,
    )


def cumulant4_prediction(n_features: int, lambda_ir: float, lambda_uv: float,
                         taper: float = 0.0) -> dict:
    from .kernels import tapered_log_measure

    lam_w = tapered_log_measure(lambda_ir, lambda_uv, taper)
    s4 = sigma_phi_squared(lambda_ir, lambda_uv, lam_w) ** 2
    return dict(
        kappa4_prediction=float(-3.0 * s4 / (8.0 * n_features)),
        excess_kurtosis_prediction=float(-3.0 / (2.0 * n_features)),
        variance_prediction=float(
            0.5 * sigma_phi_squared(lambda_ir, lambda_uv, lam_w)),
    )


def sampled_cumulant4(rng, *, n_features: int, n_samples: int, radius: float,
                      lambda_ir: float, lambda_uv: float,
                      architecture: str = "selected_log_kernel",
                      params: dict | None = None, chunk: int = 200_000) -> dict:
    r"""Connected fourth cumulant of ``phi_N`` at one point, ``N^{-1}`` scaling.

    Each Monte Carlo sample is a fresh realization ``({k_j},{gamma_j})``; the
    field is evaluated at a single point inside the window (the cumulant is
    position independent for the ensemble).  Streamed in ``chunk``-sized blocks.
    """
    from .kernels import tapered_log_measure

    lam_w = tapered_log_measure(lambda_ir, lambda_uv,
                                float((params or {}).get("regulator_taper", 0.0)))
    sigma = float(np.sqrt(sigma_phi_squared(lambda_ir, lambda_uv, lam_w)))
    z = complex(radius, 0.0)
    m1 = m2 = m3 = m4 = 0.0
    done = 0
    while done < n_samples:
        b = int(min(chunk, n_samples - done))
        fs = sample_architecture(architecture, rng, n_features * b, lambda_ir,
                                 lambda_uv, params=params)
        kc = fs.kc.reshape(b, n_features)
        gamma = rng.uniform(0.0, 2.0 * np.pi, size=(b, n_features))
        ph = np.real(np.conj(kc) * z) + gamma
        vals = (sigma / np.sqrt(n_features)) * np.cos(ph).sum(axis=1)
        m1 += vals.sum()
        m2 += (vals**2).sum()
        m3 += (vals**3).sum()
        m4 += (vals**4).sum()
        done += b
    n = float(n_samples)
    mu1 = m1 / n
    c2 = m2 / n - mu1**2
    c3 = m3 / n - 3 * mu1 * (m2 / n) + 2 * mu1**3
    c4 = m4 / n - 4 * mu1 * (m3 / n) - 3 * (m2 / n) ** 2 + 12 * mu1**2 * (m2 / n) \
        - 6 * mu1**4
    excess = c4 / c2**2 if c2 > 0 else float("nan")
    # Gaussian-theory standard error of the excess kurtosis estimator
    se_excess = float(np.sqrt(24.0 / n))
    pred = cumulant4_prediction(
        n_features, lambda_ir, lambda_uv,
        float((params or {}).get("regulator_taper", 0.0)))
    return dict(
        n_features=int(n_features), n_samples=int(n_samples),
        mean=float(mu1), variance=float(c2), cumulant3=float(c3),
        cumulant4=float(c4), excess_kurtosis=float(excess),
        excess_kurtosis_stderr=se_excess,
        cumulant4_stderr=float(se_excess * c2**2),
        evaluation_radius=float(radius), **pred,
    )
