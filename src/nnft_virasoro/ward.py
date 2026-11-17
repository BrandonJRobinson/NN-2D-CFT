r"""Vertex operators and Ward identities (section 6), three estimator classes.

Targets, eqs. (6.2)-(6.3), for a neutral configuration ``sum_i a_i = 0``::

    <J(z) prod V>/<prod V>  = sum_i a_i/(z-w_i)
    <T(z) prod V>/<prod V>  = sum_i [ h_i/(z-w_i)^2
                                    + (z-w_i)^{-1} d_{w_i} log <prod V> ]
    h_i = a_i^2/2 ,  d_{w_i} log <prod V> = sum_{j!=i} a_i a_j/(w_i - w_j)

For a Gaussian field with linear source ``X = sum_i a_i phi(w_i)``,

    f(z) := <d phi(z) prod V>/<prod V> = i sum_i a_i d_z G(z, w_i)
    <T(z) prod V>/<prod V> = -(1/2) f(z)^2      (T = -(1/2):d phi d phi:)
    <J(z) prod V>/<prod V> = i f(z)

so replacing ``G`` by the exact regulated kernel, by the phase-averaged
empirical kernel, or by a Monte Carlo average over independent networks gives
the three estimator classes of section 6 -- ``analytic_closed_kernel``,
``deterministic_finite_feature`` and ``sampled_finite_network``.  They are
computed by three separate functions here and are **never combined in one
uncertainty calculation**.

Complementary interpretation (reported verbatim in the generated summary):
pairing fidelity primarily tests *radial* spectral structure; Ward identities
test *chiral/angular* organization; post-Darboux closure tests neither.
"""
from __future__ import annotations

import numpy as np

from .kernels import k_d1, log_ratio, scalar_kernel_zero, sigma_phi_squared

__all__ = [
    "check_neutral",
    "current_target",
    "stress_target",
    "analytic_ward",
    "deterministic_ward",
    "sampled_ward",
    "COMPLEMENTARY_INTERPRETATION",
]

COMPLEMENTARY_INTERPRETATION = (
    "Pairing fidelity primarily tests the radial spectral structure of the prior; "
    "the current and stress Ward identities test the chiral/angular organization; "
    "post-Darboux closure tests neither (Proposition 3.5)."
)


def check_neutral(charges: np.ndarray, tol: float = 1e-12) -> None:
    total = float(np.sum(np.asarray(charges, dtype=float)))
    if abs(total) > tol:
        raise ValueError(
            f"Ward identities eq. (6.2)-(6.3) are stated for neutral "
            f"configurations; sum a_i = {total:.3e}"
        )


def current_target(z, charges: np.ndarray, positions: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    a = np.asarray(charges, dtype=float)
    w = np.asarray(positions, dtype=complex)
    return np.sum(a / (z[..., None] - w), axis=-1)


def stress_target(z, charges: np.ndarray, positions: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    a = np.asarray(charges, dtype=float)
    w = np.asarray(positions, dtype=complex)
    h = 0.5 * a**2
    diff = w[:, None] - w[None, :]
    np.fill_diagonal(diff, np.inf)
    dlog = np.sum((a[:, None] * a[None, :]) / diff, axis=1)
    zw = z[..., None] - w
    return np.sum(h / zw**2 + dlog / zw, axis=-1)


def _errors(estimate: np.ndarray, target: np.ndarray) -> dict:
    est = np.asarray(estimate)
    tgt = np.asarray(target)
    absolute = np.abs(est - tgt)
    denom = np.abs(tgt)
    relative = np.where(denom > 0, absolute / np.maximum(denom, 1e-300), np.nan)
    return dict(absolute=absolute, relative=relative)


# --------------------------------------------------------------------------
def analytic_ward(lambda_ir: float, lambda_uv: float, charges: np.ndarray,
                  positions: np.ndarray, probes: np.ndarray) -> dict:
    """``analytic_closed_kernel``: exact regulated Bessel kernel."""
    check_neutral(charges)
    a = np.asarray(charges, dtype=float)
    w = np.asarray(positions, dtype=complex)
    z = np.asarray(probes, dtype=complex)
    xi = z[:, None] - w[None, :]
    u = np.abs(xi) ** 2
    dG = k_d1(u, lambda_ir, lambda_uv) * np.conj(xi)   # d_z K_Lambda
    f = 1j * np.sum(a * dG, axis=1)
    cur = 1j * f
    tt = -0.5 * f**2
    return _pack(cur, tt, charges, positions, probes)


def deterministic_ward(fs, charges: np.ndarray, positions: np.ndarray,
                       probes: np.ndarray, *, chunk: int = 8192) -> dict:
    """``deterministic_finite_feature``: fixed momentum sample, phases averaged.

    ``d_z K^{(N)}(xi) = -(L/N) sum_l conj(k_l) sin(Re(conj(k_l) xi))``.
    Works for every architecture in :mod:`nnft_virasoro.architectures`, including
    the non-translation-invariant phase-correlated control.
    """
    check_neutral(charges)
    a = np.asarray(charges, dtype=float)
    w = np.asarray(positions, dtype=complex)
    z = np.asarray(probes, dtype=complex)
    lam = fs.log_measure or log_ratio(fs.lambda_ir, fs.lambda_uv)
    kc = fs.kc
    n = kc.size
    xi = z[:, None] - w[None, :]
    acc = np.zeros(xi.shape, dtype=complex)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        kk = np.conj(kc[sl])
        ph = np.real(kk * xi[..., None])
        acc += (np.sin(ph) * kk).sum(axis=-1)
    dG = -(lam / n) * acc
    f = 1j * np.sum(a * dG, axis=1)
    return _pack(1j * f, -0.5 * f**2, charges, positions, probes)


def sampled_ward(rng, *, n_features: int, n_networks: int, charges: np.ndarray,
                 positions: np.ndarray, probes: np.ndarray, lambda_ir: float,
                 lambda_uv: float, architecture: str = "selected_log_kernel",
                 params: dict | None = None, batch_size: int = 128,
                 feature_chunk: int = 8192) -> dict:
    r"""``sampled_finite_network``: Monte Carlo over independent realizations.

    Ratio estimator::

        V(Theta)      = exp(i sum_i a_i phi_N(w_i; Theta))
        <J prod V>hat = mean_Theta[ i d phi_N(z;Theta) V(Theta) ]
        <T prod V>hat = mean_Theta[ -(1/2)((d phi_N(z))^2 - c_N(Theta)) V(Theta) ]
        estimate      = numerator / mean_Theta[V]

    The normal-ordering subtraction ``c_N(Theta) = E_gamma[(d phi_N)^2] =
    (L/(2N)) sum_j conj(k_j)^2`` is exact at fixed momenta and is reported.  The
    overall vertex normal-ordering constant ``exp(sum_i a_i^2 K_Lambda(0)/2)``
    cancels in the ratio; it is recorded for provenance.
    """
    from .sampling import iter_batches

    check_neutral(charges)
    a = np.asarray(charges, dtype=float)
    w = np.asarray(positions, dtype=complex)
    z = np.asarray(probes, dtype=complex)
    points = np.concatenate([w, z])
    n_w = w.size

    num_j = np.zeros(z.size, dtype=complex)
    num_t = np.zeros(z.size, dtype=complex)
    den = 0.0 + 0.0j
    count = 0
    selfc_acc = 0.0 + 0.0j
    for batch in iter_batches(rng, n_features=n_features,
                              n_realizations=n_networks, points=points,
                              lambda_ir=lambda_ir, lambda_uv=lambda_uv,
                              architecture=architecture, params=params,
                              batch_size=batch_size, feature_chunk=feature_chunk):
        vtx = np.exp(1j * (batch.phi[:, :n_w] @ a))
        dphi = batch.dphi[:, n_w:]
        cn = batch.self_contraction[:, None]
        num_j += ((1j * dphi) * vtx[:, None]).sum(axis=0)
        num_t += ((-0.5 * (dphi**2 - cn)) * vtx[:, None]).sum(axis=0)
        den += vtx.sum()
        selfc_acc += batch.self_contraction.sum()
        count += batch.n_realizations
    if count == 0 or den == 0:
        raise RuntimeError("sampled Ward estimator produced an empty denominator")
    out = _pack(num_j / den, num_t / den, charges, positions, probes)
    out["n_networks"] = int(count)
    out["n_features"] = int(n_features)
    out["denominator_abs"] = float(abs(den / count))
    out["normal_ordering_subtraction"] = complex(selfc_acc / count)
    out["vertex_normal_ordering_constant"] = float(
        np.exp(0.5 * float(np.sum(a**2)) * scalar_kernel_zero(lambda_ir, lambda_uv))
    )
    out["sigma_phi_squared"] = float(sigma_phi_squared(lambda_ir, lambda_uv))
    return out


def _pack(current, stress, charges, positions, probes) -> dict:
    ct = current_target(probes, charges, positions)
    st = stress_target(probes, charges, positions)
    ce = _errors(current, ct)
    se = _errors(stress, st)
    return dict(
        current_estimate=np.asarray(current),
        current_target=ct,
        stress_estimate=np.asarray(stress),
        stress_target=st,
        current_absolute_error=ce["absolute"],
        current_relative_error=ce["relative"],
        stress_absolute_error=se["absolute"],
        stress_relative_error=se["relative"],
        current_max_absolute_error=float(np.nanmax(ce["absolute"])),
        current_mean_absolute_error=float(np.nanmean(ce["absolute"])),
        current_max_relative_error=float(np.nanmax(ce["relative"])),
        stress_max_absolute_error=float(np.nanmax(se["absolute"])),
        stress_mean_absolute_error=float(np.nanmean(se["absolute"])),
        stress_max_relative_error=float(np.nanmax(se["relative"])),
    )
