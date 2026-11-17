r"""Darboux canonicalization and *physical* admissibility (appendix C).

Proposition 3.5 makes symplectic canonicalization universal for nondegenerate
antisymmetric forms, so post-Darboux closure is **not** architecture evidence.
Appendix C therefore requires four diagnostics per control:

1. symplectic defect          ``||S Omega S^T - Omega_can||_F / ||Omega_can||_F``
2. involution defect          ``||conj(S) C - C S||_F / ||S||_F``, ``C_{mn}=delta_{m,-n}``
3. conditioning of ``S``
4. positivity / normalizability of the induced positive-frequency vacuum

For (4): the physical one-particle norm is
``<alpha_{-p}0 | alpha_{-q}0> = [alpha_p, alpha_{-q}] = Omega_{p,-q} = p delta_{pq}``.
In the canonicalized basis ``f_p = sum_i S_{pi} e_i`` this reads

    G = S (Omega C) S^dagger  restricted to  p > 0 ,

which equals ``diag(p)`` (positive) in the canonical case.  A control that
closes algebraically but has ``min eig G <= 0``, or a non-Hermitian ``G``, does
not define a normalizable positive-frequency Fock vacuum, so its post-Darboux
closure is strictly weaker than unitary equivalence of the readouts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "involution_matrix",
    "darboux_antisymmetric",
    "isotropic_symmetric",
    "admissibility_report",
    "car_canonicalization_report",
]


def involution_matrix(modes: np.ndarray) -> np.ndarray:
    """``C_{mn} = delta_{m,-n}``, eq. (3.31)-(3.32)."""
    m = np.asarray(modes, dtype=int)
    return (m[:, None] + m[None, :] == 0).astype(float)


def _pivot(gram: np.ndarray, rows: list[int]) -> tuple[int, int, complex]:
    sub = gram[np.ix_(rows, rows)]
    flat = int(np.argmax(np.abs(sub)))
    i, j = divmod(flat, len(rows))
    return rows[i], rows[j], sub[i, j]


def darboux_antisymmetric(omega: np.ndarray, modes: np.ndarray,
                          *, tol: float = 1e-12) -> np.ndarray:
    r"""Return ``S`` with ``S Omega S^T = Omega_can``, ``(Omega_can)_{mn}=m delta_{m+n,0}``.

    Constructive symplectic Gram-Schmidt with full pivoting: at each step the
    largest-magnitude pairing entry defines the next conjugate pair, which is
    then normalized to ``Omega(f_p, f_{-p}) = p`` and projected out of the
    remaining subspace.  Works over the complex mode space (the physical Laurent
    modes are complex, cf. eq. 3.31).
    """
    omega = np.asarray(omega, dtype=complex)
    modes = np.asarray(modes, dtype=int)
    d = modes.size
    if omega.shape != (d, d):
        raise ValueError(f"omega shape {omega.shape} does not match {d} modes")
    pos = sorted(int(m) for m in modes if m > 0)
    if len(pos) * 2 != d:
        raise ValueError("mode window must be symmetric and zero-free")

    basis = np.eye(d, dtype=complex)  # rows are the current spanning vectors
    rows = list(range(d))
    index_of = {int(m): k for k, m in enumerate(modes)}
    S = np.zeros((d, d), dtype=complex)

    for p in sorted(pos, reverse=True):
        gram = basis @ omega @ basis.T
        i, j, val = _pivot(gram, rows)
        # Prefer the *natural* mode assignment (f_p from e_p, f_{-p} from e_{-p})
        # whenever it is numerically well conditioned.  Full pivoting alone can
        # return an orientation-reversed symplectic basis: a perfectly valid
        # Darboux map, but one that fails the involution condition eq. (3.32)
        # for purely combinatorial reasons.  The suite must be able to tell a
        # genuine star-incompatibility from an artifact of the pivot order.
        ip, im = index_of.get(p), index_of.get(-p)
        if ip in rows and im in rows and abs(gram[ip, im]) >= 0.1 * abs(val):
            i, j, val = ip, im, gram[ip, im]
        if abs(val) < tol:
            raise np.linalg.LinAlgError(
                "pairing is numerically degenerate; Darboux canonicalization "
                f"aborted at p={p} with pivot {abs(val):.3e}"
            )
        v = basis[i].copy()
        u = basis[j].copy() * (p / val)
        S[index_of[p]] = v
        S[index_of[-p]] = u
        rows = [r for r in rows if r not in (i, j)]
        if rows:
            bv = basis[rows] @ omega @ v
            bu = basis[rows] @ omega @ u
            basis[rows] = basis[rows] - np.outer(bu / p, v) + np.outer(bv / p, u)
    return S


def isotropic_symmetric(eta: np.ndarray, fermion_modes: np.ndarray,
                        *, tol: float = 1e-12) -> np.ndarray:
    r"""Return ``S`` with ``S eta S^T = eta_can``, ``(eta_can)_{rs}=delta_{r+s,0}``.

    The CAR analogue of :func:`darboux_antisymmetric`: each step extracts a
    hyperbolic plane, replaces its basis by an isotropic pair ``(e,f)`` with
    ``eta(e,e)=eta(f,f)=0`` and ``eta(e,f)=1``, and projects it out.
    """
    eta = np.asarray(eta, dtype=complex)
    tt = np.asarray(fermion_modes, dtype=int)
    d = tt.size
    if eta.shape != (d, d):
        raise ValueError("eta shape does not match fermion mode list")
    pos = sorted(int(t) for t in tt if t > 0)
    if len(pos) * 2 != d:
        raise ValueError("fermion mode window must be symmetric")
    basis = np.eye(d, dtype=complex)
    rows = list(range(d))
    index_of = {int(t): k for k, t in enumerate(tt)}
    S = np.zeros((d, d), dtype=complex)

    def form(x, y):
        return complex(x @ eta @ y)

    for t in sorted(pos, reverse=True):
        gram = basis @ eta @ basis.T
        i, j, val = _pivot(gram, rows)
        if abs(val) < tol:
            raise np.linalg.LinAlgError(
                f"CAR pairing numerically degenerate at r={t}/2 (pivot {abs(val):.3e})"
            )
        v, u = basis[i].copy(), basis[j].copy()
        a, b, c = form(v, v), form(u, u), form(v, u)
        # isotropic e in span{v,u}
        if abs(b) > tol:
            disc = np.sqrt(c * c - a * b + 0j)
            s = (-c + disc) / b
            e = v + s * u
            if abs(form(e, e)) > 1e-8 * max(1.0, abs(a) + abs(b)):
                e = v + ((-c - disc) / b) * u
        elif abs(a) <= tol:
            e = v
        else:
            e = u
        # partner
        f0 = u if abs(form(e, u)) > abs(form(e, v)) else v
        pe = form(e, f0)
        if abs(pe) < tol:  # pragma: no cover
            raise np.linalg.LinAlgError("degenerate hyperbolic plane")
        f = f0 - (form(f0, f0) / (2.0 * pe)) * e
        f = f / form(e, f)
        S[index_of[t]] = e
        S[index_of[-t]] = f
        rows = [r for r in rows if r not in (i, j)]
        if rows:
            be = basis[rows] @ eta @ e
            bf = basis[rows] @ eta @ f
            basis[rows] = basis[rows] - np.outer(bf, e) - np.outer(be, f)
    return S


# --------------------------------------------------------------------------
@dataclass
class Admissibility:
    symplectic_defect: float
    star_compatibility_defect: float
    condition_number: float
    polarization_min_eigenvalue: float
    polarization_hermiticity_defect: float
    polarization_positive: bool
    vacuum_normalizable: bool
    post_darboux_pairing_error: float
    failure: str = ""


def admissibility_report(omega: np.ndarray, modes: np.ndarray,
                         *, tol_positive: float = 0.0,
                         tol_hermitian: float = 1e-6) -> Admissibility:
    """Appendix C items 1-4 for one control."""
    from .ordered_pairing import canonical_pairing, pairing_metrics

    omega = np.asarray(omega, dtype=complex)
    modes = np.asarray(modes, dtype=int)
    can = canonical_pairing(modes)
    C = involution_matrix(modes)
    try:
        S = darboux_antisymmetric(omega, modes)
    except np.linalg.LinAlgError as exc:
        return Admissibility(
            symplectic_defect=float("nan"), star_compatibility_defect=float("nan"),
            condition_number=float("inf"), polarization_min_eigenvalue=float("nan"),
            polarization_hermiticity_defect=float("nan"),
            polarization_positive=False, vacuum_normalizable=False,
            post_darboux_pairing_error=float("nan"), failure=str(exc),
        )
    transformed = S @ omega @ S.T
    sympl = float(np.linalg.norm(transformed - can) / np.linalg.norm(can))
    star = float(np.linalg.norm(np.conj(S) @ C - C @ S) / np.linalg.norm(S))
    cond = float(np.linalg.cond(S))

    gram = S @ (omega @ C) @ S.conj().T
    pos = np.where(modes > 0)[0]
    g = gram[np.ix_(pos, pos)]
    herm_defect = float(
        np.linalg.norm(g - g.conj().T) / max(np.linalg.norm(g), 1e-300)
    )
    eig = np.linalg.eigvalsh(0.5 * (g + g.conj().T))
    min_eig = float(eig.min())
    # A genuine positive-frequency vacuum needs the Gram to be both Hermitian
    # and positive definite. Both components are also reported separately so a
    # reader can apply their own threshold.
    # `polarization_positive` is the physically meaningful statement: the
    # induced positive-frequency Gram form has no non-positive direction, so a
    # normalizable Fock vacuum exists on the audited mode window.
    positive = bool(min_eig > tol_positive)
    # `vacuum_normalizable` additionally demands Hermiticity of that Gram form
    # to an ABSOLUTE tolerance. At finite width the defect is O(N^{-1/2}) and
    # cannot meet an absolute bound, so this flag is an asymptotic statement and
    # must be read together with the admissibility-vs-width fit, never alone.
    normalizable = bool(positive and herm_defect < tol_hermitian)

    post = pairing_metrics(transformed, modes)
    return Admissibility(
        symplectic_defect=sympl, star_compatibility_defect=star,
        condition_number=cond, polarization_min_eigenvalue=min_eig,
        polarization_hermiticity_defect=herm_defect,
        polarization_positive=positive, vacuum_normalizable=normalizable,
        post_darboux_pairing_error=float(post["pairing_relative_error"]),
    )


def car_canonicalization_report(eta: np.ndarray, fermion_modes: np.ndarray) -> dict:
    """CAR canonicalization defect and involution diagnostics (WP-D)."""
    eta = np.asarray(eta, dtype=complex)
    tt = np.asarray(fermion_modes, dtype=int)
    can = (tt[:, None] + tt[None, :] == 0).astype(float)
    C = (tt[:, None] + tt[None, :] == 0).astype(float)
    raw = float(np.linalg.norm(eta - can) / np.linalg.norm(can))
    try:
        S = isotropic_symmetric(eta, tt)
    except np.linalg.LinAlgError as exc:
        return dict(raw_car_defect=raw, post_canonicalization_car_defect=float("nan"),
                    star_compatibility_defect=float("nan"),
                    condition_number=float("inf"), failure=str(exc))
    post = float(np.linalg.norm(S @ eta @ S.T - can) / np.linalg.norm(can))
    star = float(np.linalg.norm(np.conj(S) @ C - C @ S) / np.linalg.norm(S))
    return dict(raw_car_defect=raw, post_canonicalization_car_defect=post,
                star_compatibility_defect=star,
                condition_number=float(np.linalg.cond(S)), failure="")
