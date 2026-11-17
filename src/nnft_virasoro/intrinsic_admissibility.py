r"""Gate 1: intrinsic physical admissibility of the PRE-Darboux pairing.

Mission rules 3-4: post-Darboux closure is universal for nondegenerate
antisymmetric forms and is therefore *not* architecture-selection evidence.
Architecture discrimination must rest on quantities intrinsic to the pre-Darboux
pairing itself.  This module computes those quantities.

Mode ordering (fixed, and different from the frozen release's ascending order)::

    (alpha_1, ..., alpha_L, alpha_{-1}, ..., alpha_{-L})

so the involution and the pairing are block matrices

    C = [[0, I], [I, 0]],        Omega = [[A, G], [-G^T, B]]

with ``A`` the positive-frequency block, ``B`` the negative-frequency block and
``G`` the cross block.

Intrinsic diagnostics
---------------------
``delta_star``  = ||C conj(Omega) C^T + Omega||_F / ||Omega||_F
    The involution (star) condition.  Vanishes iff the pairing is compatible
    with ``alpha_m^dagger = alpha_{-m}``.
``delta_A``     = ||A||_F / ||Omega||_F
    Positive-frequency block defect; the canonical form has ``A = 0``.
``delta_H``     = ||G - G^dagger||_F / ||G||_F
    Hermiticity defect of the cross block.
``lambda_min``  = min eig( D^{-1/2} H D^{-1/2} ),  H = (G + G^dagger)/2,
                  D = diag(1, ..., L)
    Normalized positive-frequency eigenvalue.  ``H`` is used *only* for this
    signed diagnostic; ``delta_H`` is reported separately so a non-Hermitian
    ``G`` cannot hide inside a symmetrized eigenvalue.
``delta_iso``   = ||D^{-1/2} G D^{-1/2} - I||_F / sqrt(L)
    Isotropy defect: how far the normalized cross block is from the identity.

Principal physical canonicalizer
--------------------------------
When ``G`` is Hermitian positive definite the *principal* canonicalizer is

    M_0 = D^{1/2} G^{-1/2},      S_0 = diag(M_0, conj(M_0))

and it satisfies ``S_0 Omega S_0^T = Omega_can`` together with the involution
condition ``conj(S_0) C = C S_0``.  Unlike an arbitrary Darboux map this is
canonical: it is built from the pairing alone, with no free choice.  The
arbitrary-canonicalizer defect from the frozen release is retained only as an
implementation comparison.

Admissibility criterion (the "iff")
-----------------------------------
A star-compatible symplectic canonicalization of ``Omega`` exists **iff**

    A = 0  (equivalently B = 0 by the star condition),  G = G^dagger,  G > 0.

Necessity: ``S_0 Omega S_0^T = Omega_can`` with ``conj(S) C = C S`` forces the
block structure and positivity.  Sufficiency: ``M_0`` above realizes it.
:func:`physical_darboux_admissible` evaluates the criterion and
:func:`principal_canonicalizer` constructs the witness, so the equivalence is
checked numerically in both directions by the test suite.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

__all__ = [
    "physical_mode_order",
    "reorder_to_physical",
    "involution_C",
    "blocks",
    "intrinsic_diagnostics",
    "principal_canonicalizer",
    "physical_darboux_admissible",
    "IntrinsicReport",
]


def physical_mode_order(mode_cutoff: int) -> np.ndarray:
    """``(1, ..., L, -1, ..., -L)`` -- the Gate-1 ordering."""
    p = np.arange(1, int(mode_cutoff) + 1)
    return np.concatenate([p, -p])


def reorder_to_physical(omega: np.ndarray, modes: np.ndarray,
                        mode_cutoff: int | None = None) -> np.ndarray:
    """Permute a pairing given in ``modes`` order into the Gate-1 ordering."""
    modes = [int(m) for m in modes]
    L = int(mode_cutoff) if mode_cutoff else max(modes)
    target = physical_mode_order(L)
    idx = [modes.index(int(t)) for t in target]
    return np.asarray(omega, dtype=complex)[np.ix_(idx, idx)]


def involution_C(mode_cutoff: int) -> np.ndarray:
    """``C = [[0, I], [I, 0]]`` in the Gate-1 ordering."""
    L = int(mode_cutoff)
    C = np.zeros((2 * L, 2 * L))
    C[:L, L:] = np.eye(L)
    C[L:, :L] = np.eye(L)
    return C


def blocks(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into ``(A, G, B)`` with ``Omega = [[A, G], [-G^T, B]]``."""
    om = np.asarray(omega, dtype=complex)
    n = om.shape[0]
    if n % 2:
        raise ValueError("pairing dimension must be even")
    L = n // 2
    return om[:L, :L], om[:L, L:], om[L:, L:]


@dataclass
class IntrinsicReport:
    mode_cutoff: int
    delta_star: float
    delta_A: float
    delta_B: float
    delta_H: float
    lambda_min_normalized: float
    lambda_max_normalized: float
    delta_iso: float
    cross_block_norm: float
    pairing_norm: float
    antisymmetry_defect: float
    G_condition_number: float
    physically_admissible: bool
    principal_symplectic_defect: float
    principal_involution_defect: float
    principal_condition_number: float
    principal_available: bool
    principal_note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def intrinsic_diagnostics(omega_phys: np.ndarray, *, tol: float = 1e-9,
                          with_principal: bool = True) -> IntrinsicReport:
    """All Gate-1 intrinsic defects for a pairing in the physical ordering."""
    om = np.asarray(omega_phys, dtype=complex)
    L = om.shape[0] // 2
    C = involution_C(L)
    A, G, B = blocks(om)

    nrm = float(np.linalg.norm(om))
    if nrm == 0:
        raise ValueError("pairing is identically zero")
    gnrm = float(np.linalg.norm(G))

    delta_star = float(np.linalg.norm(C @ np.conj(om) @ C.T + om) / nrm)
    delta_A = float(np.linalg.norm(A) / nrm)
    delta_B = float(np.linalg.norm(B) / nrm)
    delta_H = float(np.linalg.norm(G - G.conj().T) / gnrm) if gnrm > 0 else float("nan")

    D = np.diag(np.arange(1, L + 1).astype(float))
    Dm = np.diag(1.0 / np.sqrt(np.arange(1, L + 1).astype(float)))
    H = 0.5 * (G + G.conj().T)          # ONLY for the signed eigenvalue
    Hn = Dm @ H @ Dm
    eig = np.linalg.eigvalsh(0.5 * (Hn + Hn.conj().T))
    Gn = Dm @ G @ Dm
    delta_iso = float(np.linalg.norm(Gn - np.eye(L)) / np.sqrt(L))
    anti = float(np.linalg.norm(om + om.T) / nrm)
    try:
        gcond = float(np.linalg.cond(G))
    except np.linalg.LinAlgError:
        gcond = float("inf")

    admissible = bool(delta_A <= tol and delta_B <= tol and
                      (delta_H <= tol if np.isfinite(delta_H) else False) and
                      eig.min() > tol)

    sym = inv = cond = float("nan")
    avail, note = False, ""
    if with_principal:
        try:
            S0, info = principal_canonicalizer(om)
            sym, inv, cond = info["symplectic_defect"], info["involution_defect"], \
                info["condition_number"]
            avail, note = True, info.get("note", "")
        except (np.linalg.LinAlgError, ValueError) as exc:
            note = str(exc)

    return IntrinsicReport(
        mode_cutoff=L, delta_star=delta_star, delta_A=delta_A, delta_B=delta_B,
        delta_H=delta_H, lambda_min_normalized=float(eig.min()),
        lambda_max_normalized=float(eig.max()), delta_iso=delta_iso,
        cross_block_norm=gnrm, pairing_norm=nrm, antisymmetry_defect=anti,
        G_condition_number=gcond, physically_admissible=admissible,
        principal_symplectic_defect=sym, principal_involution_defect=inv,
        principal_condition_number=cond, principal_available=avail,
        principal_note=note,
    )


def _matrix_inv_sqrt(G: np.ndarray) -> np.ndarray:
    """``G^{-1/2}`` for Hermitian positive-definite ``G`` (principal branch)."""
    Gh = 0.5 * (G + G.conj().T)
    w, V = np.linalg.eigh(Gh)
    if w.min() <= 0:
        raise np.linalg.LinAlgError(
            f"cross block is not positive definite (min eig {w.min():.4g}); "
            "the principal physical canonicalizer does not exist")
    return V @ np.diag(w ** -0.5) @ V.conj().T


def principal_canonicalizer(omega_phys: np.ndarray) -> tuple[np.ndarray, dict]:
    r"""``S_0 = diag(M_0, conj(M_0))`` with ``M_0 = D^{1/2} G^{-1/2}``.

    Returns ``(S_0, info)``.  ``info`` carries the symplectic defect
    ``||S_0 Omega S_0^T - Omega_can||_F/||Omega_can||_F`` and the involution
    defect ``||conj(S_0) C - C S_0||_F/||S_0||_F``.
    """
    om = np.asarray(omega_phys, dtype=complex)
    L = om.shape[0] // 2
    A, G, B = blocks(om)
    herm = float(np.linalg.norm(G - G.conj().T))
    note = ""
    if herm > 1e-8 * max(np.linalg.norm(G), 1e-300):
        note = (f"cross block not Hermitian (defect {herm:.3g}); the principal "
                "canonicalizer is built from its Hermitian part")
    D = np.diag(np.arange(1, L + 1).astype(float))
    M0 = np.sqrt(D) @ _matrix_inv_sqrt(G)
    S0 = np.zeros((2 * L, 2 * L), dtype=complex)
    S0[:L, :L] = M0
    S0[L:, L:] = np.conj(M0)

    # canonical target in the physical ordering: Omega_can = [[0, D], [-D, 0]]
    can = np.zeros_like(om)
    can[:L, L:] = D
    can[L:, :L] = -D
    C = involution_C(L)
    info = dict(
        symplectic_defect=float(np.linalg.norm(S0 @ om @ S0.T - can)
                                / np.linalg.norm(can)),
        involution_defect=float(np.linalg.norm(np.conj(S0) @ C - C @ S0)
                                / np.linalg.norm(S0)),
        condition_number=float(np.linalg.cond(S0)),
        note=note,
    )
    return S0, info


def physical_darboux_admissible(omega_phys: np.ndarray, *,
                                tol: float = 1e-9) -> tuple[bool, dict]:
    """The Gate-1 criterion: ``A = B = 0``, ``G = G^dagger``, ``G > 0``."""
    A, G, B = blocks(np.asarray(omega_phys, dtype=complex))
    nrm = float(np.linalg.norm(omega_phys))
    gn = float(np.linalg.norm(G))
    cond = {
        "A_vanishes": float(np.linalg.norm(A) / nrm) <= tol,
        "B_vanishes": float(np.linalg.norm(B) / nrm) <= tol,
        "G_hermitian": (float(np.linalg.norm(G - G.conj().T) / gn) <= tol
                        if gn > 0 else False),
    }
    if cond["G_hermitian"]:
        w = np.linalg.eigvalsh(0.5 * (G + G.conj().T))
        cond["G_positive"] = bool(w.min() > tol)
        cond["G_min_eigenvalue"] = float(w.min())
    else:
        cond["G_positive"] = False
        cond["G_min_eigenvalue"] = float("nan")
    return bool(all(cond[k] for k in
                    ("A_vanishes", "B_vanishes", "G_hermitian", "G_positive"))), cond
