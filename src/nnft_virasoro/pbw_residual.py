r"""Gate 2: pre-Darboux Virasoro stability in a FIXED PBW basis.

For a finite mode window and a fixed normal ordering, the raw residual is

    R_raw(m,n) = [L_m, L_n]_Omega - (m-n) L_{m+n} - (m(m^2-1)/12) delta_{m+n,0} 1

built with the *measured* pairing Omega throughout.  The theorem under test is
that this residual is controlled by the pairing error,

    ||R_raw||  =  O( eps_Omega + eps_Omega^2 ),   eps_Omega = ||Omega - Omega_can||_F,

so that Virasoro closure degrades no worse than quadratically in the pre-Darboux
pairing error -- i.e. closure is inherited from the pairing, *before* any
canonicalization.

Fixed basis (critical)
----------------------
The residual is a polynomial of degree <= 2 in the oscillators.  It is expanded
in the **deterministic** PBW basis

    {1} u {alpha_i alpha_j : i <= j, i,j in the mode window}

whose ordering is fixed by the mode window alone and therefore does **not**
change with Omega.  Defining the norm in an Omega-dependent basis would make the
residual's size uninterpretable across architectures and widths, which is why
:func:`pbw_basis` takes only the mode window.

Three quantities are stored separately, as the gate requires: the quadratic
coefficient vector, the scalar coefficient, and their combined norm.
"""
from __future__ import annotations

import numpy as np

from .algebra import sugawara_L, virasoro_residual
from .superalgebra import Pairings, boson

__all__ = ["pbw_basis", "pbw_vector", "pbw_decompose", "raw_pbw_residual",
           "pairing_error_frobenius"]


def pbw_basis(modes) -> list[tuple]:
    """Deterministic degree-<=2 PBW basis for a mode window.

    Returns the ordered word list; index 0 is the identity.  Depends only on
    ``modes``, never on the pairing.

    The zero mode is included even though the measured pairing is supported on
    ``M^x = {-L..-1, 1..L}``: the truncated Sugawara sum runs over ``p`` with
    ``|p| <= L`` and ``|m-p| <= L``, so ``alpha_0`` genuinely appears in the
    residual (with vanishing pairing against everything).  Omitting it would
    make the expansion incomplete rather than making the term disappear.
    """
    ms = sorted({int(m) for m in modes} | {0})
    words: list[tuple] = [()]
    for a in range(len(ms)):
        for b in range(a, len(ms)):
            words.append((boson(ms[a]), boson(ms[b])))
    return words


def pbw_vector(element: dict, basis: list[tuple]) -> np.ndarray:
    """Coefficients of ``element`` in ``basis``; unknown words raise."""
    index = {w: i for i, w in enumerate(basis)}
    v = np.zeros(len(basis), dtype=complex)
    for word, coeff in element.items():
        try:
            v[index[word]] += coeff
        except KeyError as exc:                      # degree > 2 or out of window
            raise KeyError(
                f"word {word} is outside the fixed degree-<=2 PBW basis; the "
                "residual must be a polynomial of degree at most two") from exc
    return v


def pbw_decompose(element: dict, basis: list[tuple]) -> dict:
    """Split into scalar part, quadratic part, and combined norm."""
    v = pbw_vector(element, basis)
    scalar = complex(v[0])
    quad = v[1:]
    return dict(
        scalar_coefficient=scalar,
        scalar_abs=float(abs(scalar)),
        quadratic_vector=quad,
        quadratic_norm=float(np.linalg.norm(quad)),
        combined_pbw_norm=float(np.linalg.norm(v)),
        n_basis=len(basis),
    )


def pairing_error_frobenius(omega, modes) -> float:
    """``eps_Omega = ||Omega - Omega_can||_F`` in the given mode ordering."""
    from .ordered_pairing import canonical_pairing

    return float(np.linalg.norm(np.asarray(omega, dtype=complex)
                                - canonical_pairing(np.asarray(modes))))


def _residual_vectors(omega, modes, mode_cutoff, mode_pairs, basis,
                      central_charge):
    pair = Pairings.from_matrices(modes=[int(m) for m in modes], omega=omega)
    return [pbw_vector(virasoro_residual(int(m), int(n), int(mode_cutoff), pair,
                                         central_charge=central_charge), basis)
            for m, n in mode_pairs]


def raw_pbw_residual(omega, modes, mode_cutoff: int, mode_pairs,
                     *, central_charge: float = 1.0,
                     subtract_edge_baseline: bool = True) -> dict:
    r"""Raw PBW residual aggregated over ``mode_pairs``, in the fixed basis.

    Two quantities are returned and they must not be confused.

    ``*_raw``
        the residual as literally defined by the gate.  At finite mode cutoff it
        does **not** vanish for ``Omega = Omega_can``: the truncated Sugawara
        sum leaves an edge remainder (enumerated in the August WP-G audit) which
        annihilates the filtered Fock subspace but is a nonzero abstract
        element.  So ``*_raw`` has an ``Omega``-independent floor.

    ``*_excess``
        the same residual minus the ``Omega = Omega_can`` baseline computed in
        the *same* fixed basis.  This vanishes identically at
        ``eps_Omega = 0`` and is therefore the quantity the stability theorem
        governs; the bounded ratio ``R/(eps + eps^2)`` is formed from it.

    Reporting only ``*_raw`` would make the ratio diverge as ``eps -> 0`` purely
    because of the truncation floor, which would misattribute a known cutoff
    artifact to the architecture.
    """
    from .ordered_pairing import canonical_pairing

    modes = np.asarray(modes)
    basis = pbw_basis(modes)
    vec = _residual_vectors(omega, modes, mode_cutoff, mode_pairs, basis,
                            central_charge)
    base = (_residual_vectors(canonical_pairing(modes), modes, mode_cutoff,
                              mode_pairs, basis, central_charge)
            if subtract_edge_baseline else [np.zeros_like(v) for v in vec])

    def agg(vs):
        s2 = sum(abs(v[0]) ** 2 for v in vs)
        q2 = sum(float(np.linalg.norm(v[1:])) ** 2 for v in vs)
        c2 = sum(float(np.linalg.norm(v)) ** 2 for v in vs)
        return float(np.sqrt(s2)), float(np.sqrt(q2)), float(np.sqrt(c2))

    s_raw, q_raw, c_raw = agg(vec)
    s_ex, q_ex, c_ex = agg([v - b for v, b in zip(vec, base)])
    s_b, q_b, c_b = agg(base)
    per = [dict(m=int(m), n=int(n), scalar_abs=float(abs(v[0])),
                quadratic_norm=float(np.linalg.norm(v[1:])),
                combined_pbw_norm=float(np.linalg.norm(v)),
                combined_pbw_excess=float(np.linalg.norm(v - b)))
           for (m, n), v, b in zip(mode_pairs, vec, base)]
    return dict(
        scalar_norm_raw=s_raw, quadratic_norm_raw=q_raw,
        combined_pbw_norm_raw=c_raw,
        scalar_norm_excess=s_ex, quadratic_norm_excess=q_ex,
        combined_pbw_norm_excess=c_ex,
        edge_baseline_combined_norm=c_b, edge_baseline_scalar_norm=s_b,
        n_mode_pairs=len(per), n_basis=len(basis), per_pair=per,
    )
