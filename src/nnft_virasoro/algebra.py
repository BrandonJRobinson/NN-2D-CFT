r"""Truncated Sugawara generators, filtered Fock evaluation, and residuals.

Manuscript anchors
------------------
Eq. (4.6)   ``L_m = 1/2 sum_p :alpha_{m-p} alpha_p:``
Eq. (4.13)  truncation ``|p| <= Lambda`` and ``|m-p| <= Lambda``
Eq. (4.10)  ``alpha_0|a> = a|a>``, level ``l(v) = sum_i r_i``
Thm. 4.6    exact ``c=1`` Virasoro on ``F_a^{(<=K)}`` when ``K + m_- + n_- <= Lambda``
App. D      ``K_max(Lambda;m,n) = Lambda - m_- - n_-``
Eq. (5.5)   ``c_m = scalar([L_m,L_{-m}] - 2 m L_0)/(m(m^2-1)/12)``
Section 7   ``L_n^F = 1/4 sum_r (2r-n):beta_{n-r}beta_r:``, ``G_r = sum_n alpha_n beta_{r-n}``

The declared definition of the **raw Sugawara residual**
-------------------------------------------------------
Build ``L_m^{(Lambda)}`` in the ordered-word algebra using the *measured* pairing
``omega^{(N,Lambda)}`` for all reorderings, form

    Delta_{mn} = [L_m,L_n] - (m-n)L_{m+n} - (c/12) m(m^2-1) delta_{m+n,0} ,

and read the resulting element out on the filtered Fock space ``F_a^{(<=K)}``
with ``K + m_- + n_- <= Lambda``.  By Theorem 4.6 this is **exactly zero** when
``omega = omega_can``, so

    R_raw = sqrt(sum_{(m,n) in P} ||Delta_{mn}||_F^2)
          / sqrt(sum_{(m,n) in P} ||T_{mn}||_F^2)

measures architecture-induced deviation and not truncation.  The truncation
(edge) remainder outside the filtered domain is a separate artifact (WP-G).

No Fock representation is built from a non-canonical pairing: doing so would
require choosing a polarization, which is exactly the unrestricted Darboux
freedom that Proposition 3.5 declares non-discriminating.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from math import factorial

import numpy as np

from .superalgebra import (
    Element,
    Key,
    Pairings,
    add,
    boson,
    bracket,
    fermion,
    multiply,
    normal_ordered_pair,
    scalar_part,
    scale,
    word_str,
)

__all__ = [
    "sugawara_L",
    "fermionic_L",
    "supercurrent_G",
    "identity_element",
    "virasoro_residual",
    "central_scalar",
    "k_max",
    "FockSpace",
    "residual_norm",
    "raw_virasoro_residual",
    "super_virasoro_residuals",
    "edge_remainder_terms",
]


def m_minus(m: int) -> int:
    """``m_- = max(0, -m)``, eq. (4.14)."""
    return max(0, -int(m))


def k_max(mode_cutoff: int, m: int, n: int) -> int:
    """``K_max(Lambda;m,n) = Lambda - m_- - n_-``, eq. (D.1)."""
    return int(mode_cutoff) - m_minus(m) - m_minus(n)


def identity_element(coeff: complex = 1.0) -> Element:
    return {(): complex(coeff)}


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------
def sugawara_L(m: int, mode_cutoff: int, pairings: Pairings,
               *, include_zero_mode: bool = True) -> Element:
    """Truncated bosonic Sugawara mode, eq. (4.13)."""
    m = int(m)
    lam = int(mode_cutoff)
    el: Element = {}
    for p in range(-lam, lam + 1):
        q = m - p
        if abs(q) > lam:
            continue
        if not include_zero_mode and (p == 0 or q == 0):
            continue
        word, sign = normal_ordered_pair(boson(q), boson(p))
        el[word] = el.get(word, 0.0 + 0.0j) + 0.5 * sign
    return {w: c for w, c in el.items() if c != 0}


def fermionic_L(n: int, fermion_cutoff: int, pairings: Pairings) -> Element:
    """``L_n^F = 1/4 sum_r (2r-n) :beta_{n-r} beta_r:`` (section 7).

    ``fermion_cutoff`` is the largest odd ``|2r|`` retained.
    """
    n = int(n)
    tmax = int(fermion_cutoff)
    el: Element = {}
    for t in range(-tmax, tmax + 1):
        if t % 2 == 0:
            continue
        t_other = 2 * n - t
        if abs(t_other) > tmax or t_other % 2 == 0:
            continue
        word, sign = normal_ordered_pair(fermion(t_other), fermion(t))
        el[word] = el.get(word, 0.0 + 0.0j) + 0.25 * (t - n) * sign
    return {w: c for w, c in el.items() if c != 0}


def supercurrent_G(two_r: int, mode_cutoff: int, fermion_cutoff: int,
                   pairings: Pairings) -> Element:
    """``G_r = sum_n alpha_n beta_{r-n}`` (section 7)."""
    tr = int(two_r)
    lam = int(mode_cutoff)
    tmax = int(fermion_cutoff)
    el: Element = {}
    for nn in range(-lam, lam + 1):
        t = tr - 2 * nn
        if abs(t) > tmax or t % 2 == 0:
            continue
        word = (boson(nn), fermion(t))
        el[word] = el.get(word, 0.0 + 0.0j) + 1.0
    return {w: c for w, c in el.items() if c != 0}


# --------------------------------------------------------------------------
# Residual elements
# --------------------------------------------------------------------------
def virasoro_residual(m: int, n: int, mode_cutoff: int, pairings: Pairings,
                      *, central_charge: float = 1.0,
                      include_zero_mode: bool = True) -> Element:
    """``Delta_{mn}`` of the module docstring."""
    lm = sugawara_L(m, mode_cutoff, pairings, include_zero_mode=include_zero_mode)
    ln = sugawara_L(n, mode_cutoff, pairings, include_zero_mode=include_zero_mode)
    comm = bracket(lm, ln, pairings)
    target = scale(
        sugawara_L(m + n, mode_cutoff, pairings, include_zero_mode=include_zero_mode),
        float(m - n),
    )
    if m + n == 0:
        target = add(
            target,
            identity_element(central_charge * m * (m * m - 1) / 12.0),
        )
    return add(comm, scale(target, -1.0))


def virasoro_target(m: int, n: int, mode_cutoff: int, pairings: Pairings,
                    *, central_charge: float = 1.0) -> Element:
    target = scale(sugawara_L(m + n, mode_cutoff, pairings), float(m - n))
    if m + n == 0:
        target = add(target, identity_element(central_charge * m * (m * m - 1) / 12.0))
    return target


def central_scalar(m: int, mode_cutoff: int, pairings: Pairings) -> complex:
    """``scalar([L_m, L_{-m}] - 2 m L_0)`` -- numerator of eq. (5.5)."""
    lm = sugawara_L(m, mode_cutoff, pairings)
    lmm = sugawara_L(-m, mode_cutoff, pairings)
    l0 = sugawara_L(0, mode_cutoff, pairings)
    el = add(bracket(lm, lmm, pairings), scale(l0, -2.0 * m))
    return scalar_part(el)


# --------------------------------------------------------------------------
# Filtered Fock representation (canonical readout)
# --------------------------------------------------------------------------
BosonState = tuple[int, ...]
FermionState = tuple[int, ...]
State = tuple[BosonState, FermionState]


@lru_cache(maxsize=None)
def _boson_states(mode_cutoff: int, twice_level: int) -> tuple[BosonState, ...]:
    """Multisets of parts in ``1..mode_cutoff`` with ``2*sum <= twice_level``."""
    limit = twice_level // 2
    out: list[BosonState] = []

    def rec(max_part: int, remaining: int, acc: tuple[int, ...]) -> None:
        out.append(acc)
        for p in range(min(max_part, remaining), 0, -1):
            rec(p, remaining - p, acc + (p,))

    rec(int(mode_cutoff), limit, ())
    return tuple(sorted({tuple(sorted(s)) for s in out}))


@lru_cache(maxsize=None)
def _fermion_states(fermion_cutoff: int, twice_level: int) -> tuple[FermionState, ...]:
    odds = [t for t in range(1, int(fermion_cutoff) + 1) if t % 2]
    out: list[FermionState] = []
    for size in range(len(odds) + 1):
        for combo in itertools.combinations(odds, size):
            if sum(combo) <= twice_level:
                out.append(tuple(sorted(combo)))
    return tuple(sorted(set(out)))


class FockSpace:
    """``F_a^{(<=K)}`` with the canonical CCR/CAR, used purely as a readout."""

    def __init__(self, level: int, mode_cutoff: int, charge: float = 0.0,
                 fermion_cutoff: int = 0):
        self.level = int(level)
        self.mode_cutoff = int(mode_cutoff)
        self.fermion_cutoff = int(fermion_cutoff)
        self.charge = float(charge)
        twice = 2 * self.level
        bstates = _boson_states(self.mode_cutoff, twice)
        fstates = (
            _fermion_states(self.fermion_cutoff, twice) if self.fermion_cutoff else ((),)
        )
        basis: list[State] = []
        for b in bstates:
            for f in fstates:
                if 2 * sum(b) + sum(f) <= twice:
                    basis.append((b, f))
        self.basis: tuple[State, ...] = tuple(basis)

    # -- norms -----------------------------------------------------------
    @staticmethod
    def norm2(state: State) -> float:
        """``<state|state>``: ``prod_p p^{n_p} n_p!`` for bosons, 1 for fermions."""
        b, _ = state
        out = 1.0
        for p in set(b):
            n = b.count(p)
            out *= (p**n) * factorial(n)
        return float(out)

    # -- single generator action -----------------------------------------
    def apply_key(self, key: Key, vec: dict[State, complex]) -> dict[State, complex]:
        out: dict[State, complex] = {}
        kind, idx = key
        for (b, f), c in vec.items():
            if kind == "a":
                if idx == 0:
                    if self.charge != 0.0:
                        out[(b, f)] = out.get((b, f), 0j) + c * self.charge
                    continue
                if idx < 0:
                    p = -idx
                    if p > self.mode_cutoff:
                        continue
                    nb = tuple(sorted(b + (p,)))
                    out[(nb, f)] = out.get((nb, f), 0j) + c
                else:
                    p = idx
                    mult = b.count(p)
                    if mult == 0:
                        continue
                    lst = list(b)
                    lst.remove(p)
                    nb = tuple(lst)
                    out[(nb, f)] = out.get((nb, f), 0j) + c * mult * p
            else:
                if idx < 0:
                    t = -idx
                    if t > self.fermion_cutoff or t in f:
                        continue
                    sign = (-1.0) ** sum(1 for s in f if s > t)
                    nf = tuple(sorted(f + (t,)))
                    out[(b, nf)] = out.get((b, nf), 0j) + c * sign
                else:
                    t = idx
                    if t not in f:
                        continue
                    sign = (-1.0) ** sum(1 for s in f if s > t)
                    nf = tuple(s for s in f if s != t)
                    out[(b, nf)] = out.get((b, nf), 0j) + c * sign
        return {k: v for k, v in out.items() if v != 0}

    def apply_element(self, element: Element,
                      vec: dict[State, complex]) -> dict[State, complex]:
        total: dict[State, complex] = {}
        for word, coeff in element.items():
            cur = dict(vec)
            for key in reversed(word):
                cur = self.apply_key(key, cur)
                if not cur:
                    break
            for st, c in cur.items():
                total[st] = total.get(st, 0j) + coeff * c
        return {k: v for k, v in total.items() if v != 0}

    def frobenius(self, element: Element) -> float:
        """``||element||_F`` w.r.t. the orthonormal Fock basis of the domain."""
        acc = 0.0
        for st in self.basis:
            img = self.apply_element(element, {st: 1.0 + 0.0j})
            n_in = self.norm2(st)
            for out_st, c in img.items():
                acc += (abs(c) ** 2) * self.norm2(out_st) / n_in
        return float(np.sqrt(acc))


# --------------------------------------------------------------------------
# Aggregated residual metrics
# --------------------------------------------------------------------------
def admissible_pairs(mode_cutoff: int, level: int, m_alg: int) -> list[tuple[int, int]]:
    """Pairs with ``1 <= |m|,|n| <= m_alg``, ``m != n``, ``K + m_- + n_- <= Lambda``."""
    pairs = []
    for m in range(-m_alg, m_alg + 1):
        if m == 0:
            continue
        for n in range(-m_alg, m_alg + 1):
            if n == 0 or m == n:
                continue
            if level + m_minus(m) + m_minus(n) <= mode_cutoff:
                pairs.append((m, n))
    return pairs


@dataclass
class ResidualReport:
    value: float
    numerator: float
    denominator: float
    pairs: list[tuple[int, int]]
    level: int
    charge: float


def raw_virasoro_residual(omega: np.ndarray, modes: np.ndarray, mode_cutoff: int,
                          *, level: int = 2, m_alg: int = 3, charge: float = 0.0,
                          central_charge: float = 1.0) -> ResidualReport:
    """``R_raw`` from a measured pairing (see module docstring)."""
    pair_meas = Pairings.from_matrices(modes=[int(m) for m in modes], omega=omega)
    pair_can = Pairings.canonical()
    pairs = admissible_pairs(mode_cutoff, level, m_alg)
    if not pairs:
        raise ValueError(
            f"no admissible (m,n) pairs for Lambda={mode_cutoff}, K={level}, "
            f"m_alg={m_alg}: the filtered bound K+m_-+n_- <= Lambda excludes all"
        )
    fock = FockSpace(level=level, mode_cutoff=mode_cutoff, charge=charge)
    num = 0.0
    den = 0.0
    for m, n in pairs:
        d = virasoro_residual(m, n, mode_cutoff, pair_meas,
                              central_charge=central_charge)
        t = virasoro_target(m, n, mode_cutoff, pair_can,
                            central_charge=central_charge)
        num += fock.frobenius(d) ** 2
        den += fock.frobenius(t) ** 2
    if den <= 0:  # pragma: no cover
        raise ValueError("degenerate Virasoro target norm")
    return ResidualReport(
        value=float(np.sqrt(num / den)), numerator=float(np.sqrt(num)),
        denominator=float(np.sqrt(den)), pairs=pairs, level=level, charge=charge,
    )


def super_virasoro_residuals(omega: np.ndarray, modes: np.ndarray,
                             eta: np.ndarray, fermion_modes: np.ndarray,
                             mode_cutoff: int, fermion_cutoff: int,
                             *, level: int = 2, r_alg: int = 3,
                             charge: float = 0.0) -> dict:
    r"""``[L_m,G_r]`` and ``{G_r,G_s}`` residuals plus the ``c`` fit (section 7).

    ``r_alg`` bounds ``|2r|``.  Uses ``L_n = L_n^B + L_n^F`` and ``c = 3/2``.
    """
    pair = Pairings.from_matrices(
        modes=[int(m) for m in modes], omega=omega,
        fermion_modes=[int(t) for t in fermion_modes], eta=eta,
    )
    fock = FockSpace(level=level, mode_cutoff=mode_cutoff, charge=charge,
                     fermion_cutoff=fermion_cutoff)

    def full_L(n: int) -> Element:
        return add(sugawara_L(n, mode_cutoff, pair),
                   fermionic_L(n, fermion_cutoff, pair))

    lg_num = lg_den = 0.0
    lg_pairs = []
    for m in (-1, 1, -2, 2):
        for tr in range(-r_alg, r_alg + 1):
            if tr % 2 == 0:
                continue
            tsum = tr + 2 * m
            if abs(tsum) > fermion_cutoff:
                continue
            lhs = bracket(full_L(m), supercurrent_G(tr, mode_cutoff, fermion_cutoff, pair),
                          pair)
            rhs = scale(supercurrent_G(tsum, mode_cutoff, fermion_cutoff, pair),
                        (m / 2.0) - tr / 2.0)
            lg_num += fock.frobenius(add(lhs, scale(rhs, -1.0))) ** 2
            lg_den += fock.frobenius(rhs) ** 2
            lg_pairs.append((m, tr))

    gg_num = gg_den = 0.0
    gg_pairs = []
    c_fits = []
    for ta in range(1, r_alg + 1, 2):
        for tb in (-ta, ta):
            ga = supercurrent_G(ta, mode_cutoff, fermion_cutoff, pair)
            gb = supercurrent_G(tb, mode_cutoff, fermion_cutoff, pair)
            lhs = bracket(ga, gb, pair, graded=True)
            n_sum = (ta + tb) // 2
            rhs = scale(full_L(n_sum), 2.0)
            r_sq = (ta / 2.0) ** 2
            if ta + tb == 0:
                rhs = add(rhs, identity_element(0.5 * (r_sq - 0.25)))  # (c/3), c=3/2
                if abs(r_sq - 0.25) > 1e-12:  # r = 1/2 carries no central term
                    c_scalar = scalar_part(add(lhs, scale(full_L(0), -2.0)))
                    c_fits.append(3.0 * complex(c_scalar).real / (r_sq - 0.25))
            gg_num += fock.frobenius(add(lhs, scale(rhs, -1.0))) ** 2
            gg_den += fock.frobenius(rhs) ** 2
            gg_pairs.append((ta, tb))

    return dict(
        LG_residual=float(np.sqrt(lg_num / lg_den)) if lg_den > 0 else float("nan"),
        GG_residual=float(np.sqrt(gg_num / gg_den)) if gg_den > 0 else float("nan"),
        central_charge_fit=float(np.mean(c_fits)) if c_fits else float("nan"),
        central_charge_fit_spread=float(np.std(c_fits)) if len(c_fits) > 1 else 0.0,
        LG_pairs=lg_pairs,
        GG_pairs=gg_pairs,
    )


# --------------------------------------------------------------------------
# Edge remainder (WP-G)
# --------------------------------------------------------------------------
def edge_remainder_terms(m: int, n: int, mode_cutoff: int) -> dict:
    """Full normal-ordered edge remainder of the *canonical* truncated algebra.

    Returns the monomials of ``Delta_{mn}(omega_can)``, each labelled with the
    reason it lies outside the filtered domain of Theorem 4.6.
    """
    pair = Pairings.canonical()
    el = virasoro_residual(m, n, mode_cutoff, pair)
    kmax = k_max(mode_cutoff, m, n)
    terms = []
    for word, coeff in sorted(el.items(), key=lambda kv: (-abs(kv[1]), word_str(kv[0]))):
        idxs = [int(k[1]) for k in word]
        creation_level = sum(-i for i in idxs if i < 0)
        if not word:
            reason = "scalar_term"
        elif any(abs(i) > kmax for i in idxs) and kmax >= 0:
            reason = (
                f"oscillator index exceeds K_max={kmax}: closure is guaranteed only "
                f"on F^(<=K) with K + m_- + n_- <= Lambda"
            )
        elif kmax < 0:
            reason = f"K_max={kmax} < 0: no guaranteed filtered domain at Lambda={mode_cutoff}"
        else:
            reason = (
                f"boundary summand: required partner index would exceed "
                f"Lambda={mode_cutoff}"
            )
        terms.append(
            dict(
                normal_ordered_monomial=word_str(word),
                coefficient=complex(coeff).real
                if abs(complex(coeff).imag) < 1e-14
                else complex(coeff),
                coefficient_abs=float(abs(coeff)),
                oscillator_indices=idxs,
                creation_level=int(creation_level),
                edge_support_reason=reason,
            )
        )
    coeffs = np.array([t["coefficient_abs"] for t in terms], dtype=float)
    return dict(
        m=int(m), n=int(n), mode_cutoff=int(mode_cutoff), k_max=int(kmax),
        n_terms=int(coeffs.size),
        coefficient_max=float(coeffs.max()) if coeffs.size else 0.0,
        coefficient_l1=float(coeffs.sum()) if coeffs.size else 0.0,
        coefficient_l2=float(np.sqrt((coeffs**2).sum())) if coeffs.size else 0.0,
        terms=terms,
    )
