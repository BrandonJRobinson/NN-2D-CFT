r"""Graded ordered-word engine for the Heisenberg / Clifford mode algebra.

An element is a dict mapping an *ordered word* (tuple of generator keys) to a
complex coefficient.  Generator keys:

``("a", m)``   bosonic, ``m`` integer, even parity;
``("b", t)``   fermionic, ``t = 2r`` odd integer, odd parity.

Structure constants are supplied as pairings::

    [a_i, a_j] = omega[i, j]        (antisymmetric, central)
    {b_r, b_s} = eta[2r, 2s]        (symmetric, central)
    [a_i, b_r] = 0

Words are brought to the canonical order (bosons by ascending index, then
fermions by ascending ``2r``) by adjacent transpositions, each producing the
appropriate central contraction.  This is the ordinary super-Weyl normal form;
it is exact and independent of any representation, which is what lets the suite
build Sugawara generators from a **measured** pairing rather than from the
canonical one.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

__all__ = [
    "Key",
    "Element",
    "boson",
    "fermion",
    "Pairings",
    "normal_order",
    "multiply",
    "bracket",
    "scale",
    "add",
    "scalar_part",
    "normal_ordered_pair",
    "word_str",
]

Key = tuple[str, int]
Element = dict[tuple[Key, ...], complex]


def boson(m: int) -> Key:
    return ("a", int(m))


def fermion(two_r: int) -> Key:
    if two_r % 2 == 0:
        raise ValueError(f"NS fermion modes are half-integer; got 2r = {two_r}")
    return ("b", int(two_r))


def _order_key(k: Key) -> tuple[int, int]:
    return (0 if k[0] == "a" else 1, k[1])


def _parity(k: Key) -> int:
    return 0 if k[0] == "a" else 1


class Pairings:
    """Central structure constants for the graded algebra."""

    def __init__(self, omega: Callable[[int, int], complex] | None = None,
                 eta: Callable[[int, int], complex] | None = None):
        self._omega = omega
        self._eta = eta

    @classmethod
    def canonical(cls) -> "Pairings":
        """``[a_i,a_j] = i delta_{i+j,0}``, ``{b_r,b_s} = delta_{r+s,0}``."""
        return cls(
            omega=lambda i, j: complex(i) if i + j == 0 else 0.0 + 0.0j,
            eta=lambda s, t: 1.0 + 0.0j if s + t == 0 else 0.0 + 0.0j,
        )

    @classmethod
    def from_matrices(cls, modes: Sequence[int] | None = None,
                      omega: "object" = None,
                      fermion_modes: Sequence[int] | None = None,
                      eta: "object" = None) -> "Pairings":
        """Build from explicit matrices indexed by the given mode lists.

        Modes absent from ``modes`` / ``fermion_modes`` pair to zero -- this is
        how the zero mode ``alpha_0`` enters (it has vanishing pairing with the
        oscillators, eq. 4.9) and how the measured window ``M_Lambda^x`` is used.
        """
        omap = {int(m): i for i, m in enumerate(modes or ())}
        fmap = {int(t): i for i, t in enumerate(fermion_modes or ())}

        def om(i: int, j: int) -> complex:
            ii, jj = omap.get(i), omap.get(j)
            if ii is None or jj is None or omega is None:
                return 0.0 + 0.0j
            return complex(omega[ii, jj])

        def et(s: int, t: int) -> complex:
            ss, tt = fmap.get(s), fmap.get(t)
            if ss is None or tt is None or eta is None:
                return 0.0 + 0.0j
            return complex(eta[ss, tt])

        return cls(omega=om, eta=et)

    def contract(self, x: Key, y: Key) -> complex:
        """Central term of ``x y -/+ y x`` for adjacent generators."""
        if x[0] == "a" and y[0] == "a":
            return complex(self._omega(x[1], y[1])) if self._omega else 0.0 + 0.0j
        if x[0] == "b" and y[0] == "b":
            return complex(self._eta(x[1], y[1])) if self._eta else 0.0 + 0.0j
        return 0.0 + 0.0j


# --------------------------------------------------------------------------
# Word normal ordering
# --------------------------------------------------------------------------
def normal_order(word: Sequence[Key], pairings: Pairings,
                 coeff: complex = 1.0) -> Element:
    """Rewrite ``word`` in canonical order, emitting contraction terms."""
    out: Element = {}
    _normal_order_into(tuple(word), complex(coeff), pairings, out)
    return {w: c for w, c in out.items() if c != 0}


def _normal_order_into(word: tuple[Key, ...], coeff: complex, pairings: Pairings,
                       out: Element) -> None:
    if coeff == 0:
        return
    n = len(word)
    for i in range(n - 1):
        x, y = word[i], word[i + 1]
        px, py = _parity(x), _parity(y)
        if px == 1 and py == 1 and x == y:
            # b_r b_r = (1/2){b_r,b_r} = eta(r,r)
            c = pairings.contract(x, y)
            rest = word[:i] + word[i + 2:]
            _normal_order_into(rest, coeff * c, pairings, out)
            return
        if _order_key(x) > _order_key(y):
            rest = word[:i] + word[i + 2:]
            swapped = word[:i] + (y, x) + word[i + 2:]
            sign = -1.0 if (px == 1 and py == 1) else 1.0
            _normal_order_into(swapped, coeff * sign, pairings, out)
            if px == py:
                c = pairings.contract(x, y)
                if c != 0:
                    _normal_order_into(rest, coeff * c, pairings, out)
            return
    out[word] = out.get(word, 0.0 + 0.0j) + coeff


def normal_ordered_pair(x: Key, y: Key) -> tuple[tuple[Key, ...], float]:
    """``:x y:`` as (word, sign) -- the contraction-free reordering."""
    if _order_key(x) <= _order_key(y):
        return (x, y), 1.0
    sign = -1.0 if (_parity(x) == 1 and _parity(y) == 1) else 1.0
    return (y, x), sign


# --------------------------------------------------------------------------
# Element arithmetic
# --------------------------------------------------------------------------
def add(*elements: Element) -> Element:
    out: Element = {}
    for el in elements:
        for w, c in el.items():
            out[w] = out.get(w, 0.0 + 0.0j) + c
    return {w: c for w, c in out.items() if c != 0}


def scale(element: Element, factor: complex) -> Element:
    if factor == 0:
        return {}
    return {w: c * factor for w, c in element.items()}


def multiply(x: Element, y: Element, pairings: Pairings) -> Element:
    out: Element = {}
    for wx, cx in x.items():
        for wy, cy in y.items():
            _normal_order_into(wx + wy, cx * cy, pairings, out)
    return {w: c for w, c in out.items() if c != 0}


def bracket(x: Element, y: Element, pairings: Pairings,
            *, graded: bool = False) -> Element:
    """``x y - (-1)^{|x||y|} y x``.

    ``graded=True`` selects the anticommutator ``x y + y x``, used only for
    odd-odd pairs (``{G_r, G_s}``, eq. 7.10); the default is the commutator
    ``x y - y x``.
    """
    sign = 1.0 if graded else -1.0
    return add(multiply(x, y, pairings), scale(multiply(y, x, pairings), sign))


def scalar_part(element: Element) -> complex:
    return complex(element.get((), 0.0 + 0.0j))


def word_str(word: Sequence[Key]) -> str:
    parts = []
    for kind, idx in word:
        if kind == "a":
            parts.append(f"alpha[{idx}]")
        else:
            num = idx
            parts.append(f"beta[{num}/2]")
    return "*".join(parts) if parts else "1"
