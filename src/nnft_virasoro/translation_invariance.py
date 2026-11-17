r"""Intrinsic symmetry diagnostics: stationarity and isotropy (Gate 8).

Every other structural axis in the campaign -- ordered pairing fidelity, raw
Virasoro stability, Ward behaviour, intrinsic physical admissibility -- is built
from contour integrals or from the pairing matrix, and each of those is a
functional of the kernel that happens to be **blind to loss of translation
invariance**. A control with the correct radial spectrum and the correct
isotropic angular law, differing only by phase correlation between features,
therefore passes all of them.

This module measures the missing symmetries directly. Two are audited, because
the control set is designed to break two different ones:

    delta_TI = [ sum_{g,a} |K(x_g + a, y_g + a) - K(x_g, y_g)|^2
                 / sum_{g,a} |K(x_g, y_g)|^2 ]^{1/2}

evaluated on the **true two-argument kernel**. That matters: the one-argument
helpers in :mod:`nnft_virasoro.kernels` take only ``xi = z - w`` and so are
translation invariant *by construction*. Routing a phase-correlated
architecture through them would silently return zero defect. Here the sum runs
over the architecture's own ``idx_a`` / ``idx_b`` pairing:

    K_phi(z,w)  = (2 L_w / N) sum_m cos( Re(conj(k_{a_m}) z) - Re(conj(k_{b_m}) w) )
    K_JJ(z,w)   = -(L_w / 2N) sum_m P_m cos( Re(conj(k_{a_m}) z) - Re(conj(k_{b_m}) w) )

with ``P_m = conj(k_{a_m}) conj(k_{b_m})`` (the sample's ``prefactor``). For an
independent-phase architecture ``idx_a = idx_b``, the phase difference collapses
to ``Re(conj(k)(z-w))``, these reduce exactly to the one-argument forms, and the
defect is zero up to floating point. For a phase-correlated architecture the
cross terms survive and the defect is O(1) at every width.

**Isotropy.** The companion defect under a rigid rotation,

    delta_RI = [ sum_{g,phi} |e^{i s phi} K(e^{i phi} x_g, e^{i phi} y_g) - K(x_g, y_g)|^2
                 / sum_{g,phi} |K(x_g, y_g)|^2 ]^{1/2}

with spin weight ``s = 0`` for the scalar channel and ``s = 2`` for the current
channel (the same spin-two weight Gate 6 pins in check C7). Unlike stationarity,
isotropy is only a *statistical* property of an isotropic momentum prior: the
empirical angular distribution of N draws is anisotropic at order N^{-1/2}, so a
rotationally invariant architecture has a defect that **shrinks** with width
rather than vanishing identically. A control with a genuinely anisotropic prior
has a defect that stays pinned. The distinction between the two diagnostics is
therefore real and deliberate: stationarity is exact or broken, isotropy is
converging or pinned.
"""
from __future__ import annotations

import numpy as np

from .kernels import log_ratio

__all__ = [
    "evaluation_geometry",
    "two_point_scalar",
    "two_point_current",
    "translation_defect",
    "rotation_defect",
    "SPIN_WEIGHT",
]

#: Spin weight of each channel under a rigid rotation ``z -> e^{i phi} z``.
SPIN_WEIGHT = {"scalar": 0, "current": 2}


def evaluation_geometry(n_separations: int = 6, n_shifts: int = 16,
                        max_separation: float = 0.7,
                        shift_radius: float = 0.8) -> dict:
    """A fixed, deterministic set of point pairs and rigid translations.

    Deterministic by construction (no RNG): the diagnostic must not depend on a
    seed, so that a defect can never be confused with sampling scatter.
    """
    g = np.arange(n_separations)
    rad = max_separation * (0.25 + 0.75 * g / max(n_separations - 1, 1))
    ang = 2.0 * np.pi * g * (np.sqrt(5.0) - 1.0) / 2.0        # golden angle
    xi = rad * np.exp(1j * ang)
    a = np.arange(n_shifts)
    srad = shift_radius * np.sqrt((a + 0.5) / n_shifts)
    sang = 2.0 * np.pi * a * (np.sqrt(5.0) - 1.0) / 2.0
    shifts = srad * np.exp(1j * sang)
    base = 0.0 + 0.0j
    return dict(separations=xi, shifts=shifts, base=base,
                x=base + xi, y=np.full(xi.shape, base))


def _kernel_sum(z, w, kc, idx_a, idx_b, weights, chunk: int):
    """``sum_m weights_m cos(Re(conj(k_{a_m}) z) - Re(conj(k_{b_m}) w))``."""
    z = np.asarray(z, dtype=complex).ravel()
    w = np.asarray(w, dtype=complex).ravel()
    acc = np.zeros(z.shape, dtype=(np.complex128 if np.iscomplexobj(weights)
                                   else np.float64))
    m = idx_a.size
    for start in range(0, m, chunk):
        sl = slice(start, start + chunk)
        ka = np.conj(kc[idx_a[sl]])
        kb = np.conj(kc[idx_b[sl]])
        ph = (np.real(ka[None, :] * z[:, None])
              - np.real(kb[None, :] * w[:, None]))
        if weights is None:
            acc += np.cos(ph).sum(axis=1)
        else:
            acc += (np.cos(ph) * weights[sl][None, :]).sum(axis=1)
    return acc


def two_point_scalar(z, w, fs, *, chunk: int = 4096) -> np.ndarray:
    """``K_phi^(N)(z, w)`` honouring the architecture's own phase pairing."""
    kc = fs.kabs * np.exp(1j * fs.theta)
    n = fs.kabs.size
    lam = fs.log_measure or log_ratio(fs.lambda_ir, fs.lambda_uv)
    acc = _kernel_sum(z, w, kc, np.asarray(fs.idx_a), np.asarray(fs.idx_b),
                      None, chunk)
    return (2.0 * lam / n) * acc


def two_point_current(z, w, fs, *, chunk: int = 4096) -> np.ndarray:
    """``K_JJ^(N)(z, w)`` honouring the architecture's own phase pairing."""
    kc = fs.kabs * np.exp(1j * fs.theta)
    n = fs.kabs.size
    lam = fs.log_measure or log_ratio(fs.lambda_ir, fs.lambda_uv)
    pref = np.asarray(fs.prefactor)
    acc = _kernel_sum(z, w, kc, np.asarray(fs.idx_a), np.asarray(fs.idx_b),
                      pref, chunk)
    return -(lam / (2.0 * n)) * acc


def translation_defect(fs, geometry: dict, *, kind: str = "scalar",
                       chunk: int = 4096) -> dict:
    """``delta_TI`` for one feature sample.

    Returns the pooled defect plus the pieces needed to see whether a nonzero
    value is structural or numerical: the unshifted kernel scale, the worst
    single shift, and the number of evaluations.
    """
    fn = {"scalar": two_point_scalar, "current": two_point_current}[kind]
    x, y = geometry["x"], geometry["y"]
    shifts = geometry["shifts"]
    k0 = np.asarray(fn(x, y, fs, chunk=chunk)).ravel()
    num = 0.0
    den = 0.0
    worst = 0.0
    for a in shifts:
        ka = np.asarray(fn(x + a, y + a, fs, chunk=chunk)).ravel()
        d = np.abs(ka - k0) ** 2
        num += float(d.sum())
        den += float(np.sum(np.abs(k0) ** 2))
        s = float(np.sqrt(d.sum() / max(np.sum(np.abs(k0) ** 2), 1e-300)))
        worst = max(worst, s)
    defect = float(np.sqrt(num / den)) if den > 0 else float("nan")
    return dict(translation_defect=defect,
                translation_defect_worst_shift=worst,
                kernel_rms=float(np.sqrt(np.mean(np.abs(k0) ** 2))),
                n_evaluations=int(x.size * shifts.size),
                n_separations=int(x.size), n_shifts=int(shifts.size),
                kind=kind)


def rotation_defect(fs, geometry: dict, *, kind: str = "scalar",
                    n_angles: int = 12, chunk: int = 4096) -> dict:
    """``delta_RI`` for one feature sample, with the channel's spin weight.

    Reported alongside :func:`translation_defect` but read differently: an
    isotropic prior gives a defect of order ``N^{-1/2}``, not zero, so the
    verdict is "converging" rather than "vanishing" (see the module docstring).
    """
    fn = {"scalar": two_point_scalar, "current": two_point_current}[kind]
    spin = SPIN_WEIGHT[kind]
    x, y = geometry["x"], geometry["y"]
    phis = 2.0 * np.pi * np.arange(int(n_angles)) / int(n_angles)
    k0 = np.asarray(fn(x, y, fs, chunk=chunk)).ravel()
    base = float(np.sum(np.abs(k0) ** 2))
    num = 0.0
    den = 0.0
    worst = 0.0
    for ph in phis:
        rot = np.asarray(fn(x * np.exp(1j * ph), y * np.exp(1j * ph), fs,
                            chunk=chunk)).ravel() * np.exp(1j * spin * ph)
        d = float(np.sum(np.abs(rot - k0) ** 2))
        num += d
        den += base
        worst = max(worst, float(np.sqrt(d / max(base, 1e-300))))
    return dict(rotation_defect=float(np.sqrt(num / den)) if den > 0 else np.nan,
                rotation_defect_worst_angle=worst, n_angles=int(n_angles),
                spin_weight=spin, kind=kind)
