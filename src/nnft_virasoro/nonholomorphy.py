r"""Finite-cutoff non-holomorphy and the Cauchy-Green radius-variation bound.

Proposition 3.4 (eqs. 3.14-3.15)::

    K_JJ^Lambda(z,w)      = (zbar-wbar)^2 k''(u)
    d_zbar K_JJ^Lambda    = (zbar-wbar) [2 k''(u) + u k'''(u)]

Using the closed forms of :mod:`nnft_virasoro.kernels`,

    2 k''(u) + u k'''(u) = [ b^2 J_2(b) - a^2 J_2(a) ] / (4 u^2),
    a = Lambda_IR r,  b = Lambda_UV r ,

which vanishes identically only in the conformal limit.  The ``b^2 J_2(b)`` term
grows like ``b^{3/2}``: the sharp annular prior of eq. (2.2) makes the regulated
current kernel *strongly* non-holomorphic, and Proposition 3.6 then converts
that into a large predicted radius dependence of the extracted pairing.  The
audit measures both sides of that inequality.

Proposition 3.6 (eq. 3.22), specialized to a rotationally invariant kernel::

    |I^>_{mn}(R_in, R_out') - I^>_{mn}(R_in, R_out)|
        <= (R_in^{n+1}/pi) * 2 pi int_{R_out}^{R_out'} s^{m+1} G_{R_in}(s) ds
    G_{R_w}(s) = sup_{|w|=R_w} |d_zbar K_JJ(s e^{i.}, w)|

All bounds are expressed in the same ``R_0`` units as the reported pairing
(divided by ``R_0^{m+n}``) so they are directly comparable to the measured
variation.
"""
from __future__ import annotations

import functools

import numpy as np

from .kernels import dbar_current_kernel_analytic

#: Nodes in the cached radial envelope of |d_zbar K_JJ| (log-spaced).
_ENVELOPE_NODES = 384
#: Neighbourhood (in nodes) over which the local maximum is taken.
_ENVELOPE_WINDOW = 7
#: Multiplicative safety factor covering peaks between sampled nodes.
_ENVELOPE_SAFETY = 2.0


@functools.lru_cache(maxsize=64)
def _dbar_envelope(lambda_ir: float, lambda_uv: float, taper: float, lo: float,
                   hi: float):
    r"""Non-increasing upper envelope of ``|d_zbar K_JJ|`` on ``[lo, hi]``.

    Two facts make this both cheap and safe:

    * ``|d_zbar K_JJ|`` decays fast with separation (measured: ``~r^-3.5`` for the
      tapered regulator, from ``4e-1`` at ``r=0.35`` to ``9e-8`` at ``r=30``), so
      the expensive large-``r`` region contributes essentially nothing;
    * ``sup_{|w|=R_w} |d_zbar K_JJ(s,w)|`` is a supremum of that 1-D function over
      ``r`` in ``[|s-R_w|, s+R_w]``, and a *non-increasing* envelope makes that
      supremum a single lookup at the left endpoint.

    The envelope is a local maximum over ``_ENVELOPE_WINDOW`` nodes, then a
    cumulative maximum taken from the right (so it is non-increasing by
    construction), then inflated by ``_ENVELOPE_SAFETY`` to cover oscillation
    peaks falling between sampled nodes.  It is therefore a genuine *upper*
    bound, which is what Proposition 3.6 requires.

    Evaluating the exact oscillation on a grid fine enough to resolve it costs
    ``O((Lambda_UV r_max)^2)`` -- 25 minutes per call at ``R_0 = 21`` -- which is
    what made an earlier version of this audit intractable.
    """
    grid = np.geomspace(max(lo, 1e-12), hi, _ENVELOPE_NODES)
    vals = np.empty_like(grid)
    # per-octave k grids: the panel width is set by the local r, not by r_max,
    # so small separations are not resolved at the cost of the largest one.
    edges = np.unique(np.clip(
        np.searchsorted(grid, grid[0] * 2.0 ** np.arange(0, 64)), 0, grid.size))
    bounds = np.unique(np.concatenate([[0], edges, [grid.size]]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            vals[a:b] = dbar_magnitude(grid[a:b], lambda_ir, lambda_uv, taper,
                                       panels_per_wave=2.0)
    win = _ENVELOPE_WINDOW
    pad = win // 2
    local = np.max(np.lib.stride_tricks.sliding_window_view(
        np.pad(vals, pad, mode="edge"), win), axis=-1)
    env = np.maximum.accumulate(local[::-1])[::-1] * _ENVELOPE_SAFETY
    grid.flags.writeable = False
    env.flags.writeable = False
    return grid, env


def _snap(x: float, rule) -> float:
    """Round a radius outward on a coarse log lattice so calls share a cache entry."""
    return float(np.exp(rule(np.log(max(x, 1e-12)) * 4.0) / 4.0))


__all__ = ["nonholomorphy_norms", "dbar_sup_profile", "cauchy_green_variation_bound",
           "dbar_current_kernel_tapered", "dbar_magnitude"]


def dbar_current_kernel_tapered(z, w, lambda_ir: float, lambda_uv: float,
                                taper: float, *, panels_per_wave: float = 2.0):
    r"""``d_zbar K_JJ`` for a tapered spectral density ``w(k)``.

    From ``d^n/du^n J_0(k sqrt u) = (-1/2)^n k^n r^{-n} J_n(k r)``,

        2 k''_w(u) + u k'''_w(u)
            = (1/r^2) int dk k w(k) J_2(k r)
              - (1/(4 r)) int dk k^2 w(k) J_3(k r).

    With a smooth taper the two terms cancel to the accuracy of the window; with
    the sharp annulus of eq. (2.2) they do not, and the residual grows like
    ``(Lambda_UV r)^{3/2}``.  That contrast is the quantitative statement that
    the non-holomorphy of prop. 3.4 is a sharp-cutoff artifact.
    """
    from scipy.special import jv

    from .architectures import _tukey_log_window
    from .ordered_pairing import _gauss_legendre_panels

    xi = complex(np.asarray(z)) - complex(np.asarray(w))
    r = abs(xi)
    # same panel convention as `dbar_magnitude`, so the two agree exactly
    pts, wts = _gauss_legendre_panels(
        lambda_ir, lambda_uv,
        2.0 * np.pi / (panels_per_wave * max(r, 1e-12)), nodes=8)
    wgt = (_tukey_log_window(np.log(pts), np.log(lambda_ir), np.log(lambda_uv),
                             taper) if taper > 0 else np.ones_like(pts))
    t2 = float(np.sum(wts * pts * wgt * jv(2, pts * r))) / r**2
    t3 = float(np.sum(wts * pts**2 * wgt * jv(3, pts * r))) / (4.0 * r)
    return np.conj(xi) * (t2 - t3)


def dbar_magnitude(sep, lambda_ir: float, lambda_uv: float, taper: float = 0.0,
                   *, panels_per_wave: float = 2.0, chunk: int = 512,
                   _direct: bool = False) -> np.ndarray:
    r"""``|d_zbar K_JJ|`` as a function of the separation ``|z-w|``.

    The non-holomorphy depends on ``z`` and ``w`` only through ``r = |z-w|``, so
    one radial profile suffices.  The sharp annulus uses the closed form; a
    tapered density uses the two quadratures of
    :func:`dbar_current_kernel_tapered`, evaluated on a shared ``k`` grid and
    vectorized over ``r``.

    Threading the taper through here is load-bearing: without it every tapered
    row inherits the *sharp* dbar norm, which also inflates the Cauchy--Green
    bound and would make the tapered radius-plateau test spuriously permissive.
    """
    from scipy.special import jv

    from .architectures import _tukey_log_window
    from .ordered_pairing import _gauss_legendre_panels

    r = np.asarray(sep, dtype=float)
    flat = r.reshape(-1)
    if taper <= 0:
        return np.abs(
            dbar_current_kernel_analytic(flat.astype(complex), 0j,
                                         lambda_ir, lambda_uv)).reshape(r.shape)
    # Exact per-point quadrature -- no interpolation. The tapered profile still
    # OSCILLATES in r with wavelength ~2 pi/Lambda_UV (the taper cuts the
    # amplitude by orders of magnitude, not the frequency), so interpolating it
    # on any affordable grid would both miss peaks and risk *under*-estimating a
    # quantity that feeds an upper bound.
    #
    # `panels_per_wave` panels of 8 Gauss-Legendre nodes each per oscillation of
    # J_n(k r) is ample; the previous width of pi/(8 r) placed 128 nodes per
    # wavelength and made this path ~16x more expensive than it needs to be.
    rmax = float(np.max(flat)) if flat.size else 1.0
    pts, wts = _gauss_legendre_panels(
        lambda_ir, lambda_uv,
        2.0 * np.pi / (panels_per_wave * max(rmax, 1e-12)), nodes=8)
    wgt = _tukey_log_window(np.log(pts), np.log(lambda_ir), np.log(lambda_uv),
                            taper)
    w2 = wts * pts * wgt
    w3 = wts * pts**2 * wgt
    out = np.empty_like(flat)
    for i in range(0, flat.size, chunk):
        rr = flat[i:i + chunk]
        kr = pts[None, :] * rr[:, None]
        t2 = (jv(2, kr) * w2[None, :]).sum(axis=1) / rr**2
        t3 = (jv(3, kr) * w3[None, :]).sum(axis=1) / (4.0 * rr)
        out[i:i + chunk] = np.abs(rr * (t2 - t3))
    return out.reshape(r.shape)


def nonholomorphy_norms(r_in: float, r_out: float, lambda_ir: float,
                        lambda_uv: float, *, q: int = 256,
                        taper: float = 0.0) -> dict:
    """Sup and normalized L2 norms of ``d_zbar K_JJ`` on the two contour tori."""
    theta = 2.0 * np.pi * np.arange(q) / q
    out = {}
    sup_all = 0.0
    l2_all = 0.0
    for tag, (rz, rw) in {"oi": (r_out, r_in), "io": (r_in, r_out)}.items():
        zs = rz * np.exp(1j * theta)[:, None]
        ws = rw * np.exp(1j * theta)[None, :]
        # rotational symmetry: |z_p - w_q| depends only on (p-q) mod q, so the
        # q^2 torus points carry only q distinct separations
        seps = np.abs(zs - ws)
        uniq, inv = np.unique(np.round(seps.ravel(), 12), return_inverse=True)
        vals = dbar_magnitude(uniq, lambda_ir, lambda_uv, taper)[inv].reshape(
            seps.shape)
        out[f"nonholomorphy_sup_{tag}"] = float(vals.max())
        out[f"nonholomorphy_l2_{tag}"] = float(np.sqrt((vals**2).mean()))
        sup_all = max(sup_all, float(vals.max()))
        l2_all = max(l2_all, float(np.sqrt((vals**2).mean())))
    out["nonholomorphy_sup"] = sup_all
    out["nonholomorphy_l2"] = l2_all
    out["nonholomorphy_quadrature_points"] = int(q)
    out["nonholomorphy_taper"] = float(taper)
    return out


def dbar_sup_profile(radii: np.ndarray, r_w: float, lambda_ir: float,
                     lambda_uv: float, *, n_angle: int = 192,
                     taper: float = 0.0) -> np.ndarray:
    """``G_{R_w}(s) = sup_{|w|=R_w} |d_zbar K_JJ(s, w)|`` for each ``s`` in ``radii``.

    ``|d_zbar K_JJ|`` depends on ``z`` and ``w`` only through ``|z-w|``, so the
    separations are deduplicated before the (expensive) tapered quadrature and
    mapped back afterwards.
    """
    s = np.asarray(radii, dtype=float)
    if taper <= 0:
        phi = 2.0 * np.pi * np.arange(n_angle) / n_angle
        ws = r_w * np.exp(1j * phi)[None, :]
        vals = dbar_magnitude(np.abs(s[:, None].astype(complex) - ws),
                              lambda_ir, lambda_uv, taper)
        return vals.max(axis=1)
    # tapered: one cached non-increasing envelope makes each sup a lookup at the
    # smallest separation the w-contour can reach.
    lo = max(float(np.min(np.abs(s - r_w))), 1e-12)
    hi = float(np.max(s + r_w))
    grid, env = _dbar_envelope(float(lambda_ir), float(lambda_uv), float(taper),
                               _snap(lo, np.floor), _snap(hi, np.ceil))
    left = np.maximum(np.abs(s - r_w), grid[0])
    idx = np.clip(np.searchsorted(grid, left, "left") - 1, 0, grid.size - 1)
    return env[idx]


def _radial_term(modes: np.ndarray, s_lo: float, s_hi: float, r_fixed: float,
                 lambda_ir: float, lambda_uv: float, *, n_radial: int,
                 n_angle: int, taper: float = 0.0) -> np.ndarray:
    """``2 int_{s_lo}^{s_hi} s^{m+1} G_{r_fixed}(s) ds`` for each mode ``m``."""
    if s_hi <= s_lo:
        return np.zeros(modes.size)
    s = np.linspace(s_lo, s_hi, int(n_radial))
    g = dbar_sup_profile(s, r_fixed, lambda_ir, lambda_uv, n_angle=n_angle,
                         taper=taper)
    out = np.empty(modes.size)
    for i, m in enumerate(modes):
        out[i] = 2.0 * float(np.trapz(s ** (int(m) + 1) * g, s))
    return out


def cauchy_green_variation_bound(modes: np.ndarray, radii_a: tuple[float, float, float],
                                 radii_b: tuple[float, float, float],
                                 lambda_ir: float, lambda_uv: float, *,
                                 n_radial: int = 96, n_angle: int = 192,
                                 taper: float = 0.0) -> dict:
    """Total Prop. 3.6 bound on ``||omega(b) - omega(a)||`` between two radius pairs.

    ``radii_*`` are ``(r0, r_in, r_out)``.  The four contributions (outer/inner
    contour, both orderings) are summed entrywise, then reported both as a
    Frobenius-style aggregate and restricted to the antidiagonal (the plateau
    metric).  Reported in ``R_0`` units of ``radii_a``.
    """
    modes = np.asarray(modes, dtype=int)
    r0a, ria, roa = radii_a
    r0b, rib, rob = radii_b
    lo_out, hi_out = sorted((roa, rob))
    lo_in, hi_in = sorted((ria, rib))

    # vary the outer contour (z carries m for I^>, n for I^<)
    t_out = _radial_term(modes, lo_out, hi_out, min(ria, rib), lambda_ir, lambda_uv,
                         n_radial=n_radial, n_angle=n_angle, taper=taper)
    # vary the inner contour
    t_in = _radial_term(modes, lo_in, hi_in, max(roa, rob), lambda_ir, lambda_uv,
                        n_radial=n_radial, n_angle=n_angle, taper=taper)

    m = modes
    ri = min(ria, rib)
    ro = max(roa, rob)
    with np.errstate(over="ignore"):
        # I^> : z on the outer contour with weight m, w inner with weight n
        b_g = (np.abs(ri) ** (m[None, :] + 1)) * t_out[:, None] + (
            np.abs(ro) ** (m[:, None] + 1)
        ) * t_in[None, :]
        # I^< : roles exchanged
        b_l = (np.abs(ro) ** (m[None, :] + 1)) * t_in[:, None] + (
            np.abs(ri) ** (m[:, None] + 1)
        ) * t_out[None, :]
        total = b_g + b_l
        norm = r0a ** (-(m[:, None] + m[None, :]))
        total = total * norm

    diag_mask = (m[:, None] + m[None, :]) == 0
    anti = total[diag_mask]
    anti_target = np.abs(m.astype(float))
    return dict(
        cauchy_green_bound=float(np.sqrt(np.sum(total**2))),
        cauchy_green_bound_antidiagonal=float(
            np.sqrt(np.sum(anti**2)) / np.linalg.norm(anti_target)
        ),
        cauchy_green_radial_nodes=int(n_radial),
        cauchy_green_angular_nodes=int(n_angle),
        cauchy_green_taper=float(taper),
    )
