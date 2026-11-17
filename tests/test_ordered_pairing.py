"""Ordered two-radius pairing: Propositions 3.1, 3.2, 3.4, 3.6 (mission WP-A1)."""
from __future__ import annotations

import numpy as np
import pytest

from nnft_virasoro import architectures as A
from nnft_virasoro import nonholomorphy as NH
from nnft_virasoro import ordered_pairing as OP

MODES = OP.mode_window(4)
R0, RIN, ROUT = OP.radii_from_window(0.0, 0.0, 2.0, r0=1.0)


def test_ideal_bosonic_ordered_pairing_prop_3_1():
    """I^> = m 1_{m>0} d, I^< = (-m) 1_{m<0} d, omega = m d_{m+n,0}."""
    res = OP.pairing_ideal_quadrature(MODES, RIN, ROUT, q=64, r0=R0)
    m = MODES
    ig_t = np.where((m[:, None] + m[None, :] == 0) & (m[:, None] > 0),
                    m[:, None].astype(float), 0.0)
    il_t = np.where((m[:, None] + m[None, :] == 0) & (m[:, None] < 0),
                    -m[:, None].astype(float), 0.0)
    assert np.abs(res.i_greater - ig_t).max() < 1e-12
    assert np.abs(res.i_less - il_t).max() < 1e-12
    assert np.abs(res.omega - OP.canonical_pairing(MODES)).max() < 1e-12


def test_pairing_antisymmetry():
    res = OP.pairing_ideal_quadrature(MODES, RIN, ROUT, q=64, r0=R0)
    assert np.abs(res.omega + res.omega.T).max() < 1e-12
    rng = np.random.default_rng(3)
    fs = A.sample_architecture("selected_log_kernel", rng, 800, 1e-2, 1e2)
    e = OP.pairing_empirical_exact(MODES, RIN, ROUT, R0, fs.kabs, fs.theta,
                                   fs.prefactor, 1e-2, 1e2, idx_a=fs.idx_a,
                                   idx_b=fs.idx_b, log_measure=fs.log_measure)
    met = OP.pairing_metrics(e.omega, MODES)
    assert met["antisymmetry_defect"] < 1e-12


@pytest.mark.parametrize("q", [32, 64, 128])
def test_angular_quadrature_convergence(q):
    """Uniform periodic quadrature converges spectrally on the ideal kernel."""
    res = OP.pairing_ideal_quadrature(MODES, RIN, ROUT, q=q, r0=R0)
    err = np.abs(res.omega - OP.canonical_pairing(MODES)).max()
    assert err < (1e-7 if q == 32 else 1e-12)


def test_local_residue_equivalence_under_valid_hypotheses():
    """Proposition 3.2 holds for the meromorphic ideal kernel."""
    lr = OP.local_residue_pairing(lambda z, w: 1.0 / (z - w) ** 2, MODES, 1.0,
                                  eps=0.2, q=256, r0=1.0)
    assert np.abs(lr - OP.canonical_pairing(MODES)).max() < 1e-12


def test_local_residue_fails_for_smooth_regulated_kernel():
    """The residue prescription is NOT interchangeable at finite cutoff."""
    from nnft_virasoro.kernels import current_kernel_analytic

    lir, luv = 1e-2, 1e2
    ordered = OP.pairing_analytic_spectral(MODES, RIN, ROUT, R0, lir, luv)
    local = OP.local_residue_pairing(
        lambda z, w: current_kernel_analytic(z, w, lir, luv), MODES, 1.0,
        eps=0.2, q=256, r0=1.0)
    disagreement = np.linalg.norm(ordered.omega - (local - local.T) / 1.0)
    assert disagreement > 1e-3, (
        "the two prescriptions must NOT agree for a smooth non-holomorphic kernel")


def test_empirical_exact_matches_general_quadrature():
    """Two independent implementations of the same functional."""
    lir, luv = 1e-2, 1e2
    rng = np.random.default_rng(11)
    fs = A.sample_architecture("selected_log_kernel", rng, 600, lir, luv)
    ex = OP.pairing_empirical_exact(MODES, RIN, ROUT, R0, fs.kabs, fs.theta,
                                    fs.prefactor, lir, luv, idx_a=fs.idx_a,
                                    idx_b=fs.idx_b, log_measure=fs.log_measure)
    qd = OP.pairing_quadrature(fs.current_kernel(), MODES, RIN, ROUT, q=256, r0=R0)
    rel = np.abs(ex.omega - qd.omega).max() / np.abs(ex.omega).max()
    assert rel < 1e-10


def test_non_translation_invariant_control_also_matches():
    lir, luv = 1e-2, 1e2
    rng = np.random.default_rng(12)
    fs = A.sample_architecture("phase_correlated_features", rng, 400, lir, luv)
    ex = OP.pairing_empirical_exact(MODES, RIN, ROUT, R0, fs.kabs, fs.theta,
                                    fs.prefactor, lir, luv, idx_a=fs.idx_a,
                                    idx_b=fs.idx_b, log_measure=fs.log_measure)
    qd = OP.pairing_quadrature(fs.current_kernel(), MODES, RIN, ROUT, q=512, r0=R0)
    rel = np.abs(ex.omega - qd.omega).max() / np.abs(ex.omega).max()
    assert rel < 1e-10


def test_analytic_pairing_is_structurally_antidiagonal():
    """Rotational invariance of p_Lambda kills every off-antidiagonal entry."""
    res = OP.pairing_analytic_spectral(MODES, RIN, ROUT, R0, 1e-2, 1e2)
    met = OP.pairing_metrics(res.omega, MODES)
    assert met["pairing_leakage"] == 0.0


def test_tapered_analytic_pairing_reaches_the_canonical_form():
    """The failure of eq. (2.2) is a sharp-cutoff artifact, not architectural."""
    res = OP.pairing_analytic_spectral(MODES, RIN, ROUT, R0, 1e-3, 1e3, taper=0.15)
    met = OP.pairing_metrics(res.omega, MODES)
    assert met["pairing_relative_error"] < 1e-2
    assert abs(met["amplitude_calibration"] - 1.0) < 1e-2


def test_r0_unit_normalization_is_r0_independent():
    """The reported pairing must not depend on the arbitrary contour scale."""
    a = OP.pairing_ideal_quadrature(MODES, 0.5 / np.sqrt(2), 0.5 * np.sqrt(2),
                                    q=64, r0=0.5)
    b = OP.pairing_ideal_quadrature(MODES, 4.0 / np.sqrt(2), 4.0 * np.sqrt(2),
                                    q=64, r0=4.0)
    assert np.abs(a.omega - b.omega).max() < 1e-10


def test_projection_proxy_is_symmetric_not_a_commutator():
    """Section 2.4 / mission rule 1: covariance is not a quantum commutator."""
    lir, luv = 1e-2, 1e2
    rng = np.random.default_rng(5)
    fs = A.sample_architecture("selected_log_kernel", rng, 500, lir, luv)
    cov = OP.projection_proxy_covariance(MODES, 1.0, 1.0, fs.kabs, fs.theta,
                                         fs.prefactor, lir, luv, idx_a=fs.idx_a,
                                         idx_b=fs.idx_b,
                                         log_measure=fs.log_measure)
    met = OP.projection_proxy_metrics(cov, MODES)
    assert met["projection_symmetry_defect"] < 1e-12
    assert met["projection_antisymmetrization_norm"] < 1e-10


def test_regulated_nonholomorphy_and_cauchy_green_bound():
    """Prop. 3.4 gives a nonzero dbar; prop. 3.6 turns it into a radius bound."""
    nh = NH.nonholomorphy_norms(RIN, ROUT, 1e-2, 1e2, q=64)
    assert nh["nonholomorphy_sup"] > 0
    b = NH.cauchy_green_variation_bound(MODES, (1.0, RIN, ROUT),
                                        (1.05, 1.05 * RIN, 1.05 * ROUT),
                                        1e-2, 1e2, n_radial=32, n_angle=48)
    assert b["cauchy_green_bound"] > 0
    zero = NH.cauchy_green_variation_bound(MODES, (1.0, RIN, ROUT),
                                           (1.0, RIN, ROUT), 1e-2, 1e2,
                                           n_radial=32, n_angle=48)
    assert zero["cauchy_green_bound"] == 0.0


def test_cauchy_green_bound_dominates_the_measured_variation():
    """The bound of prop. 3.6 must actually bound the analytic pairing change."""
    lir, luv = 1e-2, 1e2
    r0a, r0b = 1.0, 1.02
    a = OP.pairing_analytic_spectral(MODES, r0a / np.sqrt(2), r0a * np.sqrt(2),
                                     r0a, lir, luv)
    b = OP.pairing_analytic_spectral(MODES, r0b / np.sqrt(2), r0b * np.sqrt(2),
                                     r0b, lir, luv)
    measured = float(np.linalg.norm(b.omega - a.omega))
    bound = NH.cauchy_green_variation_bound(
        MODES, (r0a, r0a / np.sqrt(2), r0a * np.sqrt(2)),
        (r0b, r0b / np.sqrt(2), r0b * np.sqrt(2)), lir, luv,
        n_radial=128, n_angle=192)["cauchy_green_bound"]
    assert measured <= bound


# --------------------------------------------------------------------------
# taper must reach the non-holomorphy diagnostics
# --------------------------------------------------------------------------
def test_taper_reaches_the_nonholomorphy_diagnostics():
    """A tapered row must NOT inherit the sharp dbar norm.

    Regression guard: `nonholomorphy_norms` and `cauchy_green_variation_bound`
    once ignored the taper, so tapered rows reported the sharp non-holomorphy
    and were tested against a Cauchy-Green bound ~1e5 times too large.
    """
    ri, ro = 1 / np.sqrt(2), np.sqrt(2)
    sharp = NH.nonholomorphy_norms(ri, ro, 1e-3, 1e3, q=64, taper=0.0)
    tap = NH.nonholomorphy_norms(ri, ro, 1e-3, 1e3, q=64, taper=0.15)
    assert sharp["nonholomorphy_sup"] > 1e3
    assert tap["nonholomorphy_sup"] < 1.0
    assert tap["nonholomorphy_sup"] < 1e-3 * sharp["nonholomorphy_sup"]
    assert tap["nonholomorphy_taper"] == 0.15

    modes = OP.mode_window(6)
    a = (1.0, ri, ro)
    b = (1.02, 1.02 * ri, 1.02 * ro)
    bs = NH.cauchy_green_variation_bound(modes, a, b, 1e-3, 1e3, n_radial=32,
                                         n_angle=48, taper=0.0)
    bt = NH.cauchy_green_variation_bound(modes, a, b, 1e-3, 1e3, n_radial=32,
                                         n_angle=48, taper=0.15)
    assert bt["cauchy_green_bound"] < 1e-3 * bs["cauchy_green_bound"]
    assert bt["cauchy_green_taper"] == 0.15


def test_tapered_dbar_quadrature_is_converged():
    """The tapered dbar must not depend on the panel resolution.

    The default places two 8-node Gauss-Legendre panels per oscillation of
    J_n(kr); refining by 8x must not move the answer. (An earlier version used
    128 nodes per wavelength, which was 16x more expensive for no accuracy.)
    """
    r = np.array([0.5, 1.0, 2.0])
    ref = NH.dbar_magnitude(r, 1e-3, 1e3, 0.15, panels_per_wave=16)
    got = NH.dbar_magnitude(r, 1e-3, 1e3, 0.15)
    assert np.allclose(got, ref, rtol=3e-3), (got, ref)


def test_tapered_dbar_profile_is_oscillatory_not_smooth():
    """Guards the design: the taper cuts the amplitude, not the frequency.

    This is why the sup is taken on a grid that resolves the oscillation rather
    than by interpolating a coarse profile -- interpolation would miss peaks and
    could under-estimate a quantity that feeds an upper bound.
    """
    r = np.linspace(0.5, 1.0, 800)
    d = NH.dbar_magnitude(r, 1e-3, 1e3, 0.15)
    sign_changes = int((np.diff(np.sign(np.diff(d))) != 0).sum())
    assert sign_changes > 20, sign_changes


def test_dbar_magnitude_matches_the_scalar_tapered_formula():
    z, w = 1.0 + 0.3j, 0.2 - 0.1j
    scalar = abs(complex(NH.dbar_current_kernel_tapered(z, w, 1e-3, 1e3, 0.15)))
    vec = float(NH.dbar_magnitude(np.array([abs(z - w)]), 1e-3, 1e3, 0.15)[0])
    assert vec == pytest.approx(scalar, rel=1e-10)


def test_dbar_envelope_dominates_the_true_supremum():
    """The Cauchy-Green bound is only valid if the envelope never under-estimates.

    The tapered dbar oscillates, so its supremum on a window cannot be read off
    a coarse sample. The cached envelope takes a local maximum, then a
    cumulative maximum from the right (making it non-increasing), then applies a
    safety factor. This checks the resulting values against a brute-force sup on
    a grid that actually resolves the oscillation.
    """
    lir, luv, tp = 1e-3, 1e3, 0.15
    for a, b in [(0.35, 0.6), (0.5, 1.0), (1.0, 2.0), (2.121, 4.0)]:
        n = int(np.clip((b - a) * luv / (2 * np.pi) * 10, 3000, 25000))
        fine = float(NH.dbar_magnitude(np.linspace(a, b, n), lir, luv, tp).max())
        grid, env = NH._dbar_envelope(lir, luv, tp, NH._snap(a, np.floor),
                                      NH._snap(b, np.ceil))
        i = int(np.clip(np.searchsorted(grid, a, "left") - 1, 0, grid.size - 1))
        assert float(env[i]) >= fine, (a, b, float(env[i]), fine)


def test_dbar_envelope_is_non_increasing():
    """Non-increasing by construction, so a window sup is a single lookup."""
    grid, env = NH._dbar_envelope(1e-3, 1e3, 0.15, 0.3, 10.0)
    assert np.all(np.diff(env) <= 1e-30)
    assert grid.size == NH._ENVELOPE_NODES
