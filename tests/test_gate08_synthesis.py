"""Gate 8: intrinsic symmetry diagnostics and the architecture synthesis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nnft_virasoro import architectures as AR
from nnft_virasoro import kernels as K
from nnft_virasoro import translation_invariance as TI
from nnft_virasoro.config import load_config, rng_for

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "jhep_submission_hardening"
CFG = ROOT / "configs" / "jhep_submission_hardening.yaml"
LIR, LUV, TAPER = 1e-2, 1e2, 0.15


def _load(name):
    p = RES / name
    if not p.exists():
        pytest.skip(f"{name} not generated yet")
    return pd.read_csv(p)


def _sample(arch, n, seed="controls_s01", tag="ti"):
    cfg = load_config(CFG)
    rng = rng_for(cfg.section("named_seeds"), seed, tag, arch, n)
    return AR.sample_architecture(arch, rng, n, LIR, LUV, params={
        "regulator_taper": TAPER, "anisotropy_amplitude": 0.6,
        "local_exponent_epsilon": 0.5})


# ------------------------------------------- two-argument kernel is correct ---
@pytest.mark.parametrize("arch", ["selected_log_kernel", "rbf_gaussian_prior",
                                  "angular_mode_scramble"])
def test_two_argument_kernels_reduce_to_the_one_argument_helpers(arch):
    """For idx_a == idx_b the general form must reproduce K(z-w) exactly."""
    fs = _sample(arch, 20000)
    assert np.array_equal(np.asarray(fs.idx_a), np.asarray(fs.idx_b))
    geo = TI.evaluation_geometry()
    z, w = geo["x"], geo["y"]
    kc = fs.kabs * np.exp(1j * fs.theta)
    assert np.allclose(TI.two_point_scalar(z, w, fs),
                       K.empirical_scalar_kernel(z, w, kc, LIR, LUV,
                                                 log_measure=fs.log_measure),
                       rtol=1e-12)
    assert np.allclose(TI.two_point_current(z, w, fs),
                       K.empirical_current_kernel(z, w, kc, LIR, LUV,
                                                  prefactor=fs.prefactor,
                                                  log_measure=fs.log_measure),
                       rtol=1e-12)


def test_phase_correlated_is_not_routed_through_the_one_argument_helper():
    """The whole point: its idx_a != idx_b, so K(z,w) != K(z-w)."""
    fs = _sample("phase_correlated_features", 20000)
    assert not np.array_equal(np.asarray(fs.idx_a), np.asarray(fs.idx_b))
    assert np.asarray(fs.idx_a).size == 2 * fs.kabs.size
    geo = TI.evaluation_geometry()
    z, w = geo["x"], geo["y"]
    kc = fs.kabs * np.exp(1j * fs.theta)
    true2 = TI.two_point_scalar(z, w, fs)
    naive = K.empirical_scalar_kernel(z, w, kc, LIR, LUV,
                                      log_measure=fs.log_measure)
    assert not np.allclose(true2, naive, rtol=1e-6), (
        "the one-argument helper would hide the defect")


# ------------------------------------------------------------ stationarity ---
def test_geometry_is_deterministic():
    """No RNG: a defect must never be confusable with sampling scatter."""
    a, b = TI.evaluation_geometry(), TI.evaluation_geometry()
    assert np.array_equal(a["x"], b["x"])
    assert np.array_equal(a["shifts"], b["shifts"])


@pytest.mark.parametrize("arch", ["selected_log_kernel", "rbf_gaussian_prior",
                                  "angular_mode_scramble",
                                  "anisotropic_angular_prior"])
def test_independent_phase_architectures_are_exactly_stationary(arch):
    fs = _sample(arch, 20000)
    geo = TI.evaluation_geometry()
    for kind in ("scalar", "current"):
        d = TI.translation_defect(fs, geo, kind=kind)["translation_defect"]
        assert d < 1e-12, f"{arch} {kind} defect {d}"


def test_phase_correlation_breaks_stationarity_and_does_not_converge():
    geo = TI.evaluation_geometry()
    vals = {}
    for n in (10000, 100000):
        fs = _sample("phase_correlated_features", n)
        vals[n] = TI.translation_defect(fs, geo, kind="scalar")["translation_defect"]
    assert min(vals.values()) > 0.1
    # structural, not statistical: flat across a 10x width range
    assert vals[100000] == pytest.approx(vals[10000], rel=0.15)


# ---------------------------------------------------------------- isotropy ---
def test_isotropy_defect_converges_for_isotropic_priors_and_pins_otherwise():
    geo = TI.evaluation_geometry()

    def d(arch, n):
        return TI.rotation_defect(_sample(arch, n, tag="rot"), geo,
                                  kind="scalar")["rotation_defect"]
    # isotropic prior: statistical, shrinks with width
    a, b = d("selected_log_kernel", 10000), d("selected_log_kernel", 100000)
    assert b < 0.5 * a
    # anisotropic prior: pinned
    a, b = d("anisotropic_angular_prior", 10000), d("anisotropic_angular_prior",
                                                    100000)
    assert b > 0.5 * a
    assert b > 0.02


def test_current_channel_rotation_uses_the_spin_two_weight():
    assert TI.SPIN_WEIGHT["scalar"] == 0
    assert TI.SPIN_WEIGHT["current"] == 2


# ------------------------------------------------------ generated synthesis ---
def test_selected_architecture_fails_no_axis():
    s = _load("architecture_synthesis.csv")
    sel = s[s.is_selected].iloc[0]
    assert sel.axes_failed == "none"
    assert bool(sel.passes_all_measured_axes)


def test_every_control_fails_the_axis_it_was_designed_to_violate():
    s = _load("architecture_synthesis.csv")
    ctl = s[~s.is_selected]
    assert len(ctl) == 7
    bad = ctl[ctl.fails_designed_axis != True]  # noqa: E712
    assert bad.empty, f"controls not failing their designed axis: " \
                      f"{list(bad.architecture)}"


def test_phase_correlated_is_isolated_only_by_the_new_axis():
    """Without the stationarity audit it would survive the whole suite."""
    s = _load("architecture_synthesis.csv").set_index("architecture")
    r = s.loc["phase_correlated_features"]
    assert r.axes_failed == "translation_invariance"
    assert not bool(r.axis_translation_invariance)
    # it genuinely passes every other structural axis
    assert bool(r.axis_physical_admissibility)
    assert bool(r.axis_pairing_fidelity)
    assert bool(r.axis_ward_behaviour)
    assert bool(r.axis_rotational_invariance)


def test_no_control_shares_the_selected_architectures_clean_sheet():
    s = _load("architecture_synthesis.csv")
    assert int(s.passes_all_measured_axes.sum()) == 1
    assert bool(s[s.passes_all_measured_axes].iloc[0].is_selected)


def test_pooled_exponent_is_consistent_with_minus_one_half():
    j = _load("convergence_joint_fit.csv")
    pooled = j[j.quantity == "POOLED_EXPONENT"]
    assert len(pooled) == 1
    r = pooled.iloc[0]
    assert float(r.exponent) == pytest.approx(-0.5, abs=0.05)
    assert float(r.null_half_p_value) > 0.05
    assert not bool(r.heterogeneity_detected)


def test_quantities_without_an_N_minus_half_prediction_are_not_pooled():
    j = _load("convergence_joint_fit.csv")
    for q in ("ward_stress_max_absolute_error", "scalar_translation_defect",
              "scalar_rotation_defect"):
        rows = j[j.quantity == q]
        assert len(rows) > 0
        assert not rows.pooled.any(), f"{q} must not be pooled"
        assert rows.pooling_reason.str.len().gt(10).all()
