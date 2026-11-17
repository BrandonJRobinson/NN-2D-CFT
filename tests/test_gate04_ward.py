"""Gates 4+5: normalized Ward metric, paired ratios, analytic targets."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nnft_virasoro import ward as WD

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RES = ROOT / "results" / "jhep_submission_hardening"
FROZEN = ROOT / "results" / "jhep_frozen"


def _load(name):
    p = RES / name
    if not p.exists():
        pytest.skip(f"{name} not generated yet")
    return pd.read_csv(p)


# ---------------------------------------------------------------- targets ---
def test_analytic_stress_target_matches_two_vertex_closed_form():
    """T(z) = sum_i [h_i/(z-w_i)^2 + (d_i log)/(z-w_i)] for a neutral pair."""
    a = np.array([0.3, -0.3])
    w = np.array([0.85 + 0j, -0.85 + 0j])
    z = np.array([0.0 + 1.5j, 1.7 + 0.3j])
    got = WD.stress_target(z, a, w)
    h = 0.5 * a**2
    d01 = a[0] * a[1] / (w[0] - w[1])
    d10 = a[1] * a[0] / (w[1] - w[0])
    want = (h[0] / (z - w[0]) ** 2 + d01 / (z - w[0])
            + h[1] / (z - w[1]) ** 2 + d10 / (z - w[1]))
    assert got == pytest.approx(want, rel=1e-12)


def test_current_target_is_the_neutral_charge_sum():
    a = np.array([0.3, 0.3, -0.6])
    w = np.array([0.8 + 0j, -0.4 + 0.7j, -0.4 - 0.7j])
    z = np.array([1.7 + 0.3j])
    assert WD.current_target(z, a, w) == pytest.approx(
        np.sum(a / (z[:, None] - w), axis=-1), rel=1e-12)


def test_ward_targets_require_neutrality():
    with pytest.raises(Exception):
        WD.check_neutral(np.array([0.3, -0.2]))
    WD.check_neutral(np.array([0.3, -0.3]))       # must not raise


# ------------------------------------------------------ normalized metric ---
def test_target_magnitude_reconstruction_is_exact():
    """|T_tgt| = abs_err / rel_err, and NOT |estimate - target|."""
    import run_gate04_ward as G4
    d = pd.read_csv(FROZEN / "sampled_ward_seed_rows.csv")
    rec = G4.target_magnitude(d)
    assert np.allclose(rec.to_numpy(float), d.stress_target.to_numpy(float),
                       rtol=1e-9)
    # the naive difference does not reproduce the stored error
    naive = (d.stress_estimate - d.stress_target).abs()
    assert not np.allclose(naive, d.stress_absolute_error, rtol=1e-6)


def test_e_norm_is_scale_covariant_and_vanishes_for_exact_agreement():
    import run_gate04_ward as G4
    g = pd.DataFrame({"stress_absolute_error": [0.0, 0.0, 0.0],
                      "_tgt": [1.0, 2.0, 3.0]})
    assert G4.e_norm(g) == 0.0
    # scaling errors by c scales E_norm by c
    g2 = pd.DataFrame({"stress_absolute_error": [0.1, 0.2, 0.3],
                       "_tgt": [1.0, 2.0, 3.0]})
    g3 = g2.assign(stress_absolute_error=g2.stress_absolute_error * 7.0)
    assert G4.e_norm(g3) == pytest.approx(7.0 * G4.e_norm(g2), rel=1e-12)
    # scaling BOTH leaves it invariant
    g4 = g2.assign(stress_absolute_error=g2.stress_absolute_error * 5.0,
                   _tgt=g2._tgt * 5.0)
    assert G4.e_norm(g4) == pytest.approx(G4.e_norm(g2), rel=1e-12)


def test_e_norm_equals_relative_error_for_a_single_probe():
    import run_gate04_ward as G4
    g = pd.DataFrame({"stress_absolute_error": [0.25], "_tgt": [0.5]})
    assert G4.e_norm(g) == pytest.approx(0.5, rel=1e-12)


# ------------------------------------------------------- generated outputs ---
def test_crosswalk_reproduces_every_quoted_manuscript_value():
    import run_gate04_ward as G4
    c = _load("ward_metric_crosswalk.csv")
    q = c[c.kind == "manuscript_quoted_value"]
    assert len(q) == len(G4.QUOTED)
    assert q.reproduces_quoted.all(), "a quoted value no longer reproduces"
    # and each is an ABSOLUTE metric mislabelled as normalized
    assert not q.actual_metric_is_normalized.any()
    assert not q.label_is_correct.any()
    assert not q.m_attribution_is_valid.any()
    # the normalized value is materially larger in every case
    assert (q.normalized_over_quoted > 5.0).all()


def test_estimator_classes_are_never_pooled():
    for name in ("deterministic_ward_normalized_blocks.csv",
                 "sampled_ward_normalized_blocks.csv"):
        d = _load(name)
        assert d.source_estimator_class.nunique() == 1
    det = _load("deterministic_ward_normalized_blocks.csv")
    sam = _load("sampled_ward_normalized_blocks.csv")
    assert det.source_estimator_class.iloc[0] == "deterministic_finite_feature"
    assert sam.source_estimator_class.iloc[0] == "sampled_finite_network"
    # M exists only for the sampled class
    assert "n_networks" in sam.columns
    assert "n_networks" not in det.columns or det["n_networks"].isna().all()


def test_metric_version_is_stamped_on_every_row():
    for name in ("ward_metric_crosswalk.csv",
                 "deterministic_ward_normalized_blocks.csv",
                 "sampled_ward_normalized_blocks.csv",
                 "sampled_ward_monte_carlo_scaling.csv",
                 "ward_paired_ratio.csv"):
        d = _load(name)
        assert (d.metric_version == "ward_norm_v1").all(), name


def test_sampled_ward_is_monte_carlo_noise_not_architecture():
    """The headline Gate-4 finding, as a regression."""
    m = _load("sampled_ward_monte_carlo_scaling.csv")
    f = m[m.m_scaling_exponent.notna()]
    assert len(f) >= 9
    assert f.m_scaling_exponent.mean() == pytest.approx(-0.5, abs=0.1)
    assert f.consistent_with_pure_monte_carlo.mean() > 0.8
    # and the M needed to reach E_norm < 1 is far beyond what was run
    assert (f.m_required_for_e_norm_below_1 > 1e8).all()


def test_paired_ratio_discrimination_emerges_with_width():
    p = _load("ward_paired_ratio.csv")
    d = p[p.source_estimator_class == "deterministic_finite_feature"]
    big = d[d.width_subset == "200000"].set_index("architecture")
    small = d[d.width_subset == "500"].set_index("architecture")
    # at the largest width RBF is worse than selected in every paired comparison
    assert bool(big.loc["rbf_gaussian_prior", "worse_than_selected_in_all_pairs"])
    assert big.loc["rbf_gaussian_prior", "ratio_mean"] > 10
    # at the smallest width it is not
    assert small.loc["rbf_gaussian_prior", "ratio_mean"] < 1.0
    # the nearest control is NOT excluded even at the largest width
    assert big.loc["local_exponent_direction", "ratio_mean"] < 5.0
    assert not bool(big.loc["local_exponent_direction",
                            "worse_than_selected_in_all_pairs"])


def test_leave_one_block_out_moves_nothing_materially():
    l = _load("sampled_ward_leave_one_block_out.csv")
    assert len(l) > 500
    assert l.shift_in_stderr.abs().max() < 1.5


def test_runner_refuses_a_weighting_it_was_not_defined_for():
    """ward_norm_v1 is specified for uniform weights; anything else must stop."""
    import yaml
    src = yaml.safe_load((ROOT / "configs" / "jhep_submission_hardening.yaml")
                         .read_text())
    assert src["gate04"]["geometry_weighting"] == "uniform"
