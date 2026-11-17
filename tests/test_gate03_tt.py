"""Gate 3: TT estimators, normal ordering, the cumulant identity, weighting."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nnft_virasoro import kernels as K
from nnft_virasoro import stress_tensor as ST
from nnft_virasoro.config import load_config, rng_for
from nnft_virasoro.sampling import iter_batches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results" / "jhep_submission_hardening"
CFG = ROOT / "configs" / "jhep_submission_hardening.yaml"


def _load(name):
    p = RES / name
    if not p.exists():
        pytest.skip(f"{name} not generated yet")
    return pd.read_csv(p)


def _rng(tag):
    return rng_for(load_config(CFG).section("named_seeds"), "tt_s01", tag)


# ------------------------------------------------------- normal ordering ---
def test_normal_ordering_subtraction_is_an_exact_conditional_expectation():
    """c_N = E_gamma[(d phi)^2 | {k_j}], exactly, for FIXED momenta.

    This is why the subtraction carries no self-estimation bias: conditioned on
    the momentum draw it is a closed form, not something estimated from the
    field samples. Averaging over k as well would not test this -- |c_N| has
    enormous k-variance because |k|^2 reaches Lambda_UV^2.
    """
    rng = _rng("cond")
    N, L_w = 48, K.tapered_log_measure(1e-2, 1e2, 0.15)
    sigma = np.sqrt(K.sigma_phi_squared(1e-2, 1e2, log_measure=L_w))
    from nnft_virasoro.architectures import sample_architecture
    fs = sample_architecture("selected_log_kernel", rng, N, 1e-2, 1e2,
                             params={"regulator_taper": 0.15})
    kc = np.asarray(fs.kc).ravel()[:N]
    z = 0.37 - 0.21j
    c_N = (L_w / (2.0 * N)) * np.sum(np.conj(kc) ** 2)

    G = 400000                                  # gamma draws, momenta FIXED
    gam = rng.uniform(0.0, 2 * np.pi, size=(G, N))
    ph = np.real(np.conj(kc) * z)[None, :] + gam
    dphi = -(sigma / (2.0 * np.sqrt(N))) * np.sum(np.sin(ph) * np.conj(kc), axis=1)
    emp = np.mean(dphi**2)
    assert emp == pytest.approx(c_N, rel=0.05), f"{emp} vs {c_N}"


def test_normal_ordering_constant_shrinks_like_one_over_sqrt_N():
    """|c_N| ~ L_w rms(k^2) / (2 sqrt N).

    sum_j conj(k_j)^2 is a zero-mean random walk over an isotropic angular
    prior, so it grows like sqrt(N), not N, and the 1/N prefactor leaves
    N^{-1/2}. Asserting O(1) or O(1/N) here would both be wrong.
    """
    mags = {}
    for nfeat in (64, 256, 1024):
        rng = _rng(f"sqrtscale{nfeat}")
        tot, n = 0.0, 0
        for b in iter_batches(rng, n_features=nfeat, n_realizations=3000,
                              points=np.array([0.5 + 0j]),
                              lambda_ir=1e-2, lambda_uv=1e2,
                              params={"regulator_taper": 0.15}, batch_size=750):
            tot += float(np.sum(np.abs(b.self_contraction)))
            n += b.n_realizations
        mags[nfeat] = tot / n
    # a 16x rise in N must cut |c_N| by about 4
    assert mags[64] / mags[1024] == pytest.approx(4.0, rel=0.35)
    assert mags[64] / mags[256] == pytest.approx(2.0, rel=0.35)


# -------------------------------------------------- direct / Wick identity ---
def test_direct_wick_cumulant_identity_converges_with_M():
    """G_dir = G_Wick + kappa_4/4, as a regression on the generated table."""
    d = _load("gate03_cumulant_identity.csv")
    assert set(d.n_networks.unique()) >= {50000, 200000}
    lo = d[d.n_networks == d.n_networks.min()].identity_rel_error.mean()
    hi = d[d.n_networks == d.n_networks.max()].identity_rel_error.mean()
    assert hi < lo, "residual must fall as M grows"
    assert hi < 2e-3
    assert bool(d.identity_residual_shrinks_with_M.all())


def test_identity_is_checked_where_the_direct_estimator_is_conditioned():
    """It must NOT be asserted at the production cutoff, where it is unverifiable."""
    d = _load("gate03_cumulant_identity.csv")
    assert (d.lambda_uv < 10).all(), "identity check ran at a punishing cutoff"
    cfg = load_config(CFG).section("gate03")
    assert float(cfg["identity_check"]["production_reference"]["lambda_uv"]) == 1e2


# ------------------------------------------------------------- weighting ---
def test_uniform_and_inverse_variance_fits_differ_on_the_same_curve():
    """Rule 11: one c(r) curve, two weightings, two different `c_exp`."""
    w = _load("gate03_fit_weighting_crosswalk.csv")
    assert set(w.weighting) == {"uniform", "inverse_variance"}
    uni = float(w[w.weighting == "uniform"].analytic_c_exp.iloc[0])
    inv = float(w[w.weighting == "inverse_variance"].analytic_c_exp.iloc[0])
    assert uni == pytest.approx(1.0132, abs=2e-3)
    assert inv == pytest.approx(1.0359, abs=3e-3)
    assert abs(inv - uni) > 0.015, "the weighting shift is the whole point"
    # the shift exists because c(r) is not flat across the window
    assert float(w.c_of_r_spread.iloc[0]) > 0.02


def test_fit_central_charge_falls_back_to_uniform_weights_on_zero_errors():
    """Why the analytic reference row is uniformly weighted at all."""
    seps = np.array([0.31, 0.39, 0.46, 0.53])
    vals = np.array([1.06, 1.04, 0.95, 1.00])
    zero = ST.fit_central_charge(seps, vals, np.zeros_like(vals), (0.3, 0.55))
    assert zero["c_exp"] == pytest.approx(vals.mean(), rel=1e-12)
    # with real errors it is inverse-variance weighted and differs
    err = np.array([0.035, 0.055, 0.073, 0.105])
    wgt = ST.fit_central_charge(seps, vals, err, (0.3, 0.55))
    assert wgt["c_exp"] != pytest.approx(zero["c_exp"], rel=1e-6)
    w = 1.0 / err**2
    assert wgt["c_exp"] == pytest.approx(float(np.sum(w * vals) / w.sum()), rel=1e-12)


# ------------------------------------------------ analytic regulator limit ---
def test_analytic_target_converges_to_one_as_the_regulator_widens():
    d = _load("gate03_regulator_convergence.csv").sort_values("lambda_uv")
    assert d.max_abs_deviation_from_one.iloc[-1] < 1e-4
    assert (d.max_abs_deviation_from_one.diff().dropna() < 0).all()
    assert bool(d.converges_to_one.iloc[-1])


def test_production_regulator_target_is_not_one():
    """The honest statement: at the production cutoff the target is ~1.013."""
    d = _load("gate03_regulator_convergence.csv")
    prod = d[d.lambda_uv == 1e2]
    assert len(prod) == 1
    assert float(prod.c_exp_uniform.iloc[0]) == pytest.approx(1.0132, abs=2e-3)
    assert float(prod.max_abs_deviation_from_one.iloc[0]) > 0.05


# --------------------------------------------------------- provenance ---
def test_headline_is_wick_only_at_the_smaller_width():
    p = _load("tt_estimator_provenance.csv")
    assert len(p) == 1
    r = p.iloc[0]
    assert bool(r.is_wick_only) and not bool(r.is_direct_only)
    assert not bool(r.is_combined)
    assert not bool(r.is_largest_width), "headline sits at the SMALLER width"
    assert int(r.n_features) == 250
    assert int(r.n_blocks) == 12


def test_direct_extension_is_an_independent_draw_and_reaches_its_target():
    d = _load("sampled_tt_direct_extension_summary.csv")
    direct = d[d.estimator_variant == "direct_TT"].iloc[0]
    assert int(direct.n_networks_per_block) > 100 * int(direct.frozen_n_networks_per_block)
    assert bool(direct.reached_target_se)
    # the assumption-free determination is consistent with c = 1
    assert abs(float(direct.z_vs_c1)) < 2.0
    b = _load("sampled_tt_direct_extension_blocks.csv")
    assert (b.rng_stream == "tt_direct_ext").all(), "must not reuse the frozen stream"


def test_extended_wick_matches_its_matched_weight_analytic_target():
    """The 23-sigma gap against c=1 closes against the correct reference."""
    d = _load("sampled_tt_direct_extension_summary.csv")
    w = d[d.estimator_variant == "wick_factorized"].iloc[0]
    cw = _load("gate03_fit_weighting_crosswalk.csv")
    target = float(cw[cw.weighting == "inverse_variance"].analytic_c_exp.iloc[0])
    assert abs(float(w.c) - 1.0) / float(w.c_se) > 10      # far from 1 ...
    assert abs(float(w.c) - target) / float(w.c_se) < 3    # ... but not from target


# ------------------------------------------- estimator-independent q ratio ---
def test_q_ratio_is_consistent_with_one_for_both_estimators():
    """q_g = c_g / c_g^analytic, formed per separation so it carries no fit."""
    t = _load("gate03_q_ratio_test.csv")
    assert set(t.estimator_variant) == {"direct_TT", "wick_factorized"}
    for _, r in t.iterrows():
        assert r.covariance_status == "ok"
        assert int(r.n_blocks) == 12 and int(r.n_separations) == 4
        # the GLS common ratio is consistent with 1 ...
        assert float(r.gls_p_vs_one) > 0.05, r.estimator_variant
        assert float(r.gls_ci_low) < 1.0 < float(r.gls_ci_high)
        # ... and so is the joint, covariance-aware test of all separations
        assert float(r.hotelling_p) > 0.05, r.estimator_variant


def test_wick_q_ratio_is_tight_and_the_direct_one_is_honest_about_its_tail():
    t = _load("gate03_q_ratio_test.csv").set_index("estimator_variant")
    w = t.loc["wick_factorized"]
    assert float(w.gls_common_ratio) == pytest.approx(1.0, abs=0.005)
    assert float(w.gls_common_ratio_stderr) < 0.01
    d = t.loc["direct_TT"]
    assert float(d.gls_common_ratio) == pytest.approx(1.0, abs=0.10)
    # the marginal goodness of fit is a recorded property, not a silent one
    assert np.isfinite(float(d.gls_goodness_p))


def test_q_ratio_uncertainty_is_taken_at_the_block_level():
    t = _load("gate03_q_ratio_test.csv")
    assert (t.uncertainty_level == "independent named-seed blocks").all()
    assert t.correlation_note.str.contains("block covariance").all()
    b = _load("gate03_q_ratio_blocks.csv")
    assert b.seed_name.nunique() == 12
    assert b.separation.nunique() == 4
