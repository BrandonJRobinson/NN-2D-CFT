"""Gate 7: the explicit randomized Cauchy NS architecture."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nnft_virasoro import ns_cauchy as NC
from nnft_virasoro import ns_fermion as NF
from nnft_virasoro import ordered_pairing as OP

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "jhep_submission_hardening"
TT = NF.fermion_mode_window(9)
CAN = NF.canonical_car(TT)
R0, RHO = 1.0, 2.0
_, R_IN, R_OUT = OP.radii_from_window(R0 / RHO, R0 * RHO, RHO, r0=R0)


def _load(name):
    p = RES / name
    if not p.exists():
        pytest.skip(f"{name} not generated yet")
    return pd.read_csv(p)


# ------------------------------------------------------------ proposal ---
def test_geometric_proposal_is_normalized_and_rejects_bad_q():
    P = NC.geometric_proposal(0.75, 400)
    assert P.sum() == pytest.approx(1.0, rel=1e-15)
    assert np.all(P[:-1] >= P[1:])
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            NC.geometric_proposal(bad, 100)


def test_finite_variance_threshold_is_the_radius_ratio_squared():
    for q, want in ((0.10, False), (0.20, False), (0.30, True), (0.75, True)):
        c = NC.finite_variance_condition(q, R_IN, R_OUT)
        assert c["radius_ratio_squared"] == pytest.approx(1 / RHO**2, rel=1e-12)
        assert c["finite_variance"] is want


@pytest.mark.parametrize("q,diverges", [(0.10, True), (0.20, True),
                                        (0.30, False), (0.50, False),
                                        (0.75, False), (0.90, False)])
def test_exact_variance_diverges_only_below_the_threshold(q, diverges):
    """The demonstration: exact variance vs truncation, not a sample s.d."""
    z, w = R_OUT + 0j, R_IN * np.exp(0.7j)
    v = [NC.estimator_variance(z, w, NC.geometric_proposal(q, n))
         for n in (50, 100, 200)]
    if diverges:
        assert v[-1] > 1e6 * v[0]
    else:
        assert v[-1] == pytest.approx(v[0], rel=0.01)
        assert np.isfinite(v[-1])


def test_exact_variance_matches_the_empirical_variance_where_finite():
    rng = np.random.default_rng(3)
    z, w = R_OUT + 0j, R_IN * np.exp(0.7j)
    P, N = NC.geometric_proposal(0.75, 400), 20000
    vals = [complex(NC.cauchy_kernel_estimate(z, w, NC.sample_ns_modes(rng, N, P), P))
            for _ in range(400)]
    assert np.var(vals, ddof=1) == pytest.approx(
        NC.estimator_variance(z, w, P, N), rel=0.35)


def test_optimal_q_maximizes_the_rarest_mode_weight():
    n = 4
    q = NC.optimal_geometric_q(n)
    assert q == pytest.approx(0.8, rel=1e-12)
    best = (1 - q) * q**n
    for other in (0.5, 0.6, 0.7, 0.75, 0.9, 0.95):
        assert (1 - other) * other**n <= best + 1e-15


# ----------------------------------------------------------- estimator ---
def test_cauchy_kernel_estimate_is_unbiased_in_both_orderings():
    rng = np.random.default_rng(5)
    P = NC.geometric_proposal(0.75, 400)
    for z, w in ((R_OUT + 0j, R_IN * np.exp(0.7j)),
                 (R_IN * np.exp(-0.3j), R_OUT * np.exp(1.1j))):
        est = np.array([complex(NC.cauchy_kernel_estimate(
            z, w, NC.sample_ns_modes(rng, 4000, P), P)) for _ in range(400)])
        exact = 1.0 / (z - w)
        sem = est.std(ddof=1) / np.sqrt(est.size)
        assert abs(est.mean() - exact) < 4 * sem


def test_the_two_orderings_are_antisymmetric():
    """K(z,w) = -K(w,z): the ordering discontinuity that becomes the CAR."""
    rng = np.random.default_rng(9)
    P = NC.geometric_proposal(0.75, 400)
    m = NC.sample_ns_modes(rng, 3000, P)
    z, w = R_OUT + 0j, R_IN * np.exp(0.7j)
    a = complex(NC.cauchy_kernel_estimate(z, w, m, P))
    b = complex(NC.cauchy_kernel_estimate(w, z, m, P))
    assert a == pytest.approx(-b, rel=1e-12)


# ------------------------------------------------------------- pairing ---
def test_closed_form_and_quadrature_paths_agree():
    rng = np.random.default_rng(13)
    P = NC.geometric_proposal(0.75, 400)
    m = NC.sample_ns_modes(rng, 5000, P)
    a = NC.cauchy_pairing_exact(TT, m, P)
    b = NC.cauchy_pairing_quadrature(TT, m, P, R_IN, R_OUT, q=64, r0=R0)
    assert np.max(np.abs(a.omega - b.omega)) < 1e-12


def test_car_structure_is_exact_and_leakage_is_therefore_not_an_observable():
    rng = np.random.default_rng(17)
    P = NC.geometric_proposal(0.75, 400)
    eta = NC.cauchy_pairing_exact(TT, NC.sample_ns_modes(rng, 2000, P), P).omega
    off = ~((TT[:, None] + TT[None, :]) == 0)
    assert np.all(eta[off] == 0), "structure must be exactly antidiagonal"
    assert NF.car_metrics(eta, TT)["fermionic_pairing_leakage"] == 0.0


def test_car_error_follows_the_predicted_N_to_the_minus_half_law():
    rng = np.random.default_rng(23)
    P = NC.geometric_proposal(0.75, 400)
    errs = {}
    for N in (2000, 8000, 32000):
        e = [np.max(np.abs(NC.cauchy_pairing_exact(
            TT, NC.sample_ns_modes(rng, N, P), P).omega - CAN)) for _ in range(24)]
        errs[N] = float(np.mean(e))
    assert errs[2000] / errs[8000] == pytest.approx(2.0, rel=0.4)
    assert errs[8000] / errs[32000] == pytest.approx(2.0, rel=0.4)
    pred = float(np.nanmax(NC.car_error_prediction(TT, P, 8000)))
    assert 0.3 * pred < errs[8000] < 3.0 * pred


# ------------------------------------------------------ generated output ---
def test_all_super_virasoro_rows_are_pre_darboux():
    """Mission rule 3: post-Darboux closure is not architecture evidence."""
    a = _load("ns_super_virasoro_seed_rows.csv")
    assert bool(a.pre_darboux.all())
    assert not any("canonical" in c for c in a.columns)


def test_both_constructions_converge_at_minus_one_half():
    f = _load("ns_super_virasoro_fits.csv")
    sel = f[f.architecture == "selected_log_kernel"]
    assert set(sel.ns_construction) == {"native_derived_from_boson",
                                        "randomized_cauchy"}
    assert len(sel) == 6
    assert (sel.null_half_p_value > 0.05).all(), "null -1/2 must not be rejected"
    assert (sel.exponent.abs().between(0.4, 0.6)).all()


def test_central_charge_approaches_three_halves():
    a = _load("ns_super_virasoro_seed_rows.csv")
    s = a[a.architecture == "selected_log_kernel"]
    big = s[s.n_features == s.n_features.max()]
    for cons, g in big.groupby("ns_construction"):
        assert g.raw_central_charge_fit.mean() == pytest.approx(1.5, abs=0.02), cons


def test_the_added_architecture_does_not_borrow_the_boson_width():
    p = _load("ns_pairing_seed_rows.csv")
    cau = p[p.ns_construction == "randomized_cauchy"]
    nat = p[p.ns_construction == "native_derived_from_boson"]
    assert bool(cau.independent_width.all())
    assert not bool(nat.independent_width.any())
    assert bool(nat.derived_from_boson.all())


def test_evaluation_paths_agree_in_the_generated_rows():
    p = _load("ns_pairing_seed_rows.csv")
    g = p[p.exact_vs_quadrature_max_gap.notna()]
    assert len(g) > 0
    assert g.exact_vs_quadrature_max_gap.max() < 1e-12
