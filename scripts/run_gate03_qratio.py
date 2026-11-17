#!/usr/bin/env python3
r"""Gate 3: the estimator-independent ratio ``q_g = c_dir,g / c_reg,g^analytic``.

The window-summarized central charge cannot settle the comparison on its own,
because the analytic reference row and the sampled rows summarize the same
``c(r)`` curve with different fit weightings (uniform vs inverse-variance). A
ratio of two window summaries inherits that weighting and so is not
estimator-independent.

Forming the ratio **separation by separation** removes the fit entirely: at each
geometry the direct estimator and the analytic regulated target are two numbers
for the same quantity, and their ratio should be 1 whatever weighting anyone
would have chosen.

Uncertainties are taken at the level of the independent replicate -- the block.
The four separations inside a block share the same realizations and are strongly
correlated, so the test is the covariance-aware one: Hotelling's ``T^2`` on the
block-level 4-vector, plus a GLS fit of the single common ratio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro import stress_tensor as ST                 # noqa: E402
from nnft_virasoro.architectures import _tukey_log_window     # noqa: E402
from nnft_virasoro.cli import base_parser, setup              # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows    # noqa: E402
from nnft_virasoro.ordered_pairing import _gauss_legendre_panels  # noqa: E402
from scipy.special import jv                                  # noqa: E402

SCRIPT = "scripts/run_gate03_qratio.py"


def analytic_c(r, lir, luv, taper):
    pts, wts = _gauss_legendre_panels(lir, luv, np.pi / (8 * max(r, 1e-12)),
                                      nodes=8)
    w = (_tukey_log_window(np.log(pts), np.log(lir), np.log(luv), taper)
         if taper > 0 else 1.0)
    return (r * r * float(np.sum(wts * pts * w * jv(2, pts * r)) / 2.0)) ** 2


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    sec = cfg.section("gate03")
    ex = sec["direct_extension"]
    out = outdir / "gate03_q_ratio_blocks.csv"
    if should_skip(out, args.skip_existing):
        print("  [skip] gate03 q-ratio")
        return 0
    src = outdir / "sampled_tt_direct_extension_points.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}; run the direct extension first")

    pts = pd.read_csv(src)
    lir, luv = float(ex["lambda_ir"]), float(ex["lambda_uv"])
    taper = float(ex["taper"])

    rows = []
    test_rows = []
    for variant in ("direct_TT", "wick_factorized"):
        d = pts[(pts.estimator_variant == variant) & pts.in_fit_window]
        seps = np.array(sorted(d.separation.unique()))
        ana = np.array([analytic_c(float(r), lir, luv, taper) for r in seps])
        blocks = sorted(d.seed_name.unique())
        Q = np.full((len(blocks), seps.size), np.nan)
        for i, b in enumerate(blocks):
            gb = d[d.seed_name == b].set_index("separation")
            for j, r in enumerate(seps):
                if r in gb.index:
                    Q[i, j] = float(gb.loc[r, "c_per_separation"]) / ana[j]
        for i, b in enumerate(blocks):
            for j, r in enumerate(seps):
                rows.append(stamper.stamp(dict(
                    gate="gate03", check="q_ratio", estimator_variant=variant,
                    seed_name=b, block_id=b, separation_index=int(j),
                    separation=float(r), analytic_c=float(ana[j]),
                    sampled_c=float(Q[i, j] * ana[j]), q=float(Q[i, j]),
                    lambda_ir=lir, lambda_uv=luv, regulator_taper=taper),
                    estimator_class="sampled_finite_network", seed=b))
        # ---- covariance-aware test that q == 1 ------------------------
        good = np.all(np.isfinite(Q), axis=1)
        X = Q[good]
        n, k = X.shape
        qbar = X.mean(axis=0)
        S = np.cov(X, rowvar=False, ddof=1)
        diag_se = np.sqrt(np.diag(S) / n)
        summ = dict(
            gate="gate03", check="q_ratio_test", estimator_variant=variant,
            n_blocks=int(n), n_separations=int(k),
            separations=";".join(f"{r:.4f}" for r in seps),
            q_mean=";".join(f"{v:.6f}" for v in qbar),
            q_stderr=";".join(f"{v:.6f}" for v in diag_se),
            q_grand_mean=float(qbar.mean()),
            lambda_ir=lir, lambda_uv=luv, regulator_taper=taper,
            uncertainty_level="independent named-seed blocks",
            correlation_note=("the k separations share realizations within a "
                              "block, so the test uses the block covariance, "
                              "not k independent errors"))
        try:
            Sinv = np.linalg.inv(S)
            d1 = qbar - 1.0
            T2 = float(n * d1 @ Sinv @ d1)
            F = (n - k) / (k * (n - 1)) * T2
            pF = float(stats.f.sf(F, k, n - k))
            ones = np.ones(k)
            var_l = 1.0 / float(ones @ Sinv @ ones)
            lam = float(var_l * (ones @ Sinv @ qbar))
            se_l = float(np.sqrt(var_l / n))
            tl = (lam - 1.0) / se_l
            pl = float(2 * stats.t.sf(abs(tl), n - 1))
            resid = qbar - lam
            chi2 = float(n * resid @ Sinv @ resid)
            summ.update(
                hotelling_T2=T2, hotelling_F=float(F), hotelling_dof1=int(k),
                hotelling_dof2=int(n - k), hotelling_p=pF,
                rejects_q_equals_one_at_5pct=bool(pF < 0.05),
                gls_common_ratio=lam, gls_common_ratio_stderr=se_l,
                gls_ci_low=float(lam - 1.96 * se_l),
                gls_ci_high=float(lam + 1.96 * se_l),
                gls_t_vs_one=float(tl), gls_p_vs_one=pl,
                gls_consistent_with_one=bool(pl >= 0.05),
                gls_goodness_chi2=chi2, gls_goodness_dof=int(k - 1),
                gls_goodness_p=float(stats.chi2.sf(chi2, k - 1)),
                covariance_status="ok")
        except np.linalg.LinAlgError as exc:
            summ.update(covariance_status=f"singular: {exc}")
        test_rows.append(stamper.stamp(
            summ, estimator_class="sampled_finite_network", seed="multi"))
        if summ.get("covariance_status") == "ok":
            print(f"  {variant}: GLS common ratio "
                  f"{summ['gls_common_ratio']:.5f} +/- "
                  f"{summ['gls_common_ratio_stderr']:.5f} "
                  f"(vs 1: p={summ['gls_p_vs_one']:.3f}); "
                  f"Hotelling p={summ['hotelling_p']:.3f} over "
                  f"{summ['n_blocks']} blocks x {summ['n_separations']} separations")

    write_rows(out, rows, required=[
        "gate", "check", "estimator_variant", "seed_name", "separation_index",
        "separation", "analytic_c", "sampled_c", "q"],
        sort_by=["estimator_variant", "separation_index", "seed_name"])
    write_rows(outdir / "gate03_q_ratio_test.csv", test_rows, required=[
        "gate", "check", "estimator_variant", "n_blocks", "n_separations",
        "uncertainty_level", "correlation_note"],
        sort_by=["estimator_variant"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
