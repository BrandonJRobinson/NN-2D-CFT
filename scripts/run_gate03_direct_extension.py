#!/usr/bin/env python3
"""Gate 3: extend the DIRECT four-field TT estimator until it can stand alone.

The frozen direct determination at the headline setting is c = 2.06 +/- 0.58.
That neither establishes c = 1 nor excludes it, so no assumption-free estimator
currently supports the manuscript's "independent sampled <TT> determination".
The mission permits extending direct Monte Carlo exactly in this case.

Nothing about the measurement changes except the number of realizations:
geometry, regulator, architecture and fit window are the frozen ones. The RNG
stream label differs from the frozen run, so this is an INDEPENDENT draw and
not a correlated superset of the frozen samples; the frozen direct result
therefore remains usable as a separate consistency check.
"""
from __future__ import annotations

import os

# Pin the BLAS/OpenMP thread pools BEFORE numpy is imported. Twelve worker
# processes each spinning up their own thread pool oversubscribes the machine
# and halves throughput; the numerics are unaffected (verified bit-identical
# block values with and without pinning).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro import stress_tensor as ST                # noqa: E402
from nnft_virasoro.cli import base_parser, setup             # noqa: E402
from nnft_virasoro.config import load_config, rng_for        # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows   # noqa: E402
from nnft_virasoro.sampling import block_statistics, bootstrap_ci  # noqa: E402

SCRIPT = "scripts/run_gate03_direct_extension.py"
STREAM = "tt_direct_ext"       # distinct from the frozen run's "tt" stream


def _one_block(job):
    """Run a single named seed. Executed in a worker process."""
    cfg_path, seed_name, n_networks = job
    cfg = load_config(cfg_path)
    ex = cfg.section("gate03")["direct_extension"]
    geo = ST.circle_geometry(ex["n_circle_points"], ex["circle_radius"])
    rng = rng_for(cfg.section("named_seeds"), seed_name, STREAM,
                  ex["architecture"], ex["n_features"], ex["regulator_class"])
    t0 = time.time()
    res = ST.sampled_tt(
        rng, n_features=ex["n_features"], n_networks=n_networks, geometry=geo,
        lambda_ir=ex["lambda_ir"], lambda_uv=ex["lambda_uv"],
        architecture=ex["architecture"],
        params={"regulator_taper": ex["taper"]}, batch_size=ex["batch_size"])
    win = tuple(ex["fit_window"])
    out = {}
    points = {}
    for variant, vals, errs in (
            ("direct_TT", res["c_direct_per_separation"], res["c_direct_stderr"]),
            ("wick_factorized", res["c_wick_per_separation"],
             res["c_wick_stderr"])):
        out[variant] = ST.fit_central_charge(res["separations"], vals, errs, win)
        points[variant] = (np.asarray(vals, dtype=float).tolist(),
                           np.asarray(errs, dtype=float).tolist())
    return dict(seed_name=seed_name, n_networks=int(res["n_networks"]),
                separations=np.asarray(res["separations"],
                                       dtype=float).tolist(),
                points=points,
                normal_ordering_subtraction=float(np.real(
                    res["normal_ordering_subtraction"])),
                wall_seconds=time.time() - t0, fits=out)


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--quick", action="store_true",
                   help="1/500 of the realizations, to exercise the path only")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    ex = cfg.section("gate03")["direct_extension"]
    out_path = outdir / "sampled_tt_direct_extension_blocks.csv"
    if should_skip(out_path, args.skip_existing):
        print("  [skip] gate03 direct extension")
        return 0
    if not ex.get("enabled", False):
        print("  direct_extension disabled in config; nothing to do")
        return 0

    M = int(ex["n_networks"])
    if args.quick:
        M = max(2000, M // 500)
    seeds = list(ex["seeds"])
    nw = min(int(ex["n_workers"]), len(seeds), os.cpu_count() or 1)
    print(f"  direct extension: {len(seeds)} blocks x M={M:,} "
          f"({len(seeds) * M:,} realizations) on {nw} workers")

    cfg_path = str(args.config)
    jobs = [(cfg_path, s, M) for s in seeds]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as pool:
        done = []
        for r in pool.map(_one_block, jobs):
            done.append(r)
            print(f"    block {r['seed_name']}: "
                  f"c_direct={r['fits']['direct_TT']['c_exp']:+.4f}  "
                  f"({r['wall_seconds'] / 60:.1f} min)", flush=True)
    wall = time.time() - t0
    print(f"  all blocks finished in {wall / 60:.1f} min wall")

    rows = []
    for r in done:
        for variant, fit in r["fits"].items():
            rows.append(stamper.stamp(dict(
                gate="gate03", campaign="direct_extension",
                estimator_variant=variant,
                spectral_regulator_class=ex["regulator_class"],
                regulator_taper=float(ex["taper"]),
                architecture=ex["architecture"], seed_name=r["seed_name"],
                block_id=r["seed_name"], n_features=int(ex["n_features"]),
                n_networks=r["n_networks"],
                lambda_ir=float(ex["lambda_ir"]), lambda_uv=float(ex["lambda_uv"]),
                n_circle_points=int(ex["n_circle_points"]),
                circle_radius=float(ex["circle_radius"]),
                rng_stream=STREAM, fit_window_index=0,
                normal_ordering_subtraction=r["normal_ordering_subtraction"],
                wall_seconds=r["wall_seconds"], **fit),
                estimator_class="sampled_finite_network", seed=r["seed_name"]))
    write_rows(out_path, rows, required=[
        "gate", "campaign", "estimator_variant", "spectral_regulator_class",
        "architecture", "seed_name", "n_features", "n_networks", "c_exp",
        "rng_stream"],
        sort_by=["estimator_variant", "seed_name"])

    # ---- per-separation values, for the estimator-independent ratio -----
    # q_g = c_direct,g / c_analytic,g must be formed SEPARATION BY SEPARATION;
    # a ratio of window-summarized numbers would inherit the fit weighting and
    # so could not be estimator-independent.
    lo, hi = [float(x) for x in ex["fit_window"]]
    pt_rows = []
    for r in done:
        seps = np.asarray(r["separations"], dtype=float)
        for variant, (vals, errs) in r["points"].items():
            for g, (sep, v, e) in enumerate(zip(seps, vals, errs)):
                pt_rows.append(stamper.stamp(dict(
                    gate="gate03", campaign="direct_extension",
                    estimator_variant=variant, seed_name=r["seed_name"],
                    block_id=r["seed_name"], separation_index=int(g),
                    separation=float(sep), c_per_separation=float(v),
                    c_per_separation_stderr=float(e),
                    in_fit_window=bool(lo - 1e-12 <= sep <= hi + 1e-12),
                    n_features=int(ex["n_features"]), n_networks=r["n_networks"],
                    spectral_regulator_class=ex["regulator_class"],
                    rng_stream=STREAM),
                    estimator_class="sampled_finite_network",
                    seed=r["seed_name"]))
    write_rows(outdir / "sampled_tt_direct_extension_points.csv", pt_rows,
               required=["gate", "estimator_variant", "seed_name",
                         "separation_index", "separation", "c_per_separation",
                         "in_fit_window"],
               sort_by=["estimator_variant", "separation_index", "seed_name"])

    # ---- block aggregate, and the comparison that decides the claim -----
    frozen = pd.read_csv(ROOT / cfg.section("gate03")["frozen_seed_rows"])
    fsel = frozen[(frozen.architecture == ex["architecture"])
                  & (frozen.regulator_class == ex["regulator_class"])
                  & (frozen.n_features == ex["n_features"])
                  & (frozen.fit_window_index == 0)]
    d = pd.DataFrame(rows)
    summ = []
    for variant, sub in d.groupby("estimator_variant"):
        v = sub["c_exp"].to_numpy(float)
        st = block_statistics(v)
        lo, hi = bootstrap_ci(v, seed=20260730)
        fz = fsel[fsel.estimator_variant == variant]["c_exp"].to_numpy(float)
        fst = block_statistics(fz) if len(fz) else {"mean": np.nan,
                                                    "stderr": np.nan,
                                                    "n_blocks": 0}
        dif = st["mean"] - fst["mean"]
        dse = float(np.hypot(st["stderr"], fst["stderr"]))
        rec = dict(
            gate="gate03", campaign="direct_extension",
            estimator_variant=variant,
            spectral_regulator_class=ex["regulator_class"],
            n_features=int(ex["n_features"]), n_networks_per_block=M,
            n_blocks=st["n_blocks"], c=st["mean"], c_se=st["stderr"],
            c_ci_low=lo, c_ci_high=hi,
            z_vs_c1=((st["mean"] - 1.0) / st["stderr"]
                     if st["stderr"] > 0 else float("nan")),
            compatible_with_c1=bool(st["stderr"] > 0
                                    and abs(st["mean"] - 1.0) <= 2 * st["stderr"]),
            excludes_c1_at_2sigma=bool(st["stderr"] > 0
                                       and abs(st["mean"] - 1.0) > 2 * st["stderr"]),
            frozen_c=fst["mean"], frozen_c_se=fst["stderr"],
            frozen_n_blocks=fst["n_blocks"],
            frozen_n_networks_per_block=int(ex["frozen_n_networks"]),
            extension_minus_frozen=dif, extension_minus_frozen_se=dse,
            extension_frozen_z=dif / dse if dse > 0 else float("nan"),
            consistent_with_frozen_2sigma=bool(dse > 0 and abs(dif) <= 2 * dse),
            se_shrink_vs_frozen=(fst["stderr"] / st["stderr"]
                                 if st["stderr"] > 0 else float("nan")),
            se_shrink_expected=float(np.sqrt(M / ex["frozen_n_networks"])),
            target_block_se=float(ex["target_block_se"]),
            reached_target_se=bool(st["stderr"] <= 1.25 * ex["target_block_se"]),
            wall_minutes=wall / 60.0)
        summ.append(stamper.stamp(rec, estimator_class="sampled_finite_network",
                                  seed="multi"))
    write_rows(outdir / "sampled_tt_direct_extension_summary.csv", summ, required=[
        "gate", "campaign", "estimator_variant", "c", "c_se", "z_vs_c1",
        "compatible_with_c1", "extension_frozen_z", "se_shrink_vs_frozen"],
        sort_by=["estimator_variant"])

    for r in summ:
        print(f"  {r['estimator_variant']:>16}: c = {r['c']:.4f} +/- {r['c_se']:.4f}"
              f"  (z vs c=1: {r['z_vs_c1']:+.2f};  SE shrink {r['se_shrink_vs_frozen']:.1f}x,"
              f" expected {r['se_shrink_expected']:.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
