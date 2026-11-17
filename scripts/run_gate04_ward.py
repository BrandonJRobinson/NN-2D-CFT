#!/usr/bin/env python3
"""Gates 4+5: Ward-identity metric crosswalk and a normalized Ward metric.

Two separate problems, both provenance rather than physics:

1. The manuscript quotes four Ward numbers as **normalized** and attributes them
   to "the largest N and M". They are **maximum absolute** errors from the
   ``deterministic_finite_feature`` campaign, whose file has no ``n_networks``
   column at all -- ``M`` is undefined for that estimator class. No normalized
   Ward metric exists anywhere in the frozen release.

2. Two different campaigns publish a column named
   ``ward_stress_max_absolute_error`` with different probe geometries, different
   vertex positions and different width ranges. Mission rule 11 requires an
   explicit crosswalk rather than a silent comparison.

This gate therefore (a) defines a normalized metric, pre-specified here and
versioned, (b) computes it for both estimator classes, kept strictly separate,
and (c) writes the crosswalk. It does not re-run any estimator: every quantity
is reconstructed exactly from the frozen per-point rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro import plotstyle as PS                     # noqa: E402
from nnft_virasoro.cli import base_parser, setup              # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_rows    # noqa: E402
from nnft_virasoro.sampling import block_statistics, bootstrap_ci  # noqa: E402

SCRIPT = "scripts/run_gate04_ward.py"
METRIC_VERSION = "ward_norm_v1"
DET = "results/jhep_frozen/deterministic_feature_ward_points.csv"
SAM = "results/jhep_frozen/sampled_ward_seed_rows.csv"
CTL = "results/jhep_frozen/architecture_controls_seed_rows.csv"

#: The four values the manuscript quotes, with the label it gives them.
QUOTED = [
    (0.129304, "selected_log_kernel", 200000, "headline: selected at largest N"),
    (12.191757, "angular_mode_scramble", 200000, "control at largest N"),
    (2.152960, "rbf_gaussian_prior", 200000, "control at largest N"),
    (7.574309, "selected_log_kernel", 500, "selected at smallest N"),
]


def target_magnitude(df):
    """``|T_target|``, reconstructed exactly as ``abs_err / rel_err``.

    The stored ``stress_estimate`` / ``stress_target`` columns are magnitudes of
    complex quantities, so ``est - tgt`` does NOT reproduce the absolute error;
    the ratio of the two stored errors does, by definition.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t = df["stress_absolute_error"] / df["stress_relative_error"]
    return t


def e_norm(g):
    """Principal normalized Ward metric (``ward_norm_v1``).

        E_norm = sqrt( sum_g w_g |T_est - T_tgt|^2 / sum_g w_g |T_tgt|^2 )

    over the probe set, with uniform geometry weight ``w_g = 1``. Uniform is
    recorded in the config as a deliberate choice: no principled pre-specified
    weighting existed, so inventing one here would be a post-hoc fit window.

    Pooled in L2 rather than pointwise, because several probes sit near a zero
    of the target where a pointwise ratio is numerically unstable.
    """
    e = g["stress_absolute_error"].to_numpy(float)
    t = g["_tgt"].to_numpy(float)
    ok = np.isfinite(e) & np.isfinite(t)
    den = float((t[ok] ** 2).sum())
    if den <= 0:
        return float("nan")
    return float(np.sqrt((e[ok] ** 2).sum() / den))


def metric_variants(g):
    """Every normalization a reader might reasonably have meant."""
    e = g["stress_absolute_error"].to_numpy(float)
    t = g["_tgt"].to_numpy(float)
    ok = np.isfinite(e) & np.isfinite(t) & (t > 0)
    r = e[ok] / t[ok]
    return dict(
        max_absolute=float(e.max()) if e.size else float("nan"),
        mean_absolute=float(e.mean()) if e.size else float("nan"),
        rms_absolute=float(np.sqrt((e**2).mean())) if e.size else float("nan"),
        normalized_l2_pooled=e_norm(g),
        mean_pointwise_relative=float(r.mean()) if r.size else float("nan"),
        median_pointwise_relative=float(np.median(r)) if r.size else float("nan"),
        max_pointwise_relative=float(r.max()) if r.size else float("nan"),
        n_points=int(ok.sum()),
    )


def main() -> int:
    p = base_parser(__doc__ or "")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    sec = cfg.section("gate04")
    cross_path = outdir / "ward_metric_crosswalk.csv"
    if should_skip(cross_path, args.skip_existing):
        print("  [skip] gate04")
        return 0
    weighting = str(sec.get("geometry_weighting", "uniform"))
    if weighting != "uniform":
        raise SystemExit(f"ward_norm_v1 is defined for uniform weighting only, "
                         f"config says {weighting!r}")

    det = pd.read_csv(ROOT / DET)
    sam = pd.read_csv(ROOT / SAM)
    det["_tgt"] = target_magnitude(det)
    sam["_tgt"] = target_magnitude(sam)
    TAP = "tapered_log_window"

    # ---- 1. the crosswalk ------------------------------------------------
    cross = []
    dt = det[det.regulator_class == TAP]
    for value, arch, nf, role in QUOTED:
        g = dt[(dt.architecture == arch) & (dt.n_features == nf)]
        mv = metric_variants(g)
        cross.append(stamper.stamp(dict(
            gate="gate04", kind="manuscript_quoted_value",
            quoted_value=value, role=role, architecture=arch,
            manuscript_label="normalized",
            manuscript_width_attribution="largest N and M",
            actual_metric="stress_max_absolute_error",
            actual_metric_is_normalized=False,
            actual_value=mv["max_absolute"],
            reproduces_quoted=bool(abs(mv["max_absolute"] - value)
                                   <= 1e-6 * max(abs(value), 1.0)),
            source_file=DET, estimator_class_of_source="deterministic_finite_feature",
            n_features=int(nf),
            n_networks_in_source_file="ABSENT: no n_networks column",
            m_attribution_is_valid=False,
            metric_version=METRIC_VERSION,
            principal_normalized_value=mv["normalized_l2_pooled"],
            normalized_over_quoted=mv["normalized_l2_pooled"] / value,
            label_is_correct=False,
            **{k: v for k, v in mv.items() if k != "max_absolute"}),
            estimator_class="deterministic_finite_feature", seed="multi"))

    # the shared-label collision between the two campaigns
    ctl = pd.read_csv(ROOT / CTL)
    ctl_t = ctl[(ctl.regulator_class == TAP)
                & (ctl.architecture == "selected_log_kernel")
                & (ctl.n_features == ctl.n_features.max())]
    for name, src, geom, probes, nfmax, val in (
            ("controls_grid", CTL,
             "two vertices at +/-0.9, charges +/-0.3", 3,
             int(ctl.n_features.max()),
             float(ctl_t.ward_stress_max_absolute_error.max())),
            ("wpC_ward_campaign", DET,
             "three configurations (2/3/4 vertices); two_vertex at +/-0.85", 4,
             int(det.n_features.max()),
             float(dt[(dt.architecture == "selected_log_kernel")
                      & (dt.n_features == det.n_features.max())]
                   .stress_absolute_error.max()))):
        cross.append(stamper.stamp(dict(
            gate="gate04", kind="shared_label_collision",
            shared_column_name="ward_stress_max_absolute_error",
            campaign=name, source_file=src, geometry=geom,
            n_probe_points=probes, n_features_max=nfmax,
            selected_value_at_max_width=val,
            estimator_class_of_source="deterministic_finite_feature",
            comparable_across_campaigns=False,
            metric_version=METRIC_VERSION),
            estimator_class="deterministic_finite_feature", seed="multi"))
    write_rows(cross_path, cross, required=[
        "gate", "kind", "metric_version", "source_file"],
        sort_by=["kind", "architecture", "n_features"])

    # ---- 2. normalized metric, per block, each class kept separate -------
    def blocks_for(df, cls, extra_keys):
        keys = (["regulator_class", "architecture", "configuration",
                 "n_features"] + extra_keys + ["seed_name"])
        rows = []
        for kv, g in df.groupby(keys, dropna=False):
            rec = dict(zip(keys, kv))
            rec.update(metric_variants(g))
            rec.update(gate="gate04", metric_version=METRIC_VERSION,
                       geometry_weighting=weighting, block_id=rec["seed_name"],
                       source_estimator_class=cls)
            rows.append(stamper.stamp(rec, estimator_class=cls,
                                      seed=rec["seed_name"]))
        return rows

    det_blocks = blocks_for(det, "deterministic_finite_feature", [])
    sam_blocks = blocks_for(sam, "sampled_finite_network", ["n_networks"])
    write_rows(outdir / "deterministic_ward_normalized_blocks.csv", det_blocks,
               required=["gate", "architecture", "configuration", "n_features",
                         "seed_name", "normalized_l2_pooled", "metric_version"],
               sort_by=["regulator_class", "architecture", "n_features",
                        "configuration", "seed_name"])
    write_rows(outdir / "sampled_ward_normalized_blocks.csv", sam_blocks,
               required=["gate", "architecture", "configuration", "n_features",
                         "n_networks", "seed_name", "normalized_l2_pooled",
                         "metric_version"],
               sort_by=["regulator_class", "architecture", "n_features",
                        "n_networks", "configuration", "seed_name"])

    # ---- 3. block summary + leave-one-block-out -------------------------
    def summarize(rows, cls, extra_keys):
        d = pd.DataFrame(rows)
        keys = ["regulator_class", "architecture", "configuration",
                "n_features"] + extra_keys
        out, lobo = [], []
        for kv, g in d.groupby(keys, dropna=False):
            v = g["normalized_l2_pooled"].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            st = block_statistics(v)
            lo, hi = bootstrap_ci(v, seed=20260730)
            rec = dict(zip(keys, kv))
            rec.update(gate="gate04", metric_version=METRIC_VERSION,
                       source_estimator_class=cls, n_blocks=st["n_blocks"],
                       normalized_mean=st["mean"], normalized_sd=st["sd"],
                       normalized_stderr=st["stderr"],
                       normalized_boot_ci_low=lo, normalized_boot_ci_high=hi,
                       normalized_max=float(v.max()),
                       max_absolute_mean=float(g["max_absolute"].mean()))
            out.append(stamper.stamp(rec, estimator_class=cls, seed="multi"))
            # leave-one-block-out: no single block may drive the result
            seeds = g["seed_name"].to_numpy()
            for i, s in enumerate(seeds):
                keep = np.delete(v, i) if v.size == seeds.size else v
                if keep.size < 2:
                    continue
                stk = block_statistics(keep)
                lrec = dict(zip(keys, kv))
                lrec.update(gate="gate04", metric_version=METRIC_VERSION,
                            source_estimator_class=cls, dropped_block=str(s),
                            n_blocks_kept=stk["n_blocks"],
                            normalized_mean_full=st["mean"],
                            normalized_mean_lobo=stk["mean"],
                            normalized_stderr_lobo=stk["stderr"],
                            shift_in_stderr=((stk["mean"] - st["mean"])
                                             / st["stderr"]
                                             if st["stderr"] > 0 else float("nan")))
                lobo.append(stamper.stamp(lrec, estimator_class=cls, seed=str(s)))
        return out, lobo

    det_sum, det_lobo = summarize(det_blocks, "deterministic_finite_feature", [])
    sam_sum, sam_lobo = summarize(sam_blocks, "sampled_finite_network",
                                  ["n_networks"])
    write_rows(outdir / "deterministic_ward_normalized_summary.csv", det_sum,
               required=["gate", "architecture", "n_features",
                         "normalized_mean", "normalized_stderr", "n_blocks"],
               sort_by=["regulator_class", "architecture", "n_features",
                        "configuration"])
    write_rows(outdir / "sampled_ward_normalized_summary.csv", sam_sum,
               required=["gate", "architecture", "n_features", "n_networks",
                         "normalized_mean", "normalized_stderr", "n_blocks"],
               sort_by=["regulator_class", "architecture", "n_features",
                        "n_networks", "configuration"])
    write_rows(outdir / "sampled_ward_leave_one_block_out.csv",
               sam_lobo + det_lobo,
               required=["gate", "source_estimator_class", "architecture",
                         "dropped_block", "normalized_mean_full",
                         "normalized_mean_lobo", "shift_in_stderr"],
               sort_by=["source_estimator_class", "regulator_class",
                        "architecture", "n_features", "configuration",
                        "dropped_block"])

    # ---- 4. paired architecture ratio, at identical geometry ------------
    # Ratios are formed BLOCK BY BLOCK at the same configuration, width and
    # seed, so the geometry and the draw cancel; an unpaired ratio of pooled
    # means would not control for either.
    pair_rows = []
    for cls, rows, extra in (("deterministic_finite_feature", det_blocks, []),
                             ("sampled_finite_network", sam_blocks,
                              ["n_networks"])):
        d = pd.DataFrame(rows)
        keys = ["regulator_class", "configuration", "n_features"] + extra + \
               ["seed_name"]
        base = d[d.architecture == "selected_log_kernel"].set_index(keys)
        others = d[d.architecture != "selected_log_kernel"]
        groups = [((a, "all"), g) for a, g in others.groupby("architecture")]
        groups += [((a, int(n)), g) for (a, n), g
                   in others.groupby(["architecture", "n_features"])]
        for (arch, width), g in groups:
            ratios, kept = [], []
            for _, r in g.iterrows():
                k = tuple(r[x] for x in keys)
                if k not in base.index:
                    continue
                b = base.loc[k]
                b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
                if not np.isfinite(b["normalized_l2_pooled"]) or \
                        b["normalized_l2_pooled"] <= 0:
                    continue
                ratios.append(r["normalized_l2_pooled"] / b["normalized_l2_pooled"])
                kept.append(k)
            if not ratios:
                continue
            v = np.asarray(ratios, dtype=float)
            st = block_statistics(v)
            lo, hi = bootstrap_ci(v, seed=20260730)
            pair_rows.append(stamper.stamp(dict(
                gate="gate04", source_estimator_class=cls, architecture=arch,
                width_subset=str(width),
                baseline="selected_log_kernel", metric_version=METRIC_VERSION,
                n_paired=int(v.size), ratio_mean=st["mean"],
                ratio_sd=st["sd"], ratio_stderr=st["stderr"],
                ratio_median=float(np.median(v)),
                ratio_boot_ci_low=lo, ratio_boot_ci_high=hi,
                ratio_min=float(v.min()), ratio_max=float(v.max()),
                worse_than_selected_in_all_pairs=bool(np.all(v > 1.0)),
                fraction_worse=float(np.mean(v > 1.0))),
                estimator_class=cls, seed="multi"))
    write_rows(outdir / "ward_paired_ratio.csv", pair_rows, required=[
        "gate", "source_estimator_class", "architecture", "width_subset",
        "baseline", "n_paired", "ratio_mean", "ratio_stderr",
        "worse_than_selected_in_all_pairs"],
        sort_by=["source_estimator_class", "architecture", "width_subset"])

    # ---- 4b. paired ratio at the INDEPENDENT replicate level ------------
    # The 48 pairs behind a cell are 2 regulator classes x 3 configurations x
    # 8 seeds. Configurations and regulator classes are different measurements
    # on the SAME draw, not independent replicates: only the seed is. Treating
    # "worse in 48/48" as 48 Bernoulli trials would overstate the evidence by
    # about a factor of six in effective sample size. Here the log-ratio is
    # averaged within each seed first, and the uncertainty is taken across
    # seeds.
    from scipy import stats as _st
    blk_rows = []
    for cls, rows_, extra in (("deterministic_finite_feature", det_blocks, []),
                              ("sampled_finite_network", sam_blocks,
                               ["n_networks"])):
        d = pd.DataFrame(rows_)
        keys = ["regulator_class", "configuration", "n_features"] + extra + \
               ["seed_name"]
        base = d[d.architecture == "selected_log_kernel"].set_index(keys)
        for (rc, arch, nf), g in d[
                d.architecture != "selected_log_kernel"].groupby(
                ["regulator_class", "architecture", "n_features"]):
            per_seed = {}
            for _, r in g.iterrows():
                k = tuple(r[x] for x in keys)
                if k not in base.index:
                    continue
                b = base.loc[k]
                b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
                bv = float(b["normalized_l2_pooled"])
                rv = float(r["normalized_l2_pooled"])
                if not (np.isfinite(bv) and np.isfinite(rv) and bv > 0 and rv > 0):
                    continue
                per_seed.setdefault(r["seed_name"], []).append(np.log(rv / bv))
            if len(per_seed) < 3:
                continue
            seed_means = np.array([np.mean(v) for v in per_seed.values()])
            n_seeds = seed_means.size
            m = float(seed_means.mean())
            se = float(seed_means.std(ddof=1) / np.sqrt(n_seeds))
            t = m / se if se > 0 else np.nan
            pval = (float(2 * _st.t.sf(abs(t), n_seeds - 1))
                    if np.isfinite(t) else np.nan)
            n_pairs = sum(len(v) for v in per_seed.values())
            blk_rows.append(stamper.stamp(dict(
                gate="gate04", source_estimator_class=cls,
                spectral_regulator_class=rc, architecture=arch,
                baseline="selected_log_kernel", n_features=int(nf),
                metric_version=METRIC_VERSION,
                n_independent_blocks=int(n_seeds),
                n_raw_pairs=int(n_pairs),
                pairs_per_block=float(n_pairs / n_seeds),
                independence_note=("seeds are the independent replicates; "
                                   "configurations and regulator classes are "
                                   "repeated measurements on the same draw"),
                mean_log_ratio=m, mean_log_ratio_stderr=se,
                ratio_geometric_mean=float(np.exp(m)),
                ratio_ci_low=float(np.exp(m - 1.96 * se)),
                ratio_ci_high=float(np.exp(m + 1.96 * se)),
                t_statistic=float(t), dof=int(n_seeds - 1), p_value=pval,
                worse_than_selected=bool(m > 0),
                significant_at_5pct=bool(np.isfinite(pval) and pval < 0.05),
                n_blocks_worse=int((seed_means > 0).sum()),
                naive_pair_count_worse=int(sum(
                    1 for v in per_seed.values() for x in v if x > 0)),
                naive_binomial_p_DO_NOT_USE=float(
                    2 ** (-n_pairs) if n_pairs == sum(
                        1 for v in per_seed.values() for x in v if x > 0)
                    else np.nan)),
                estimator_class=cls, seed="multi"))
    write_rows(outdir / "ward_paired_block_level.csv", blk_rows, required=[
        "gate", "source_estimator_class", "architecture", "n_features",
        "n_independent_blocks", "n_raw_pairs", "ratio_geometric_mean",
        "p_value", "independence_note"],
        sort_by=["source_estimator_class", "spectral_regulator_class",
                 "architecture", "n_features"])

    # ---- 5. is the sampled Ward error measuring anything but noise? -----
    # If E_norm ~ M^{-1/2} with no floor, the sampled campaign is measuring
    # Monte Carlo variance, not Ward-identity violation, and cannot discriminate
    # architectures at these M. Only the selected architecture was run over a
    # range of M (the controls were run at the largest M only), so the scaling
    # can be fitted for it alone -- which is itself worth recording.
    mc_rows = []
    sb = pd.DataFrame(sam_blocks)
    for kv, g in sb.groupby(["regulator_class", "architecture", "configuration",
                             "n_features"], dropna=False):
        agg = g.groupby("n_networks")["normalized_l2_pooled"].mean()
        agg = agg[np.isfinite(agg.values) & (agg.values > 0)]
        rec = dict(zip(["regulator_class", "architecture", "configuration",
                        "n_features"], kv))
        rec.update(gate="gate04", metric_version=METRIC_VERSION,
                   n_m_values=int(agg.size),
                   m_values=";".join(str(int(x)) for x in agg.index))
        if agg.size >= 2:
            b, a = np.polyfit(np.log(agg.index.values.astype(float)),
                              np.log(agg.values), 1)
            resid = np.log(agg.values) - (b * np.log(agg.index.values) + a)
            ss = float(np.sum(resid**2))
            tot = float(np.sum((np.log(agg.values) - np.log(agg.values).mean())**2))
            rec.update(m_scaling_exponent=float(b),
                       m_scaling_r2=float(1.0 - ss / tot) if tot > 0 else float("nan"),
                       consistent_with_pure_monte_carlo=bool(abs(b + 0.5) <= 0.12),
                       e_norm_at_largest_m=float(agg.iloc[-1]),
                       largest_m=int(agg.index[-1]))
            # M needed for the normalized error to drop below 1, at this rate
            rec["m_required_for_e_norm_below_1"] = float(
                agg.index[-1] * agg.iloc[-1] ** (-1.0 / b)) if b < 0 else float("nan")
        else:
            rec.update(m_scaling_exponent=float("nan"), m_scaling_r2=float("nan"),
                       consistent_with_pure_monte_carlo=None,
                       e_norm_at_largest_m=float(agg.iloc[-1]) if agg.size else
                       float("nan"),
                       largest_m=int(agg.index[-1]) if agg.size else -1,
                       m_required_for_e_norm_below_1=float("nan"))
        mc_rows.append(stamper.stamp(rec,
                                     estimator_class="sampled_finite_network",
                                     seed="multi"))
    write_rows(outdir / "sampled_ward_monte_carlo_scaling.csv", mc_rows, required=[
        "gate", "architecture", "configuration", "n_features",
        "m_scaling_exponent", "consistent_with_pure_monte_carlo",
        "metric_version"],
        sort_by=["regulator_class", "architecture", "n_features", "configuration"])

    # ---- 6. figure: what the relabelling costs --------------------------
    figpath = crosswalk_figure(det, figdir)

    fitted = [r for r in mc_rows if np.isfinite(r["m_scaling_exponent"])]
    if fitted:
        ex = np.array([r["m_scaling_exponent"] for r in fitted])
        npure = sum(1 for r in fitted if r["consistent_with_pure_monte_carlo"])
        print(f"  sampled M-scaling: exponent {ex.mean():+.3f} +/- {ex.std(ddof=1):.3f} "
              f"over {len(fitted)} cells; {npure}/{len(fitted)} consistent with "
              f"pure Monte Carlo (M^-1/2)")
    for r in blk_rows:
        if (r["architecture"] == "rbf_gaussian_prior"
                and r["n_features"] == 200000
                and r["spectral_regulator_class"] == "tapered_log_window"):
            print(f"  RBF vs selected @N=2e5 (tapered): ratio "
                  f"{r['ratio_geometric_mean']:.2f} "
                  f"[{r['ratio_ci_low']:.2f}, {r['ratio_ci_high']:.2f}], "
                  f"{r['n_independent_blocks']} independent blocks "
                  f"(not {r['n_raw_pairs']} pairs), p={r['p_value']:.2e}")
    print(f"  crosswalk {len(cross)} rows; blocks det={len(det_blocks)} "
          f"sam={len(sam_blocks)}; summaries det={len(det_sum)} sam={len(sam_sum)}; "
          f"lobo={len(sam_lobo) + len(det_lobo)}; paired={len(pair_rows)}")
    print(f"  figure {figpath.name}")
    for r in cross[:4]:
        print(f"    quoted {r['quoted_value']:.6g} ({r['architecture']}, "
              f"N={r['n_features']}): max_absolute={r['actual_value']:.6g}, "
              f"normalized={r['principal_normalized_value']:.4g} "
              f"({r['normalized_over_quoted']:.1f}x the quoted number)")
    return 0


def crosswalk_figure(det, figdir):
    """Absolute vs normalized Ward error against width, per architecture."""
    d = det[det.regulator_class == "tapered_log_window"].copy()
    fig, (ax, az) = PS.new_figure(1, 2, figsize=(9.6, 4.3))
    order = ["selected_log_kernel", "local_exponent_direction",
             "rbf_gaussian_prior", "angular_mode_scramble"]
    m = PS.MARKERS["deterministic_finite_feature"]
    for i, arch in enumerate(order):
        g = d[d.architecture == arch]
        if not len(g):
            continue
        nf = np.array(sorted(g.n_features.unique()), dtype=float)
        absmax = [g[g.n_features == n].stress_absolute_error.max() for n in nf]
        norm = [e_norm(g[g.n_features == n]) for n in nf]
        lbl = arch.replace("_", " ")
        ax.plot(nf, absmax, m + "-", color=PS.COLORS[i], ms=5, lw=1.2, label=lbl)
        az.plot(nf, norm, m + "-", color=PS.COLORS[i], ms=5, lw=1.2, label=lbl)
    for a, t, yl in ((ax, "max ABSOLUTE stress-Ward error\n(what the numbers are)",
                      r"$\max_g |T_{\rm est}-T_{\rm tgt}|$"),
                     (az, "NORMALIZED stress-Ward error\n(what they are called)",
                      r"$E_{\rm norm}$")):
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("$N$ (features)"); a.set_ylabel(yl)
        a.set_title(t, fontsize=9)
        a.axhline(1.0, color="0.55", lw=1.0, ls=":")
    ax.annotate("quoted as 0.129", xy=(200000, 0.129304), xytext=(-14, 24),
                textcoords="offset points", fontsize=8, ha="right",
                color=PS.COLORS[0],
                arrowprops=dict(arrowstyle="->", color=PS.COLORS[0], lw=0.8))
    az.annotate("same setting: 0.92", xy=(200000, 0.918826), xytext=(-14, 26),
                textcoords="offset points", fontsize=8, ha="right",
                color=PS.COLORS[0],
                arrowprops=dict(arrowstyle="->", color=PS.COLORS[0], lw=0.8))
    ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Gate 4: the Ward numbers are absolute, not normalized "
                 "(deterministic finite feature, tapered)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return PS.save(fig, "gate04_ward_absolute_vs_normalized.pdf", outdir=figdir)


if __name__ == "__main__":
    raise SystemExit(main())
