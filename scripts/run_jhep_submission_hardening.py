#!/usr/bin/env python3
"""Master runner for the JHEP submission-hardening release.

Runs every gate in dependency order from a clean committed tree, then builds the
release bookkeeping and verifies release integrity.

    python scripts/run_jhep_submission_hardening.py            # full production
    python scripts/run_jhep_submission_hardening.py --quick    # smoke path
    python scripts/run_jhep_submission_hardening.py --only-gate gate03,gate08
    python scripts/run_jhep_submission_hardening.py --skip-existing

Gate order is a dependency order, not a numbering: Gate 6 pins the conventions
every later gate inherits and therefore runs first; Gate 2 post-processes the
pairing matrices Gate 1 persists; Gate 3's q-ratio needs the direct extension's
per-separation rows; Gate 8 pools the fits emitted by Gates 1, 2 and 7.

This script never touches ``results/jhep_frozen/`` or ``figures_jhep_frozen/``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro.config import load_config                  # noqa: E402
from nnft_virasoro.provenance import (file_sha256, git_commit,  # noqa: E402
                                      git_dirty, utc_now)

SUITE = "jhep_virasoro_submission_hardening"
SUITE_VERSION = 1

#: (gate id, script, extra args). Dependency order -- see the module docstring.
STAGES = [
    ("gate06", "scripts/run_gate06_normalization.py", []),
    ("gate01", "scripts/run_gate01_intrinsic.py", []),
    ("gate02", "scripts/run_gate02_pbw.py", []),
    ("gate03", "scripts/run_gate03_tt.py", []),
    ("gate03_extension", "scripts/run_gate03_direct_extension.py", []),
    ("gate03_qratio", "scripts/run_gate03_qratio.py", []),
    ("gate04", "scripts/run_gate04_ward.py", []),
    ("gate07", "scripts/run_gate07_ns.py", []),
    ("gate08", "scripts/run_gate08_synthesis.py", []),
    ("ledger", "scripts/build_hardening_ledger.py", []),
]

#: Artifacts that must exist at the end of a full production run.
EXPECTED = [
    "gate06_kernel_normalization_checks.csv",
    "gate01_intrinsic_admissibility_seed_rows.csv",
    "gate01_intrinsic_admissibility_fits.csv",
    "gate02_predarboux_virasoro_seed_rows.csv",
    "gate02_predarboux_virasoro_fits.csv",
    "gate02_stability_epsilon_sweep.csv",
    "tt_estimator_provenance.csv",
    "sampled_tt_wick_blocks.csv",
    "sampled_tt_direct_blocks.csv",
    "sampled_tt_estimator_comparison.csv",
    "gate03_cumulant_identity.csv",
    "gate03_fit_weighting_crosswalk.csv",
    "gate03_regulator_convergence.csv",
    "sampled_tt_direct_extension_blocks.csv",
    "sampled_tt_direct_extension_summary.csv",
    "sampled_tt_direct_extension_points.csv",
    "gate03_q_ratio_blocks.csv",
    "gate03_q_ratio_test.csv",
    "ward_metric_crosswalk.csv",
    "deterministic_ward_normalized_blocks.csv",
    "sampled_ward_normalized_blocks.csv",
    "deterministic_ward_normalized_summary.csv",
    "sampled_ward_normalized_summary.csv",
    "sampled_ward_leave_one_block_out.csv",
    "sampled_ward_monte_carlo_scaling.csv",
    "ward_paired_ratio.csv",
    "ward_paired_block_level.csv",
    "ns_kernel_seed_rows.csv",
    "ns_pairing_seed_rows.csv",
    "ns_super_virasoro_seed_rows.csv",
    "ns_super_virasoro_fits.csv",
    "intrinsic_stationarity_seed_rows.csv",
    "convergence_joint_fit.csv",
    "architecture_synthesis.csv",
    "hardening_claim_ledger.csv",
]

EXPECTED_FIGURES = [
    "gate03_tt_estimator_comparison.pdf",
    "gate04_ward_absolute_vs_normalized.pdf",
    "gate08_intrinsic_symmetry_audit.pdf",
]


def _rel(p: Path) -> str:
    """Path relative to the repo root when it lies inside it, else absolute.

    The production directories are inside the tree; a smoke run may point the
    output elsewhere, and that must not crash the attribution snapshot.
    """
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _snapshot(dirs):
    out = {}
    for d in dirs:
        for p in Path(d).rglob("*"):
            if p.is_file():
                out[_rel(p)] = p.stat().st_mtime_ns
    return out


def run_stage(gate, script, extra, args, commands, produced):
    cmd = [sys.executable, str(ROOT / script), "--config", str(args.config),
           "--output-dir", str(args.output_dir),
           "--figure-dir", str(args.figure_dir)] + list(extra)
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.quick:
        cmd.append("--quick")
    pretty = " ".join(["python", script, "--config", str(args.config),
                       "--output-dir", str(args.output_dir),
                       "--figure-dir", str(args.figure_dir)] + list(extra)
                      + (["--skip-existing"] if args.skip_existing else [])
                      + (["--quick"] if args.quick else []))
    commands.append(pretty)
    before = _snapshot([args.output_dir, args.figure_dir])
    print(f"\n=== {gate}: {script} ===", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if res.returncode != 0:
        raise SystemExit(f"{gate} failed with exit code {res.returncode}")
    after = _snapshot([args.output_dir, args.figure_dir])
    for path, mt in after.items():
        if before.get(path) != mt:
            produced[path] = dict(gate=gate, script=script, command=pretty)
    print(f"--- {gate} finished in {dt / 60:.2f} min", flush=True)
    return dt


# --------------------------------------------------------------------------
def build_all_metrics(outdir) -> pd.DataFrame:
    """Wide union of every gate CSV, tagged by source artifact."""
    frames = []
    for p in sorted(Path(outdir).glob("*.csv")):
        if p.name in ("all_metrics.csv", "hardening_claim_ledger.csv"):
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        df = df.copy()
        df.insert(0, "source_artifact", p.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)








def write_environment(outdir):
    import scipy
    import matplotlib
    lines = [
        f"generated_utc: {utc_now()}",
        f"python: {sys.version.split()[0]} ({platform.python_implementation()})",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"cpu_count: {os.cpu_count()}",
        f"numpy: {np.__version__}",
        f"scipy: {scipy.__version__}",
        f"pandas: {pd.__version__}",
        f"matplotlib: {matplotlib.__version__}",
        "",
        "# Thread pinning used by the parallel stages (see",
        "# scripts/run_gate03_direct_extension.py): the numerics are unaffected,",
        "# verified bit-identical with and without pinning.",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
    ]
    (Path(outdir) / "environment.txt").write_text("\n".join(lines) + "\n")




def main() -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--config", type=Path,
                   default=ROOT / "configs" / "jhep_submission_hardening.yaml")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "jhep_submission_hardening")
    p.add_argument("--figure-dir", type=Path,
                   default=ROOT / "figures_jhep_submission_hardening")
    p.add_argument("--quick", action="store_true",
                   help="smoke path: every stage runs its reduced grid")
    p.add_argument("--only-gate", default="",
                   help="comma-separated stage ids, e.g. gate03,gate08")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    for d in (args.output_dir, args.figure_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    only = [s.strip() for s in args.only_gate.split(",") if s.strip()]
    stages = [s for s in STAGES if not only or s[0] in only]
    if only and not stages:
        raise SystemExit(f"--only-gate {args.only_gate!r} matched no stage; "
                         f"known: {', '.join(s[0] for s in STAGES)}")

    cfg = load_config(args.config)
    commit = git_commit()
    print(f"suite      : {SUITE}")
    print(f"commit     : {commit}")
    print(f"config     : {args.config.name}  hash {cfg.hash}")
    print(f"dirty code : {git_dirty()}")
    print(f"stages     : {', '.join(s[0] for s in stages)}"
          f"{'  [QUICK]' if args.quick else ''}")

    commands, produced, timings = [], {}, {}
    t0 = time.time()
    for gate, script, extra in stages:
        timings[gate] = run_stage(gate, script, extra, args, commands, produced)
    wall = time.time() - t0

    if only:
        print(f"\n--only-gate run finished in {wall / 60:.1f} min; "
              f"bookkeeping skipped")
        return 0

    # ---------------- release bookkeeping ------------------------------
    print("\n=== release bookkeeping ===", flush=True)
    outdir = Path(args.output_dir)
    am = build_all_metrics(outdir)
    am.to_csv(outdir / "all_metrics.csv", index=False)
    print(f"  all_metrics.csv: {len(am)} rows x {am.shape[1]} columns")

    write_environment(outdir)

    sh = ["#!/usr/bin/env bash",
          "# Exact commands for the submission-hardening release.",
          f"# commit {commit}", f"# config hash {cfg.hash}",
          "set -euo pipefail", ""] + commands + [
          "", "# release bookkeeping and integrity check are performed by",
          "# scripts/run_jhep_submission_hardening.py after the stages above."]
    (outdir / "run_commands.sh").write_text("\n".join(sh) + "\n")
    os.chmod(outdir / "run_commands.sh", 0o755)

    ledger = pd.read_csv(outdir / "hardening_claim_ledger.csv").to_dict("records")
    artifacts = []
    for base in (outdir, Path(args.figure_dir)):
        for f in sorted(Path(base).rglob("*")):
            if not f.is_file() or f.name == "manifest.json":
                continue
            rel = _rel(f)
            src = produced.get(rel, {})
            artifacts.append(dict(
                path=rel, bytes=f.stat().st_size, sha256=file_sha256(f),
                generated_by=src.get("script", "scripts/run_jhep_submission_"
                                               "hardening.py (bookkeeping)"),
                stage=src.get("gate", "bookkeeping"),
                command=src.get("command", "release bookkeeping phase"),
                config_hash=cfg.hash, git_commit=commit))
    manifest = dict(
        suite=SUITE, suite_version=SUITE_VERSION, generated_utc=utc_now(),
        git_commit=commit, git_dirty_code=git_dirty(), config=str(
            args.config.relative_to(ROOT)), config_hash=cfg.hash,
        quick=bool(args.quick),
        estimator_classes=sorted(
            am.estimator_class.dropna().unique().tolist())
        if "estimator_class" in am else [],
        gates=[s[0] for s in STAGES],
        stage_wall_seconds={k: round(v, 2) for k, v in timings.items()},
        production_wall_seconds=round(wall, 2),
        n_ledger_rows=len(ledger), n_all_metrics_rows=int(len(am)),
        artifacts=artifacts)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"  manifest.json: {len(artifacts)} artifacts")

    missing = [name for name in EXPECTED if not (outdir / name).is_file()]
    missing += [name for name in EXPECTED_FIGURES
                if not (Path(args.figure_dir) / name).is_file()]
    if missing:
        raise SystemExit("Missing numerical outputs: " + ", ".join(missing))
    print(f"Numerical campaign finished in {wall / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
