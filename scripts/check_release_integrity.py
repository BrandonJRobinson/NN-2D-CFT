#!/usr/bin/env python3
"""Release-integrity gate for the submission-hardening campaign.

Exits non-zero if any release-blocking condition holds. Every check prints its
verdict, so a failure names the condition rather than just the exit code.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro.provenance import file_sha256                # noqa: E402

#: Collaborator-owned alpha-sweep artifacts. None may appear in the release,
#: and nothing in the release may reference them (mission rule 5).
FORBIDDEN = [
    "current_kernel_values.csv",
    "current_exponent_fits.csv",
    "current_conformal_error_vs_alpha.png",
    "current_exponent_vs_alpha.png",
    "current_kernel_powerlaws_by_alpha.png",
]

ALLOWED_STATUSES = {"accepted", "accepted_with_reformulation",
                    "needs_new_computation", "fails_current_claim",
                    "diagnostic_only"}

NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")

#: The write-ups are prose and use typographic minus signs. Normalize them
#: before parsing, or a quoted "-0.4991" reads as +0.4991 and the anchor
#: silently compares against the wrong number.
_DASHES = {"\u2212": "-", "\u2013": "-", "\u2014": "-", "\u2010": "-",
           "\u2011": "-", "\u00ad": ""}


class Checker:
    def __init__(self):
        self.failures = []
        self.n = 0

    def check(self, ok, label, detail=""):
        self.n += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
              + (f"  -- {detail}" if detail and not ok else ""))
        if not ok:
            self.failures.append(f"{label}: {detail}")
        return ok


def _numbers(text):
    for a, b in _DASHES.items():
        text = text.replace(a, b)
    return [float(m) for m in NUM.findall(text)]


def anchor(ck, doc_path, value, tol, label):
    """A number the write-up quotes must match the clean-run CSV it came from."""
    if not doc_path.exists():
        return ck.check(False, f"anchor {label}", f"missing {doc_path.name}")
    nums = _numbers(doc_path.read_text())
    if not nums:
        return ck.check(False, f"anchor {label}", "no numbers in document")
    best = min(nums, key=lambda x: abs(x - value))
    return ck.check(abs(best - value) <= tol, f"anchor {label}",
                    f"{doc_path.name} has no number within {tol} of "
                    f"{value:.6g} (closest {best:.6g})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "results" / "jhep_submission_hardening")
    p.add_argument("--figure-dir", type=Path,
                   default=ROOT / "figures_jhep_submission_hardening")
    args = p.parse_args()
    outdir, figdir = Path(args.output_dir), Path(args.figure_dir)
    ck = Checker()

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_jhep_submission_hardening import EXPECTED, EXPECTED_FIGURES

    # ---- 1-3: single commit, single config hash, no dirty rows ---------
    hashes, commits, dirty = set(), set(), set()
    csvs = [p for p in sorted(outdir.glob("*.csv"))]
    for f in csvs:
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "config_hash" in d:
            hashes |= set(d.config_hash.dropna().astype(str).unique())
        if "git_commit" in d:
            commits |= set(d.git_commit.dropna().astype(str).unique())
        if "git_dirty" in d:
            dirty |= set(str(x) for x in d.git_dirty.dropna().unique())
    ck.check(len(hashes) == 1, "exactly one config hash", f"found {sorted(hashes)}")
    ck.check(len(commits) == 1, "exactly one git commit",
             f"found {sorted(c[:12] for c in commits)}")
    ck.check(dirty <= {"False"}, "no git_dirty=True row", f"found {sorted(dirty)}")

    # ---- 4: no needs_new_computation, and no obsolete vocabulary -------
    led_path = outdir / "hardening_claim_ledger.csv"
    if ck.check(led_path.exists(), "claim ledger present"):
        led = pd.read_csv(led_path)
        bad = set(led.status.unique()) - ALLOWED_STATUSES
        ck.check(not bad, "only adjudicated statuses", f"unexpected {sorted(bad)}")
        ck.check("failed_gate" not in set(led.status.unique()),
                 "no obsolete failed_gate vocabulary")
        nnc = int((led.status == "needs_new_computation").sum())
        ck.check(nnc == 0, "no needs_new_computation claim", f"{nnc} rows")

    # ---- 5: every expected gate artifact exists and is non-empty -------
    missing = [n for n in EXPECTED if not (outdir / n).exists()
               or (outdir / n).stat().st_size == 0]
    ck.check(not missing, "all expected gate artifacts present",
             f"missing/empty: {missing}")
    for extra in ("all_metrics.csv", "manifest.json", "environment.txt",
                  "run_commands.sh", "summary.md", "GATE_MATRIX.md"):
        ck.check((outdir / extra).exists(), f"bookkeeping artifact {extra}")

    # ---- 6-7: figures are from this run, not carried over --------------
    man_path = outdir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    by_path = {a["path"]: a for a in manifest.get("artifacts", [])}
    csv_mtimes = [f.stat().st_mtime for f in csvs]
    oldest_csv = min(csv_mtimes) if csv_mtimes else 0.0
    for fig in EXPECTED_FIGURES:
        fp = figdir / fig
        if not ck.check(fp.exists(), f"figure present: {fig}"):
            continue
        ck.check(fp.stat().st_mtime >= oldest_csv,
                 f"figure not stale: {fig}",
                 "predates the release CSVs, so it was carried over")
        try:
            key = str(fp.resolve().relative_to(ROOT))
        except ValueError:
            key = str(fp.resolve())
        rec = by_path.get(key, {})
        ck.check(rec.get("stage", "bookkeeping") != "bookkeeping",
                 f"figure attributed to a gate stage: {fig}",
                 f"stage={rec.get('stage')!r}")
        ck.check(rec.get("sha256") == file_sha256(fp),
                 f"figure hash matches manifest: {fig}")

    # ---- 8: the August frozen release is untouched ---------------------
    st = subprocess.run(["git", "status", "--porcelain", "--",
                         "results/jhep_frozen", "figures_jhep_frozen",
                         "tables_jhep_frozen"],
                        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    ck.check(st == "", "frozen namespace has no uncommitted change", st[:200])
    fman = ROOT / "results" / "jhep_frozen" / "manifest.json"
    if fman.exists():
        fm = json.loads(fman.read_text())
        broken = []
        for a in fm.get("artifacts", []):
            fp = ROOT / a["path"]
            if not fp.exists() or file_sha256(fp) != a["sha256"]:
                broken.append(a["path"])
        ck.check(not broken, "frozen artifacts match their recorded SHA-256",
                 f"{len(broken)} differ: {broken[:3]}")
        ck.check(fm.get("git_commit", "").startswith("b3fd73dd887b"),
                 "frozen release still points at b3fd73dd887b",
                 fm.get("git_commit", "")[:12])
        ck.check(fm.get("config_hash") == "278e438b94465564",
                 "frozen config hash unchanged", str(fm.get("config_hash")))

    # ---- 9: no collaborator-owned alpha-sweep artifact -----------------
    present = [n for n in FORBIDDEN
               if list(outdir.rglob(n)) or list(figdir.rglob(n))]
    ck.check(not present, "no collaborator alpha-sweep artifact in the release",
             f"{present}")
    referenced = []
    for f in csvs:
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        for n in FORBIDDEN:
            if n in txt:
                referenced.append((f.name, n))
    ck.check(not referenced, "no release CSV references an alpha-sweep artifact",
             f"{referenced[:3]}")

    # ---- 10: Gate 8 stationarity never routes through a 1-arg helper ---
    ti_src = (ROOT / "src" / "nnft_virasoro" / "translation_invariance.py").read_text()
    g8_src = (ROOT / "scripts" / "run_gate08_synthesis.py").read_text()
    for name, src in (("translation_invariance.py", ti_src),
                      ("run_gate08_synthesis.py", g8_src)):
        code = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
        used = [h for h in ("empirical_scalar_kernel", "empirical_current_kernel")
                if h in code]
        ck.check(not used, f"no one-argument kernel helper in {name}",
                 f"uses {used}")
    ti_path = outdir / "intrinsic_stationarity_seed_rows.csv"
    if ck.check(ti_path.exists(), "stationarity rows present"):
        ti = pd.read_csv(ti_path)
        pc = ti[ti.architecture == "phase_correlated_features"]
        ck.check(len(pc) > 0 and not pc.phase_pairing_is_trivial.any(),
                 "phase_correlated uses its true two-argument pairing",
                 "idx_a == idx_b, so a one-argument helper was used")
        ck.check(len(pc) > 0 and (pc.scalar_translation_defect > 0.1).all(),
                 "phase_correlated stationarity defect is resolved",
                 "defect ~0 would mean the helper flattened it")
        indep = ti[ti.phase_pairing_is_trivial]
        ck.check((indep.scalar_translation_defect < 1e-12).all(),
                 "independent-phase architectures are exactly stationary",
                 f"max {indep.scalar_translation_defect.max():.3e}")

    # ---- numeric anchors: write-ups must match the clean-run CSVs ------
    def rd(n):
        p = outdir / n
        return pd.read_csv(p) if p.exists() else None
    D = outdir

    g1 = rd("gate01_intrinsic_admissibility_fits.csv")
    if g1 is not None:
        t = g1[g1.spectral_regulator_class == "tapered_log_window"].set_index(
            "architecture")
        anchor(ck, D / "gate01_intrinsic_admissibility_summary.md",
               float(t.loc["rbf_gaussian_prior"].delta_iso_at_max_width), 5e-3,
               "gate01 RBF delta_iso")
    sw = rd("gate02_stability_epsilon_sweep.csv")
    if sw is not None:
        ref = sw[sw.direction_index == 0].residual_over_pairing_plus_square
        anchor(ck, D / "gate02_predarboux_virasoro_summary.md",
               float(ref.min()), 5e-4, "gate02 epsilon-sweep plateau low")
        anchor(ck, D / "gate02_predarboux_virasoro_summary.md",
               float(ref.max()), 5e-4, "gate02 epsilon-sweep plateau high")
    qt = rd("gate03_q_ratio_test.csv")
    if qt is not None:
        q = qt.set_index("estimator_variant")
        anchor(ck, D / "gate03_tt_summary.md",
               float(q.loc["direct_TT"].gls_common_ratio), 5e-4,
               "gate03 direct q ratio")
        anchor(ck, D / "gate03_tt_summary.md",
               float(q.loc["wick_factorized"].gls_common_ratio), 5e-5,
               "gate03 Wick q ratio")
    ex = rd("sampled_tt_direct_extension_summary.csv")
    if ex is not None:
        e = ex.set_index("estimator_variant")
        anchor(ck, D / "gate03_tt_summary.md", float(e.loc["direct_TT"].c), 5e-4,
               "gate03 direct TT normalization")
        anchor(ck, D / "gate03_tt_summary.md",
               float(e.loc["wick_factorized"].c), 5e-4,
               "gate03 Wick TT normalization")
    bl = rd("ward_paired_block_level.csv")
    if bl is not None:
        s = bl[(bl.architecture == "rbf_gaussian_prior")
               & (bl.n_features == 200000)
               & (bl.spectral_regulator_class == "tapered_log_window")]
        if len(s):
            anchor(ck, D / "gate04_ward_summary.md",
                   float(s.iloc[0].ratio_geometric_mean), 5e-2,
                   "gate04 block-level RBF ratio")
    g6 = rd("gate06_kernel_normalization_checks.csv")
    if g6 is not None:
        graded = g6[g6.check_kind != "diagnostic"]
        anchor(ck, D / "gate06_normalization_crosswalk.md",
               float(len(graded)), 0.5, "gate06 graded check count")
    alg = rd("ns_super_virasoro_seed_rows.csv")
    if alg is not None:
        big = alg[(alg.architecture == "selected_log_kernel")
                  & (alg.n_features == alg.n_features.max())]
        for cons, g in big.groupby("ns_construction"):
            anchor(ck, D / "gate07_ns_summary.md",
                   float(g.raw_central_charge_fit.mean()), 5e-4,
                   f"gate07 central charge [{cons}]")
    ti2 = rd("intrinsic_stationarity_seed_rows.csv")
    if ti2 is not None:
        pc = ti2[ti2.architecture == "phase_correlated_features"]
        anchor(ck, D / "gate08_synthesis_summary.md",
               float(pc.scalar_translation_defect.mean()), 5e-3,
               "gate08 phase-correlated stationarity defect")
    jf = rd("convergence_joint_fit.csv")
    if jf is not None:
        pooled = jf[jf.quantity == "POOLED_EXPONENT"]
        if len(pooled):
            for doc in ("gate08_synthesis_summary.md", "GATE_MATRIX.md"):
                anchor(ck, D / doc, float(pooled.iloc[0].exponent), 5e-4,
                       f"pooled exponent in {doc}")

    print(f"\n  {ck.n - len(ck.failures)}/{ck.n} checks passed")
    if ck.failures:
        print("\nRELEASE BLOCKED:")
        for f in ck.failures:
            print(f"  - {f}")
        return 1
    print("  release integrity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
