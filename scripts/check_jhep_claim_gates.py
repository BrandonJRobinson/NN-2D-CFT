#!/usr/bin/env python3
"""Submission checker for the JHEP Virasoro frozen release.

    python scripts/check_jhep_claim_gates.py --submission

Fails (exit 1) when any of the following holds:

  * a claim gate lacks an accepted ledger row;
  * a manuscript number is absent from the ledger;
  * an excluded later-paper citation reappears;
  * collaborator-owned exponent-sweep artifacts appear;
  * figures or tables are stale relative to their source CSV;
  * any frozen run has uncommitted code provenance;
  * tests fail.

Without ``--submission`` it reports the same information and exits 0, so
manuscript editing is never blocked.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro.config import load_config  # noqa: E402
from nnft_virasoro.ledger import validate_ledger  # noqa: E402
from nnft_virasoro.provenance import git_dirty  # noqa: E402

RESULTS = ROOT / "results" / "jhep_frozen"
FIGURES = ROOT / "figures_jhep_frozen"
TABLES = ROOT / "tables_jhep_frozen"

#: Manuscript claim-gate anchors and the ledger work package that must cover them.
GATE_WORKPACKAGE = {
    "FM-01": "WP-0", "FM-02": "WP-B", "FM-03": "WP-F", "FM-04": "WP-C",
    "FM-05": "WP-D", "FM-06": "WP-G", "FM-07": "WP-B", "FM-08": "WP-H",
    "FM-09": "WP-H", "FM-10": "WP-A", "FM-11": "WP-A", "FM-12": "WP-E",
}

#: Figures and the CSV they must not be older than.
FIGURE_SOURCES = {
    "ordered_pairing_radius_plateau.pdf": "ordered_pairing_grid.csv",
    "predarboux_pairing_vs_width.pdf": "predarboux_convergence_seed_rows.csv",
    "raw_virasoro_vs_width.pdf": "predarboux_convergence_seed_rows.csv",
    "predarboux_sampling_error_vs_width.pdf":
        "predarboux_convergence_seed_rows.csv",
    "architecture_controls_predarboux.pdf": "architecture_controls_seed_rows.csv",
    "architecture_controls_ward.pdf": "architecture_controls_seed_rows.csv",
    "architecture_controls_postdarboux.pdf": "architecture_controls_seed_rows.csv",
    "sampled_tt_central_charge.pdf": "sampled_tt_central_charge.csv",
    "sampled_ward_convergence.pdf": "sampled_ward_summary.csv",
    "ward_controls.pdf": "sampled_ward_summary.csv",
    "ns_pairing_convergence.pdf": "ns_frozen_metrics.csv",
    "ns_super_virasoro_residuals.pdf": "ns_frozen_metrics.csv",
    "raw_modewise_central_charge_offset_model.pdf":
        "central_offset_mode_extension.csv",
    "sampled_vertex_weights.pdf": "sampled_vertex_weights.csv",
    "finite_width_cumulant.pdf": "finite_width_cumulant.csv",
    "darboux_admissibility_vs_width.pdf": "darboux_physical_admissibility.csv",
}

NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_\\])(-?\d+\.\d{3,}(?:\s*\\times\s*10\^\{?-?\d+\}?)?"
    r"|-?\d\.\d+e-?\d+)")


def report(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return ok


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--submission", action="store_true")
    p.add_argument("--config", type=Path,
                   default=ROOT / "configs" / "jhep_frozen.yaml")
    p.add_argument("--manuscript", type=Path,
                   default=ROOT / "Draft-JHEP-Frozen.tex")
    p.add_argument("--bib", type=Path, default=ROOT / "Virasoro_JHEP_Frozen.bib")
    p.add_argument("--skip-tests", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    failures: list[str] = []
    print("JHEP frozen claim-gate check")
    print("=" * 78)

    # ---- 1. ledger present and schema-valid --------------------------
    ledger_path = RESULTS / "claim_ledger.csv"
    if not ledger_path.exists():
        failures.append("claim_ledger.csv missing")
        report(False, "claim ledger present")
        ledger = pd.DataFrame()
    else:
        ledger = pd.read_csv(ledger_path)
        problems = validate_ledger(ledger.to_dict("records"))
        if problems:
            failures.append(f"ledger schema: {problems[:3]}")
        report(not problems, "claim-ledger schema",
               f"{len(ledger)} rows" if not problems else str(problems[:2]))

    # ---- 2. every claim gate has an accepted ledger row ---------------
    accepted = set()
    if len(ledger):
        acc = ledger[ledger["status"].isin(["accepted", "accepted_with_caveat"])]
        accepted = set(acc["work_package"].unique())
    for gate, wp in sorted(GATE_WORKPACKAGE.items()):
        ok = wp in accepted
        if not ok:
            failures.append(f"gate {gate} ({wp}) has no accepted ledger row")
        report(ok, f"gate {gate} -> {wp}")

    # ---- 3. manuscript numbers resolve to the ledger ------------------
    if args.manuscript.exists():
        text = args.manuscript.read_text(encoding="utf-8", errors="replace")
        body = re.sub(r"%.*", "", text)
        # Front/back matter carries identifiers (grant numbers, DOIs, addresses),
        # not claims; the gate is about *quantitative results*.
        body = re.sub(r"\\acknowledgments.*?(?=\\appendix|\\bibliography)", "",
                      body, flags=re.S)
        body = re.sub(r"\\affiliation.*?\n\n", "", body, flags=re.S)
        # Every ledger text column is an admissible source for a manuscript number.
        ledger_blob = " ".join(
            " ".join(str(v) for v in ledger.get(col, []))
            for col in ("value", "uncertainty", "confidence_interval", "metric",
                        "notes", "seed_set"))
        # Compare numerically, not textually: the manuscript renders a ledger
        # value in LaTeX scientific notation, so "5.93394\\times10^{5}" must
        # resolve to the ledger's "593394.3".
        led_nums = []
        for tok in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", ledger_blob):
            try:
                led_nums.append(float(tok))
            except ValueError:
                pass
        led_nums = np.array(led_nums, dtype=float)

        def resolves(x: float, tol: float) -> bool:
            """Match to the precision the manuscript actually displays.

            The manuscript deliberately shows no more digits than the
            uncertainty supports, so a displayed value must agree with its
            ledger row to within half a unit in its last displayed place.
            """
            if led_nums.size == 0:
                return False
            return bool(np.any(np.abs(led_nums - x) <= tol))

        missing = []
        for m in set(NUMBER_RE.findall(body)):
            mant = m.split("\\times")[0].strip()
            exp = re.search(r"10\^\{?(-?\d+)\}?", m)
            try:
                scale = 10.0 ** int(exp.group(1)) if exp else 1.0
                val = float(mant) * scale
            except ValueError:
                continue
            decimals = len(mant.split(".")[1]) if "." in mant else 0
            tol = 0.5 * (10.0 ** -decimals) * scale * 1.000001
            if not (resolves(val, tol) or resolves(-val, tol)):
                missing.append(m.strip())
        ok = not missing
        if not ok:
            failures.append(f"manuscript numbers not in ledger: {sorted(missing)[:6]}")
        report(ok, "manuscript numbers resolve to ledger rows",
               f"{len(missing)} unresolved" if missing else "")
    else:
        report(True, "manuscript numbers (frozen manuscript not built yet)")

    # ---- 3b. single config hash across the whole frozen release --------
    hashes: dict[str, set] = {}
    for csv_path in sorted(RESULTS.glob("*.csv")):
        if csv_path.name in ("claim_ledger.csv", "all_metrics.csv"):
            continue
        try:
            col = pd.read_csv(csv_path, usecols=["config_hash"])["config_hash"]
        except Exception:
            continue
        for h in col.dropna().unique():
            hashes.setdefault(str(h), set()).add(csv_path.name)
    expected = cfg.hash
    ok = len(hashes) == 1 and expected in hashes
    if not ok:
        stale = {h: sorted(v)[:3] for h, v in hashes.items() if h != expected}
        failures.append(
            f"frozen artifacts span {len(hashes)} config hashes (expected only "
            f"{expected}); stale: {stale}")
    report(ok, "all frozen artifacts share one config hash",
           f"{expected}" if ok else f"{len(hashes)} distinct hashes")
    if len(ledger):
        lh = set(str(x) for x in ledger["config_hash"].dropna().unique())
        ok2 = lh == {expected}
        if not ok2:
            failures.append(f"claim ledger config hashes {lh} != {{{expected}}}")
        report(ok2, "claim ledger stamped with the live config hash")

    # ---- 3c. single code commit across the whole frozen release --------
    commits: dict[str, set] = {}
    for csv_path in sorted(RESULTS.glob("*.csv")):
        if csv_path.name in ("claim_ledger.csv", "all_metrics.csv"):
            continue
        try:
            col = pd.read_csv(csv_path, usecols=["git_commit"])["git_commit"]
        except Exception:
            continue
        for c in col.dropna().unique():
            commits.setdefault(str(c), set()).add(csv_path.name)
    ok = len(commits) <= 1
    if not ok:
        failures.append(
            f"frozen artifacts span {len(commits)} code commits "
            f"{sorted(c[:8] for c in commits)}; a release must be one clean pass")
    report(ok, "all frozen artifacts share one code commit",
           f"{sorted(commits)[0][:12]}" if len(commits) == 1 else
           f"{len(commits)} distinct commits")

    # ---- 4. excluded later-paper citations ----------------------------
    pats = cfg.section("wp_0")["excluded_citation_patterns"]
    keys = cfg.section("wp_0")["excluded_citation_keys"]
    hits = []
    for path in (args.manuscript, args.bib):
        if not path.exists():
            continue
        blob = path.read_text(encoding="utf-8", errors="replace")
        for k in keys:
            if k and k in blob:
                hits.append(f"{path.name}:{k}")
        if path.suffix == ".bib":
            for entry in re.findall(r"@\w+\{([^,]+),(.*?)\n\}", blob, re.S):
                key, body = entry
                title = " ".join(re.findall(r"title\s*=\s*[{\"](.+?)[}\"]", body,
                                            re.S | re.I)).lower()
                year = re.findall(r"year\s*=\s*[{\"]?(\d{4})", body)
                yr = int(year[0]) if year else 0
                if yr >= 2025 and any(pt.lower() in title for pt in pats):
                    hits.append(f"{path.name}:{key} ({yr}) title matches "
                                f"an excluded later-paper pattern")
    ok = not hits
    if not ok:
        failures.append(f"excluded citations present: {hits[:4]}")
    report(ok, "no excluded later-paper citations", "; ".join(hits[:2]))

    # ---- 5. collaborator-owned exponent-sweep artifacts ---------------
    forbidden = cfg.section("wp_0")["forbidden_artifacts"]
    found = []
    for pth in list(ROOT.rglob("*.csv")) + list(ROOT.rglob("*.json")) \
            + list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.tex")):
        if ".git/" in str(pth):
            continue
        name = pth.name.lower()
        for f in forbidden:
            if f in name:
                found.append(str(pth.relative_to(ROOT)))
    # also: no code may read such a file
    for pth in ROOT.rglob("*.py"):
        if ".git/" in str(pth):
            continue
        blob = pth.read_text(encoding="utf-8", errors="replace").lower()
        for f in forbidden:
            if f"{f}.csv" in blob:
                found.append(f"{pth.relative_to(ROOT)} references {f}.csv")
    ok = not found
    if not ok:
        failures.append(f"collaborator exponent-sweep artifacts: {found[:4]}")
    report(ok, "no collaborator-owned exponent-sweep artifacts",
           "; ".join(found[:2]))

    # ---- 6. figure freshness ------------------------------------------
    stale = []
    for fig, src in FIGURE_SOURCES.items():
        fp, sp = FIGURES / fig, RESULTS / src
        if not sp.exists():
            continue
        if not fp.exists():
            stale.append(f"{fig} missing")
        elif fp.stat().st_mtime < sp.stat().st_mtime:
            stale.append(f"{fig} older than {src}")
    derived = {"claim_ledger.csv", "all_metrics.csv"}
    for tbl in TABLES.glob("*.tex"):
        newest = max((q.stat().st_mtime for q in RESULTS.glob("*.csv")
                      if q.name not in derived), default=0)
        if tbl.stat().st_mtime < newest:
            stale.append(f"{tbl.name} older than the frozen CSVs")
    ok = not stale
    if not ok:
        failures.append(f"stale figures/tables: {stale[:4]}")
    report(ok, "figures and tables fresh w.r.t. source CSVs", "; ".join(stale[:2]))

    # ---- 7. code provenance -------------------------------------------
    dirty = git_dirty()
    if dirty:
        failures.append("uncommitted code provenance (dirty tracked code tree)")
    report(not dirty, "frozen runs have committed code provenance")

    manifest = RESULTS / "manifest.json"
    if manifest.exists():
        import json

        man = json.loads(manifest.read_text())
        ok = not man.get("git_dirty_code", True)
        if not ok:
            failures.append("manifest records a dirty code tree at run time")
        report(ok, "manifest records clean code provenance")

    # ---- 8. tests ------------------------------------------------------
    if not args.skip_tests:
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-m", "not slow",
             str(ROOT / "tests")], cwd=ROOT, capture_output=True, text=True)
        ok = res.returncode == 0
        if not ok:
            failures.append("test suite failed")
        report(ok, "quick test suite",
               res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "")

    print("=" * 78)
    if failures:
        print(f"{len(failures)} gate failure(s):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("All submission gates pass.")
    if args.submission and failures:
        print("Submission gate failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
