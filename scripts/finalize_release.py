#!/usr/bin/env python3
"""Assemble every release deliverable from whatever frozen CSVs exist.

Run this after the work-package campaigns have finished (or at any time, to
refresh the release from the current artifacts):

    python scripts/finalize_release.py

It runs, in dependency order:

    scripts/make_figures.py             figures from the frozen CSVs
    scripts/run_jhep_frozen_suite.py    ledger, manifest, all_metrics, summary,
      --stages none                     run_commands, environment, git status,
                                        and the frozen manuscript + tables
    scripts/make_progress_reports.py    per-work-package progress reports
    scripts/make_submission_bundle.py   JHEP_frozen_submission_bundle.zip
    scripts/check_jhep_claim_gates.py   the submission gate report

Nothing here recomputes physics; it is pure assembly, so it is cheap and safe
to re-run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("figures", ["scripts/make_figures.py"]),
    ("release artifacts + frozen manuscript",
     ["scripts/run_jhep_frozen_suite.py", "--stages", "none", "--no-figures"]),
    ("progress reports", ["scripts/make_progress_reports.py"]),
    ("submission bundle", ["scripts/make_submission_bundle.py"]),
]


def main() -> int:
    for label, args in STEPS:
        print(f"\n=== {label} ===", flush=True)
        res = subprocess.run([sys.executable, str(ROOT / args[0]), *args[1:]],
                             cwd=ROOT)
        if res.returncode != 0:
            print(f"  [error] {label} failed with exit code {res.returncode}")
            return res.returncode
    print("\n=== submission gate report ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/check_jhep_claim_gates.py")],
                   cwd=ROOT)
    print("\nRun `python scripts/check_jhep_claim_gates.py --submission` to enforce "
          "the gates (non-zero exit on failure).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
