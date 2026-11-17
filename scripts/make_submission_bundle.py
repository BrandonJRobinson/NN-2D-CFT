#!/usr/bin/env python3
"""Assemble `JHEP_frozen_submission_bundle.zip` from the frozen release."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nnft_virasoro.cli import base_parser, setup  # noqa: E402
from nnft_virasoro.provenance import file_sha256, git_commit, utc_now  # noqa: E402

SCRIPT = "scripts/make_submission_bundle.py"
OUT = ROOT / "JHEP_frozen_submission_bundle.zip"

INCLUDE_FILES = [
    "Draft-JHEP-Frozen.tex",
    "Virasoro_JHEP_Frozen.bib",
    "configs/jhep_frozen.yaml",
    "CLAIM_COMPLETION_BOARD.csv",
    "CLAIM_COMPLETION_BOARD.md",
    "CURRENT_NUMERICAL_PROVENANCE_v3.csv",
]
INCLUDE_TREES = [
    "results/jhep_frozen",
    "figures_jhep_frozen",
    "tables_jhep_frozen",
    "src/nnft_virasoro",
    "scripts",
    "tests",
]
SKIP_SUFFIX = {".pyc"}
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    members: list[Path] = []
    for rel in INCLUDE_FILES:
        p = ROOT / rel
        if p.exists():
            members.append(p)
    for tree in INCLUDE_TREES:
        base = ROOT / tree
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_dir() or p.suffix in SKIP_SUFFIX:
                continue
            if any(d in p.parts for d in SKIP_DIRS):
                continue
            members.append(p)

    lines = [
        "JHEP Virasoro frozen submission bundle",
        f"generated_utc  {utc_now()}",
        f"git_commit     {git_commit()}",
        f"config_hash    {cfg.hash}",
        f"n_files        {len(members)}",
        "",
        "sha256                                                            path",
    ]
    for p in members:
        lines.append(f"{file_sha256(p)}  {p.relative_to(ROOT)}")
    listing = "\n".join(lines) + "\n"

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in members:
            z.write(p, arcname=str(p.relative_to(ROOT)))
        z.writestr("BUNDLE_MANIFEST.txt", listing)
    print(f"  wrote {OUT.name}: {len(members)} files, "
          f"{OUT.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
