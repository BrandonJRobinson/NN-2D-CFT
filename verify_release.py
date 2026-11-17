"""Verify the numerical reproduction package without running experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(records, root=ROOT):
    for row in records:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        path = root / relative
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError("Missing or changed file: " + str(relative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-archive-times", action="store_true",
                        help="Normalize verified file times for the legacy figure-freshness test.")
    args = parser.parse_args()
    payload = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())["files"]
    check(payload)
    for row in payload:
        path = Path(row["path"])
        if path.suffix in {".tex", ".bib"} or path.parts[0] in {
                "docs", "legacy", "provenance", "manuscript_evidence"}:
            raise ValueError("Non-reproduction file in payload: " + str(path))
    sources = json.loads((ROOT / "PROVENANCE.json").read_text())["source_files"]
    check(sources)
    artifacts = 0
    for folder in ["results/jhep_frozen", "results/jhep_submission_hardening"]:
        records = json.loads((ROOT / folder / "manifest.json").read_text())["artifacts"]
        check(records)
        artifacts += len(records)
    for folder in ["verification/boundary", "verification/sectors"]:
        records = json.loads((ROOT / folder / "manifest.json").read_text())["files"]
        check(records, ROOT / folder)
    for folder in ["evidence/figure_sources", "evidence/major_revision"]:
        records = json.loads((ROOT / folder / "INPUTS.json").read_text())
        check(records, ROOT / folder)
    if args.restore_archive_times:
        for row in payload:
            os.utime(ROOT / row["path"], (1794009600, 1794009600))
    print(f"PASS: {len(payload)} reproduction files, {artifacts} numerical artifacts, "
          f"{len(sources)} unchanged generation sources.")


if __name__ == "__main__":
    main()
