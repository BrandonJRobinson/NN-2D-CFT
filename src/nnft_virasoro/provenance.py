"""Provenance stamping and row-schema enforcement.

Mission engineering rule: every output row must carry the code commit, config
hash, seed, timestamp and estimator class.  Missing columns fail loudly.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import ESTIMATOR_CLASSES, __version__

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Columns required on every frozen output row.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "git_commit",
    "git_dirty",
    "config_hash",
    "seed",
    "timestamp_utc",
    "estimator_class",
    "script",
    "suite_version",
)


class ProvenanceError(RuntimeError):
    """Raised when provenance or schema requirements are violated."""


def _run_git(*args: str) -> str:
    try:
        out = subprocess.run(
            ("git", *args),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        raise ProvenanceError(f"git {' '.join(args)} failed: {exc}") from exc
    return out.stdout.strip()


def git_commit() -> str:
    return _run_git("rev-parse", "HEAD")


def git_dirty() -> bool:
    """True when tracked code differs from HEAD.

    Generated artifacts under ``results/`` and the release figure/table
    namespaces are excluded: a run necessarily writes them, so their presence is
    not an uncommitted-*code* condition. Both the frozen and the submission-
    hardening namespaces are listed, so a hardening production run can report
    ``git_dirty = False`` on exactly the same terms the frozen release did.
    """
    status = _run_git("status", "--porcelain")
    generated_prefixes = (
        "results/",
        "figures_jhep_frozen/",
        "tables_jhep_frozen/",
        "figures_jhep_submission_hardening/",
        "tables_jhep_submission_hardening/",
        "Draft-JHEP-Frozen.tex",
        "Virasoro_JHEP_Frozen.bib",
        "JHEP_frozen_submission_bundle.zip",
        "test-output.xml",
    )
    for line in status.splitlines():
        # Porcelain lines are "XY path", but the surrounding strip() removes the
        # leading space of an unstaged-only status ("' M path'" -> "'M path'"),
        # so the status token must be split off by whitespace, not by offset.
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        path = parts[1].strip().strip('"')
        if "->" in path:  # rename
            path = path.split("->")[-1].strip().strip('"')
        if not path.startswith(generated_prefixes):
            return True
    return False


def git_status_short() -> str:
    return _run_git("status", "--short")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_object(obj: Any) -> str:
    """Deterministic sha256 over a JSON-serializable object."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Stamper:
    """Attaches provenance to every emitted row."""

    def __init__(self, *, script: str, config_hash: str, allow_dirty: bool = True):
        self.script = script
        self.config_hash = config_hash
        self.commit = git_commit()
        self.dirty = git_dirty()
        if self.dirty and not allow_dirty:
            raise ProvenanceError(
                "refusing to stamp frozen rows from a dirty code tree; commit first"
            )

    def stamp(self, row: Mapping[str, Any], *, estimator_class: str, seed: Any) -> dict:
        if estimator_class not in ESTIMATOR_CLASSES:
            raise ProvenanceError(
                f"unknown estimator_class {estimator_class!r}; "
                f"expected one of {ESTIMATOR_CLASSES}"
            )
        out = dict(row)
        out.update(
            git_commit=self.commit,
            git_dirty=bool(self.dirty),
            config_hash=self.config_hash,
            seed="" if seed is None else seed,
            timestamp_utc=utc_now(),
            estimator_class=estimator_class,
            script=self.script,
            suite_version=__version__,
        )
        return out


def require_columns(rows: Sequence[Mapping[str, Any]], columns: Iterable[str]) -> None:
    """Fail loudly on missing columns (mission engineering rule)."""
    columns = tuple(columns)
    if not rows:
        raise ProvenanceError("no rows produced; refusing to emit an empty artifact")
    missing = [c for c in columns if c not in rows[0]]
    if missing:
        raise ProvenanceError(f"missing required columns: {missing}")
    for idx, row in enumerate(rows):
        gaps = [c for c in columns if c not in row]
        if gaps:
            raise ProvenanceError(f"row {idx} missing columns: {gaps}")


def environment_text() -> str:
    import numpy
    import scipy
    import matplotlib
    import pandas

    lines = [
        "JHEP Virasoro frozen suite - environment",
        f"suite_version   {__version__}",
        f"git_commit      {git_commit()}",
        f"git_dirty_code  {git_dirty()}",
        f"generated_utc   {utc_now()}",
        "",
        f"python          {sys.version.splitlines()[0]}",
        f"executable      {sys.executable}",
        f"platform        {platform.platform()}",
        f"machine         {platform.machine()}",
        f"cpu_count       {os.cpu_count()}",
        "",
        f"numpy           {numpy.__version__}",
        f"scipy           {scipy.__version__}",
        f"matplotlib      {matplotlib.__version__}",
        f"pandas          {pandas.__version__}",
    ]
    try:
        import yaml

        lines.append(f"pyyaml          {yaml.__version__}")
    except Exception:  # pragma: no cover
        pass
    lines.append("")
    lines.append("BLAS/LAPACK configuration (numpy):")
    try:
        cfg = numpy.__config__.get_info("blas_opt")  # type: ignore[attr-defined]
        lines.append(f"  {cfg}")
    except Exception:
        lines.append("  (unavailable)")
    return "\n".join(lines) + "\n"
