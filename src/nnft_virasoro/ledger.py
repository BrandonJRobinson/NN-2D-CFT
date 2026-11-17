"""Frozen claim ledger: schema, construction helpers, and validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = ["LEDGER_COLUMNS", "LedgerRow", "ledger_row", "validate_ledger",
           "significant_figures"]

LEDGER_COLUMNS: tuple[str, ...] = (
    "claim_id",
    "work_package",
    "manuscript_location",
    "metric",
    "value",
    "uncertainty",
    "confidence_interval",
    "units",
    "estimator_class",
    "script",
    "config",
    "config_hash",
    "seed_set",
    "input_artifact",
    "output_artifact",
    "git_commit",
    "status",
    "notes",
)

#: Accepted ledger statuses.  Only ``accepted`` satisfies a claim gate.
STATUSES = ("accepted", "accepted_with_caveat", "failed_gate", "inconclusive",
            "diagnostic")


@dataclass
class LedgerRow:
    claim_id: str
    work_package: str
    manuscript_location: str
    metric: str
    value: Any
    uncertainty: Any = ""
    confidence_interval: str = ""
    units: str = "dimensionless"
    estimator_class: str = ""
    script: str = ""
    config: str = "configs/jhep_frozen.yaml"
    config_hash: str = ""
    seed_set: str = ""
    input_artifact: str = ""
    output_artifact: str = ""
    git_commit: str = ""
    status: str = "accepted"
    notes: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        return {c: d[c] for c in LEDGER_COLUMNS}


def significant_figures(value: float, uncertainty: float) -> str:
    """Render ``value`` with no more precision than ``uncertainty`` supports.

    Convention (PDG-style): the uncertainty is carried to two significant
    digits and the value is quoted to the same decimal place, i.e. one digit
    beyond the uncertainty's leading digit.  ``significant_figures(1.23456,
    0.01) == "1.235"``.  With a non-finite or non-positive uncertainty the value
    falls back to six significant digits.
    """
    if not np.isfinite(value):
        return str(value)
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        return f"{value:.6g}"
    exp = int(np.floor(np.log10(abs(uncertainty))))
    digits = max(0, -exp + 1)
    return f"{value:.{digits}f}"


def ledger_row(**kwargs) -> dict:
    row = LedgerRow(**kwargs)
    if row.status not in STATUSES:
        raise ValueError(f"unknown ledger status {row.status!r}; expected {STATUSES}")
    return row.as_dict()


def validate_ledger(rows: Sequence[dict]) -> list[str]:
    """Return a list of schema problems (empty means valid)."""
    problems: list[str] = []
    if not rows:
        return ["claim ledger is empty"]
    seen: set[str] = set()
    for i, r in enumerate(rows):
        missing = [c for c in LEDGER_COLUMNS if c not in r]
        if missing:
            problems.append(f"row {i}: missing columns {missing}")
            continue
        cid = str(r["claim_id"])
        if cid in seen:
            problems.append(f"duplicate claim_id {cid!r}")
        seen.add(cid)
        if str(r["status"]) not in STATUSES:
            problems.append(f"row {cid}: bad status {r['status']!r}")
        for req in ("work_package", "metric", "estimator_class", "script",
                    "config_hash", "output_artifact", "git_commit"):
            if str(r[req]).strip() == "":
                problems.append(f"row {cid}: empty required field {req!r}")
        if str(r["value"]).strip() == "":
            problems.append(f"row {cid}: empty value")
    return problems
