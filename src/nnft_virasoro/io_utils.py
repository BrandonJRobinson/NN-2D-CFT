"""CSV/JSON artifact writing with `--skip-existing` and schema enforcement."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .provenance import PROVENANCE_COLUMNS, require_columns

__all__ = ["write_rows", "write_json", "should_skip", "read_rows", "RESULTS", "FIGURES",
           "ROOT"]

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "jhep_frozen"
FIGURES = ROOT / "figures_jhep_frozen"


def should_skip(path: Path, skip_existing: bool) -> bool:
    return bool(skip_existing and Path(path).exists() and Path(path).stat().st_size > 0)


def write_rows(path: Path | str, rows: Sequence[Mapping[str, Any]], *,
               required: Sequence[str] = (), sort_by: Sequence[str] | None = None,
               float_format: str | None = "%.12g") -> Path:
    """Write rows to CSV after enforcing provenance + required columns."""
    path = Path(path)
    cols = tuple(required) + tuple(PROVENANCE_COLUMNS)
    require_columns(rows, cols)
    df = pd.DataFrame(list(rows))
    if sort_by:
        df = df.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format)
    return path


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return {"re": obj.real, "im": obj.imag}
        return str(obj)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=default)
        handle.write("\n")
    return path


def read_rows(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)
