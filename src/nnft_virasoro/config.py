"""Frozen-configuration loading, named seeds, and config hashing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .provenance import hash_object


@dataclass(frozen=True)
class Config:
    path: Path
    data: dict

    @property
    def hash(self) -> str:
        return hash_object(self.data)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def section(self, name: str) -> dict:
        try:
            sec = self.data[name]
        except KeyError as exc:
            raise KeyError(
                f"configuration {self.path} is missing required section {name!r}"
            ) from exc
        if not isinstance(sec, dict):
            raise TypeError(f"configuration section {name!r} must be a mapping")
        return sec

    def section_hash(self, name: str) -> str:
        return hash_object({name: self.data[name], "global": self.data.get("global", {})})


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not parse to a mapping")
    return Config(path=path, data=data)


# --------------------------------------------------------------------------
# Named seeds
# --------------------------------------------------------------------------
def seed_value(named_seeds: Mapping[str, int], name: str) -> int:
    """Resolve an explicit named seed; unknown names fail loudly."""
    try:
        return int(named_seeds[name])
    except KeyError as exc:
        raise KeyError(
            f"seed name {name!r} is not declared in the configuration's "
            f"named_seeds block (declared: {sorted(named_seeds)})"
        ) from exc


def rng_for(named_seeds: Mapping[str, int], name: str, *stream: Any) -> np.random.Generator:
    """Deterministic generator from a *named* seed plus an integer stream label.

    ``stream`` entries are hashed into a stable 64-bit spawn key so that
    e.g. (seed name, n_features, mode_cutoff, architecture) always maps to the
    same generator regardless of iteration order.
    """
    base = seed_value(named_seeds, name)
    key = int(hash_object([name, base, [str(s) for s in stream]])[:12], 16)
    return np.random.default_rng(np.random.SeedSequence([base, key]))


def seed_names(cfg: Config, block: str = "named_seeds") -> list[str]:
    return sorted(cfg.section(block))
