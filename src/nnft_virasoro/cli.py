"""Shared CLI scaffolding: every runner exposes --config/--output-dir/--skip-existing."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config, load_config
from .io_utils import FIGURES, RESULTS
from .provenance import Stamper

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "jhep_frozen.yaml"


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="frozen YAML configuration")
    p.add_argument("--output-dir", type=Path, default=RESULTS,
                   help="directory for CSV/JSON artifacts")
    p.add_argument("--figure-dir", type=Path, default=FIGURES,
                   help="directory for figures")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip artifacts that already exist and are non-empty")
    p.add_argument("--allow-dirty", action="store_true", default=True,
                   help="allow stamping from a dirty code tree; the recorded "
                        "provenance includes its dirty state")
    return p


def setup(args, script: str) -> tuple[Config, Stamper, Path, Path]:
    cfg = load_config(args.config)
    stamper = Stamper(script=script, config_hash=cfg.hash,
                      allow_dirty=bool(args.allow_dirty))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figure_dir)
    figdir.mkdir(parents=True, exist_ok=True)
    return cfg, stamper, outdir, figdir
