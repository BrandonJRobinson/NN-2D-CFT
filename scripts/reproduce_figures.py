"""Regenerate all nine current figure assets from saved numerical inputs."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    "regulator_and_pairing", "predarboux_width", "architecture_symmetries",
    "tt_geometry", "ward_independent_blocks", "exact_phase_ward",
    "ns_two_constructions", "boundary_kernels", "pbw_epsilon_sweep",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for script in [
        "evidence/figure_sources/redraw_figures.py",
        "evidence/major_revision/plot_ward_revision.py",
        "scripts/reproduce_remaining_figures.py",
    ]:
        subprocess.run([sys.executable, str(ROOT / script),
                        "--output-dir", str(output)], cwd=ROOT, check=True)
    missing = [name for name in FIGURES
               if not (output / (name + ".pdf")).is_file()]
    if missing:
        raise SystemExit("Missing figures: " + ", ".join(missing))
    print(f"Regenerated all {len(FIGURES)} figures in {output}.")


if __name__ == "__main__":
    main()
