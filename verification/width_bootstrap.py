"""Recompute the equal-weight width exponent from saved independent seeds."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/jhep_submission_hardening"


def load(name):
    frame = pd.read_csv(DATA / name)
    frame = frame[frame.architecture == "selected_log_kernel"]
    if "spectral_regulator_class" in frame:
        frame = frame[frame.spectral_regulator_class == "tapered_log_window"]
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    names = ["gate01_intrinsic_admissibility_seed_rows.csv",
             "gate02_predarboux_virasoro_seed_rows.csv",
             "ns_pairing_seed_rows.csv", "ns_super_virasoro_seed_rows.csv"]
    g1, g2, car, algebra = [load(name) for name in names]
    # The legacy delta_iso column is called delta_G in the manuscript.
    specs = [("delta_iso", "controls", g1, "delta_iso"),
             ("b_omega", "controls", g2, "pairing_error_frobenius")]
    for construction in sorted(car.ns_construction.unique()):
        for label, frame, column in [
                ("car_relative_error", car, "fermionic_pairing_relative_error"),
                ("LG_residual", algebra, "raw_LG_residual"),
                ("GG_residual", algebra, "raw_GG_residual")]:
            specs.append((f"{label}[{construction}]", "ns",
                          frame[frame.ns_construction == construction], column))
    assert len(specs) == 8
    arrays, families, estimates, family_seeds = [], [], [], {}
    for label, family, frame, column in specs:
        table = frame.pivot(index="seed_name", columns="n_features", values=column)
        table = table.sort_index().sort_index(axis=1)
        values = table.to_numpy(float)
        assert np.isfinite(values).all() and (values > 0).all()
        seeds = list(table.index)
        if family in family_seeds:
            assert seeds == family_seeds[family]
        family_seeds[family] = seeds
        x = np.log(table.columns.to_numpy(float))
        x = (x - x.mean()) / np.sum((x - x.mean())**2)
        exponent = float(np.log(values.mean(axis=0)) @ x)
        arrays.append((values, x))
        families.append(family)
        estimates.append({"quantity": label, "family": family,
                          "exponent": exponent,
                          "widths": [int(n) for n in table.columns]})
    # Match the production fits before performing any resampling.
    archived = pd.read_csv(DATA / "convergence_joint_fit.csv")
    archived = archived[(archived.architecture == "selected_log_kernel")
                        & (archived.quantity != "POOLED_EXPONENT")]
    for row in estimates:
        old = archived[archived.quantity == row["quantity"]]
        assert len(old) == 1
        assert abs(row["exponent"] - float(old.iloc[0].exponent)) < 1e-10
    seed, replicates = 20261118, 1000
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        selections = {family: rng.integers(len(seeds), size=len(seeds))
                      for family, seeds in family_seeds.items()}
        slopes = [float(np.log(values[selections[family]].mean(axis=0)) @ x)
                  for (values, x), family in zip(arrays, families)]
        draws.append(float(np.mean(slopes)))
    interval = np.percentile(draws, [2.5, 97.5]).tolist()
    summary = {
        "method": "Equal mean of eight log-log OLS slopes of seed-mean errors; "
                  "seed clusters resampled independently within controls and NS; "
                  "one common draw per family across all its observables and widths.",
        "point_estimate": float(np.mean([r["exponent"] for r in estimates])),
        "percentile_interval_95": interval,
        "bootstrap_seed": seed, "replicates": replicates,
        "rng": "numpy.random.default_rng / PCG64",
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "family_seeds": family_seeds, "observables": estimates,
        "inputs": [{"path": str((DATA / name).relative_to(ROOT)),
                    "sha256": hashlib.sha256((DATA / name).read_bytes()).hexdigest()}
                   for name in names + ["convergence_joint_fit.csv"]],
        "scope": "Reanalysis of saved numerical evidence; no feature simulations.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "width_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    pd.DataFrame({"replicate": np.arange(replicates), "exponent": draws}).to_csv(
        args.output_dir / "width_bootstrap_replicates.csv", index=False,
        float_format="%.17g")
    print(json.dumps({k: summary[k] for k in ["point_estimate",
          "percentile_interval_95", "bootstrap_seed", "replicates"]}, indent=2))


if __name__ == "__main__":
    main()
