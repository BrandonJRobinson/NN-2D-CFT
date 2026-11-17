#!/usr/bin/env python3
"""Form full-ensemble Ward ratios from exact phase-integrated momentum blocks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
seed = pd.read_csv(ROOT / "ward_exact_seed_rows.csv")
points = pd.read_csv(ROOT / "ward_exact_probe_rows.csv")
cfg = json.loads((ROOT / "ward_benchmark_config.json").read_text())
a = np.array(cfg["charges"])
w = np.array([complex(*v) for v in cfg["positions"]])
z = np.array([complex(*v) for v in cfg["probes"]])
target = .5 * np.sum(a / (z[:, None] - w), axis=1)**2
norm = np.linalg.norm(target)
rows = []
for (uv, n), g in seed.groupby(["lambda_uv", "n_features"]):
    d = points[(points.lambda_uv == uv) & (points.n_features == n)]
    R = (d.pivot(index="seed_index", columns="probe_index", values="stress_exact_real").to_numpy()
         +1j*d.pivot(index="seed_index", columns="probe_index", values="stress_exact_imag").to_numpy())
    logden = g.sort_values("seed_index").log_vertex_denominator.to_numpy()
    weights = np.exp(logden - logden.max())
    estimate = np.sum(weights[:, None] * R, axis=0) / weights.sum()
    B = len(weights)
    leave = np.array([(np.sum(weights[:, None] * R, axis=0) - weights[i] * R[i])
                      / (weights.sum() - weights[i]) for i in range(B)])
    errors = np.linalg.norm(leave - target, axis=1) / norm
    error = np.linalg.norm(estimate - target) / norm
    se = np.sqrt((B - 1) / B * np.sum((errors - errors.mean())**2))
    rows.append(dict(lambda_uv=uv, n_features=n, independent_momentum_blocks=B,
                     full_ratio_relative_rms=float(error), rms_error_jackknife_se=float(se),
                     effective_momentum_blocks=float(weights.sum()**2 / np.sum(weights**2))))
pd.DataFrame(rows).to_csv(ROOT / "ward_full_ensemble_summary.csv", index=False, float_format="%.17g")
print(pd.DataFrame(rows).query("n_features == 2097152").to_string(index=False))
