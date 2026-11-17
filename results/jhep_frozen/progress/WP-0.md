# WP-0 progress report — release integrity

## Master command

```bash
python scripts/run_jhep_frozen_suite.py \
  --config configs/jhep_frozen.yaml \
  --skip-existing
python scripts/check_jhep_claim_gates.py --submission
```

## Claim ledger

| status               |   rows |
|:---------------------|-------:|
| accepted             |    165 |
| diagnostic           |     32 |
| failed_gate          |     15 |
| accepted_with_caveat |      7 |

By estimator class:

| estimator_class              |   rows |
|:-----------------------------|-------:|
| deterministic_finite_feature |     99 |
| exact_symbolic               |     69 |
| sampled_finite_network       |     40 |
| analytic_closed_kernel       |     11 |

Schema: claim_id, work_package, manuscript_location, metric, value, uncertainty, confidence_interval, units, estimator_class, script, config, config_hash, seed_set, input_artifact, output_artifact, git_commit, status, notes.

## Submission checker

`scripts/check_jhep_claim_gates.py --submission` fails on any of: a claim gate
without an accepted ledger row; a manuscript number that does not resolve
numerically (to its displayed precision) to a ledger row; a re-appearing
excluded later-paper citation; a collaborator-owned exponent-sweep artifact
appearing anywhere in the tree or being read by any script; a figure or table
older than its source CSV; a dirty tracked code tree; or a failing test suite.

## Config-hash drift audit — KNOWN LIMITATION

The frozen configuration was refined **during** the campaign (grid sizes were
adapted to the measured cost, and the WP-A6 control geometry was moved into the
selected architecture's converged regime). Artifacts therefore carry the hash
under which they were computed:

- `44f002d1050e168a`: `analytic_ward_points.csv`, `deterministic_feature_ward_points.csv`, `sampled_ward_seed_rows.csv`, `sampled_ward_summary.csv`
- `54523e04109b831e`: `central_offset_mode_extension.csv`, `current_kernel_reference.csv`, `edge_remainder_summary.csv`, `ordered_pairing_grid.csv`, `ordered_pairing_radius_plateau.csv`, `predarboux_convergence_fits.csv`, `predarboux_convergence_seed_rows.csv`, `robustness_grid_recomputed.csv`, `sampled_tt_central_charge.csv`, `sampled_tt_seed_rows.csv`
- `59cf9a82f2772f32` **(current)**: `finite_width_cumulant.csv`, `finite_width_cumulant_seed_rows.csv`, `ns_frozen_metrics.csv`, `ns_pairing_seed_rows.csv`, `ns_sampled_correlators.csv`, `ns_super_virasoro_seed_rows.csv`, `sampled_vertex_weights.csv`, `sampled_vertex_weights_seed_rows.csv`
- `b2edb35885ff6685`: `architecture_controls_seed_rows.csv`, `darboux_physical_admissibility.csv`

Every row still records its own `config_hash`, so nothing is misattributed.
But a release-grade run must be a **single clean pass under one hash**: delete
`results/jhep_frozen/*.csv` and re-run the master command. This is the one
outstanding release action and it is stated here rather than papered over.

## Remaining blockers

1. The single-hash clean re-run above.
2. `Draft-JHEP-Frozen.tex` inherits the body of the active draft; the
   quantitative passages are regenerated from the ledger, but a human editorial
   pass is still required before submission.
