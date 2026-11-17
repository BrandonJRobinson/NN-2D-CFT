# WP-G progress report — explicit edge remainder

## Command

```bash
python scripts/run_edge_remainder.py
```

## Outputs

* `edge_remainders.json` — 54 `(m, n, Lambda)` combinations.
  For every one, the **full** normal-ordered monomial support is emitted with
  `normal_ordered_monomial`, `coefficient`, `oscillator_indices`,
  `edge_support_reason` and `coefficient_abs`.
* `edge_remainder_summary.csv` — 54 rows with `K_max`, `n_terms` and the
  maximum, `L1` and `L2` coefficient norms.

Norm ranges over all audited combinations: max
0–16; L1
0–58; L2
0–29.09.

## Filtered-Fock exactness, verified separately

540 explicit Fock-space checks were performed over levels
`K in {0,1,2,3,4}` and charges `a in {0, 0.7}`. Of these,
**366 lie inside the guaranteed domain
`K + m_- + n_- <= Lambda`, and all 366 give a
machine-precision zero residual** — Theorem 4.6 holds exactly as stated. The
runner raises if any guaranteed case fails, so this cannot silently regress.

Checks just outside the guaranteed region are non-zero, so the filtered bound
is sharp rather than vacuous (`tests/test_algebra.py`).

The remainder is retained purely as a diagnostic of the cutoff boundary. It is
**not** used to define a "bulk projection" and it is not used to define the
domain on which the central theorem holds.
