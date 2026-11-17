#!/usr/bin/env python3
"""WP-G: explicit finite-cutoff edge remainder outside the filtered domain.

Emits every normal-ordered monomial of ``Delta_{mn}(omega_can)`` with its
coefficient, oscillator indices and the reason it lies outside the guaranteed
domain, plus max / L1 / L2 coefficient norms.  Separately verifies that
``K + m_- + n_- <= Lambda`` gives *exact* closure on the filtered Fock subspace
(Theorem 4.6).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nnft_virasoro import algebra as AL  # noqa: E402
from nnft_virasoro.cli import base_parser, setup  # noqa: E402
from nnft_virasoro.io_utils import should_skip, write_json, write_rows  # noqa: E402
from nnft_virasoro.superalgebra import Pairings  # noqa: E402

SCRIPT = "scripts/run_edge_remainder.py"


def main() -> int:
    args = base_parser(__doc__ or "").parse_args()
    cfg, stamper, outdir, figdir = setup(args, SCRIPT)
    json_path = outdir / "edge_remainders.json"
    csv_path = outdir / "edge_remainder_summary.csv"
    if should_skip(json_path, args.skip_existing) and should_skip(
            csv_path, args.skip_existing):
        print("  [skip] WP-G")
        return 0

    sec = cfg.section("wp_g")
    canonical = Pairings.canonical()
    payload = []
    rows = []
    filtered_checks = []
    for m, n in sec["mode_pairs"]:
        for lam in sec["mode_cutoffs"]:
            rec = AL.edge_remainder_terms(int(m), int(n), int(lam))
            payload.append(rec)
            rows.append(stamper.stamp(dict(
                m=rec["m"], n=rec["n"], **{"lambda": rec["mode_cutoff"]},
                K_max=rec["k_max"], n_terms=rec["n_terms"],
                coefficient_max=rec["coefficient_max"],
                coefficient_l1=rec["coefficient_l1"],
                coefficient_l2=rec["coefficient_l2"],
                has_edge_support=bool(rec["n_terms"] > 0),
            ), estimator_class="exact_symbolic", seed=None))

            # Theorem 4.6: exact closure on F^{(<=K)} when K + m_- + n_- <= Lambda
            el = AL.virasoro_residual(int(m), int(n), int(lam), canonical)
            for K in sec["filtered_check_levels"]:
                guaranteed = (K + AL.m_minus(m) + AL.m_minus(n)) <= lam
                for charge in sec["charges"]:
                    fock = AL.FockSpace(level=int(K), mode_cutoff=int(lam),
                                        charge=float(charge))
                    nrm = fock.frobenius(el)
                    filtered_checks.append(dict(
                        m=int(m), n=int(n), mode_cutoff=int(lam), level=int(K),
                        charge=float(charge), guaranteed=bool(guaranteed),
                        residual_frobenius=float(nrm),
                        exact_closure=bool(nrm <= 1e-10),
                        consistent_with_theorem=bool(
                            (not guaranteed) or nrm <= 1e-10),
                    ))
    violations = [c for c in filtered_checks if not c["consistent_with_theorem"]]
    if violations:
        raise AssertionError(
            f"Theorem 4.6 violated in {len(violations)} guaranteed cases: "
            f"{violations[:3]}")

    write_json(json_path, dict(
        description=(
            "Full normal-ordered edge remainder Delta_{mn}(omega_can) = "
            "[L_m^(Lambda), L_n^(Lambda)] - (m-n)L_{m+n}^(Lambda) - "
            "(c/12)m(m^2-1)delta_{m+n,0} outside the filtered domain of "
            "Theorem 4.6. The remainder is a diagnostic of the cutoff boundary "
            "and is NOT used to define a bulk projection."),
        theorem_domain="K + m_- + n_- <= Lambda  (eq. 4.15)",
        k_max_formula="K_max(Lambda;m,n) = Lambda - m_- - n_-  (eq. D.1)",
        config_hash=cfg.hash,
        filtered_fock_exactness_checks=filtered_checks,
        n_filtered_checks=len(filtered_checks),
        n_guaranteed_checks=sum(1 for c in filtered_checks if c["guaranteed"]),
        n_guaranteed_exact=sum(1 for c in filtered_checks
                               if c["guaranteed"] and c["exact_closure"]),
        remainders=payload))
    write_rows(csv_path, rows, required=[
        "m", "n", "lambda", "K_max", "n_terms", "coefficient_max",
        "coefficient_l1", "coefficient_l2"], sort_by=["m", "n", "lambda"])
    n_g = sum(1 for c in filtered_checks if c["guaranteed"])
    print(f"  wrote {json_path.name} ({len(payload)} remainders, "
          f"{len(filtered_checks)} filtered checks, {n_g} guaranteed, all exact) "
          f"and {csv_path.name} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
