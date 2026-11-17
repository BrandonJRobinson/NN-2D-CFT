#!/usr/bin/env python3
"""Fock-readout checks and exact phase-averaged Ward benchmark.

This supplement leaves the September evidence unchanged. It reads the
twenty preserved scalar pairing matrices and generates new independent
momentum samples for a declared neutral two-vertex configuration.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import j0, j1, jv
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
for item in json.loads((ROOT / "INPUTS.json").read_text()):
    assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def fock_checks():
    df = pd.read_csv(DATA / "gate01_intrinsic_admissibility_seed_rows.csv")
    df = df[(df.architecture == "selected_log_kernel") &
            (df.spectral_regulator_class == "tapered_log_window")]
    rows = []
    for _, rec in df.iterrows():
        omega = np.load(DATA / rec.omega_file)
        L = omega.shape[0] // 2
        D = np.diag(np.arange(1, L + 1, dtype=float))
        target = np.block([[0 * D, D], [-D, 0 * D]])
        C = np.block([[0 * D, np.eye(L)], [np.eye(L), 0 * D]])
        G = omega[:L, L:]
        H = (G + G.conj().T) / 2
        projected = np.block([[0 * D, H], [-H.T, 0 * D]])
        raw = np.linalg.norm(omega - target)
        err = np.linalg.norm(projected - target)
        correction = np.linalg.norm(projected - omega)
        assert np.linalg.norm(omega + omega.T) < 1e-11
        assert err <= raw + 1e-12
        assert abs(raw**2 - err**2 - correction**2) < 1e-10
        assert np.linalg.norm(C @ projected.conj() @ C.T + projected) < 1e-12
        di = np.diag(1 / np.sqrt(np.arange(1, L + 1)))
        ds = np.diag(np.sqrt(np.arange(1, L + 1)))
        h = di @ H @ di
        eta = np.linalg.norm(h - np.eye(L), ord=2)
        vals, U = np.linalg.eigh(H)
        positive = vals[0] > 0
        row = dict(
            n_features=int(rec.n_features), seed_name=rec.seed_name,
            raw_pairing_error=raw / np.linalg.norm(target),
            projected_pairing_error=err / np.linalg.norm(target),
            projection_correction=correction / np.linalg.norm(target),
            raw_A_norm=float(np.linalg.norm(omega[:L, :L])),
            raw_hermiticity_defect=float(np.linalg.norm(G - G.conj().T) / np.linalg.norm(G)),
            min_H_eigenvalue=float(vals[0]),
            min_normalized_H_eigenvalue=float(np.linalg.eigvalsh(h)[0]),
            eta=float(eta), projected_positive=bool(positive),
        )
        if positive:
            M = ds @ ((U * vals**-0.5) @ U.conj().T)
            S = np.block([[M, 0 * M], [0 * M, M.conj()]])
            row["principal_canonical_error"] = float(np.linalg.norm(S @ projected @ S.T - target))
            row["principal_involution_error"] = float(np.linalg.norm(S.conj() @ C - C @ S))
            assert row["principal_canonical_error"] < 1e-10
            assert row["principal_involution_error"] < 1e-12
            hv, V = np.linalg.eigh(h)
            Mn = ds @ ((V * hv**-0.5) @ V.conj().T) @ di
            assert np.linalg.norm(Mn @ H @ Mn.conj().T - D) < 1e-10
            row["normalized_map_distance"] = float(np.linalg.norm(di @ Mn @ ds - np.eye(L), ord=2))
            if eta < 1:
                row["normalized_map_distance_bound"] = float((1 - eta)**-0.5 - 1)
                assert row["normalized_map_distance"] <= row["normalized_map_distance_bound"] + 1e-11
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "fock_completion_seed_rows.csv", index=False, float_format="%.17g")
    summary = []
    for N, d in out.groupby("n_features"):
        summary.append(dict(
            n_features=int(N), samples=len(d),
            raw_exact_admissible=int(((d.raw_A_norm < 1e-10) &
                                     (d.raw_hermiticity_defect < 1e-10)).sum()),
            projected_positive=int(d.projected_positive.sum()),
            normalized_positivity_certificate=int((d.eta < 1).sum()),
            raw_error_mean=float(d.raw_pairing_error.mean()),
            projected_error_mean=float(d.projected_pairing_error.mean()),
            projection_correction_mean=float(d.projection_correction.mean()),
        ))
    (ROOT / "fock_completion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def taper(u, tau):
    return np.where(u < tau, .5 * (1 - np.cos(np.pi * u / tau)),
                    np.where(u > 1 - tau, .5 * (1 - np.cos(np.pi * (1 - u) / tau)), 1.))


def sample_momenta(rng, n, lo, hi, tau):
    out = np.empty(n)
    filled = 0
    while filled < n:
        count = min(262144, 2 * (n - filled) + 32)
        u = rng.uniform(size=count)
        keep = u[rng.uniform(size=count) < taper(u, tau)][:n - filled]
        out[filled:filled + len(keep)] = lo * np.exp(np.log(hi / lo) * keep)
        filled += len(keep)
    return out * np.exp(2j * np.pi * rng.uniform(size=n))


def exact_ward(k, sigma, charges, positions, probes):
    """Exact independent uniform-phase average at fixed momenta."""
    n = len(k)
    s = sigma / np.sqrt(n)
    j_exact = np.zeros(len(probes), dtype=complex)
    j_gaussian = np.zeros_like(j_exact)
    correction = np.zeros_like(j_exact)
    log_characteristic = 0.
    for start in range(0, n, 32768):
        kb = k[start:start + 32768].conj()
        Q = np.exp(1j * np.real(kb[:, None] * positions)) @ charges
        magnitude = abs(Q)
        arg = s * magnitude
        chi = j0(arg)
        if (chi <= 0).any():
            raise ValueError("This benchmark requires positive one-feature J0 factors.")
        log_characteristic += np.log(chi).sum()
        r1 = j1(arg) / chi
        r2 = jv(2, arg) / chi
        delta = np.real(kb[:, None] * probes) - np.angle(Q)[:, None]
        sine = np.sin(delta)
        j_exact += (s / 2 * kb[:, None] * sine * r1[:, None]).sum(axis=0)
        j_gaussian += (s**2 / 4 * kb[:, None] * magnitude[:, None] * sine).sum(axis=0)
        correction += (-s**2 / 16 * kb[:, None]**2 * np.cos(2 * delta) * r2[:, None]
                       -s**2 / 8 * kb[:, None]**2 * sine**2 * r1[:, None]**2).sum(axis=0)
    return j_exact, .5 * j_exact**2 + correction, j_gaussian, .5 * j_gaussian**2, log_characteristic


def phase_quadrature_check():
    k = np.array([1.2 + .7j, -.4 + 1.1j])
    sigma = 1.3
    a = np.array([.3, -.3])
    w = np.array([-.5, .5], dtype=complex)
    z = np.array([-.5 + .2j, .5 + .35j])
    je, te, _, _, log_den = exact_ward(k, sigma, a, w, z)
    phases = np.arange(256) * 2 * np.pi / 256
    g1, g2 = np.meshgrid(phases, phases)
    gamma = np.stack([g1.ravel(), g2.ravel()], axis=1)
    s = sigma / np.sqrt(2)
    phi = s * np.cos(gamma[:, :, None] + np.real(k.conj()[:, None] * w)).sum(axis=1)
    source = np.exp(1j * (phi @ a))
    fields = -1j * s / 2 * (k.conj()[None, :, None] *
              np.sin(gamma[:, :, None] + np.real(k.conj()[:, None] * z))).sum(axis=1)
    c = -s**2 / 8 * np.sum(k.conj()**2)
    jq = (fields * source[:, None]).mean(axis=0) / source.mean()
    tq = (.5 * (fields**2 - c) * source[:, None]).mean(axis=0) / source.mean()
    gaps = dict(current=float(abs(je - jq).max()), stress=float(abs(te - tq).max()),
                denominator=float(abs(np.exp(log_den) - source.mean())))
    assert max(gaps.values()) < 1e-12, gaps
    return gaps


def regulated_current(lo, hi, tau, charges, positions, probes):
    ell = np.log(hi / lo)
    separation = probes[:, None] - positions
    I = []
    for distance in abs(separation).ravel():
        values = [quad(lambda u: ell * lo * np.exp(ell * u) *
                   float(taper(u, tau)) * j1(lo * np.exp(ell * u) * distance),
                   low, high, epsabs=2e-12, epsrel=2e-12, limit=600)[0]
                  for low, high in [(0, tau), (tau, 1 - tau), (1 - tau, 1)]]
        I.append(sum(values))
    radial = np.array(I).reshape(separation.shape)
    return (charges * separation.conj() / abs(separation) * radial).sum(axis=1)


def historical_ward_summary():
    blocks = pd.read_csv(DATA / "deterministic_ward_normalized_blocks.csv")
    blocks = blocks[(blocks.regulator_class == "tapered_log_window") & (blocks.n_features == 200000)]
    p = blocks.pivot(index=["seed_name", "configuration"], columns="architecture",
                     values="normalized_l2_pooled")
    logs = np.log(p.rbf_gaussian_prior / p.selected_log_kernel).groupby("seed_name").mean()
    assert len(logs) == 8
    mean = float(logs.mean())
    se = float(logs.std(ddof=1) / np.sqrt(len(logs)))
    interval = np.exp(mean + np.array([-1, 1]) * student_t.ppf(.975, 7) * se)
    summary = dict(ratio=float(np.exp(mean)), mean_log_ratio=mean,
                   log_ratio_standard_error=se, independent_seeds=8,
                   interval_method="Student t on eight seed-averaged log ratios, 7 degrees of freedom",
                   ci95=interval.tolist())
    (ROOT / "historical_ward_ratio_t7.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def benchmark():
    cfg = dict(
        date="2026-11-12", method="exact uniform-phase integration conditional on i.i.d. momenta",
        widths=[8192, 32768, 131072, 524288, 2097152],
        regulators=[dict(lambda_ir=1 / b, lambda_uv=b, tau=.15) for b in [30., 100., 300.]],
        seed_indices=list(range(8)), seed_sequence_entropy=[20261112, "regulator_index", "seed_index"],
        charges=[.3, -.3], positions=[[-.5, 0.], [.5, 0.]],
        probes=[[x, y] for y in [.2, .35, .5] for x in [-.5, .5]],
        nesting="One maximum-width momentum sample per regulator/seed; width grids use prefixes.",
        uncertainty="Standard error over eight independent momentum seeds at each cell; widths share prefixes.",
        inference="Descriptive accuracy benchmark; no fitted convergence exponents or significance tests.",
    )
    (ROOT / "ward_benchmark_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    a = np.array(cfg["charges"])
    w = np.array([complex(*v) for v in cfg["positions"]])
    z = np.array([complex(*v) for v in cfg["probes"]])
    j_cft = np.sum(a / (z[:, None] - w), axis=1)
    t_cft = .5 * j_cft**2
    denominator = np.linalg.norm(t_cft)
    rows, points, references = [], [], []
    for reg_index, reg in enumerate(cfg["regulators"]):
        lo, hi, tau = reg["lambda_ir"], reg["lambda_uv"], reg["tau"]
        Lw = (1 - tau) * np.log(hi / lo)
        jr = regulated_current(lo, hi, tau, a, w, z)
        tr = .5 * jr**2
        bias = float(np.linalg.norm(tr - t_cft) / denominator)
        for j in range(len(z)):
            references.append(dict(lambda_uv=hi, probe_index=j,
                current_reg_real=jr[j].real, current_reg_imag=jr[j].imag,
                stress_reg_real=tr[j].real, stress_reg_imag=tr[j].imag,
                stress_cft_real=t_cft[j].real, stress_cft_imag=t_cft[j].imag))
        for seed_index in cfg["seed_indices"]:
            rng = np.random.default_rng(np.random.SeedSequence([20261112, reg_index, seed_index]))
            k = sample_momenta(rng, max(cfg["widths"]), lo, hi, tau)
            for n in cfg["widths"]:
                je, te, jg, tg, log_den = exact_ward(k[:n], np.sqrt(4 * Lw), a, w, z)
                rows.append(dict(lambda_ir=lo, lambda_uv=hi, tau=tau,
                    n_features=n, seed_index=seed_index, regulator_error=bias,
                    exact_total_error=float(np.linalg.norm(te - t_cft) / denominator),
                    exact_to_regulated_error=float(np.linalg.norm(te - tr) / denominator),
                    gaussian_proxy_error=float(np.linalg.norm(tg - t_cft) / denominator),
                    exact_proxy_difference=float(np.linalg.norm(te - tg) / denominator),
                    current_total_error=float(np.linalg.norm(je - j_cft) / np.linalg.norm(j_cft)),
                    log_vertex_denominator=float(log_den)))
                for j in range(len(z)):
                    points.append(dict(lambda_uv=hi, n_features=n, seed_index=seed_index,
                        probe_index=j, current_exact_real=je[j].real, current_exact_imag=je[j].imag,
                        stress_exact_real=te[j].real, stress_exact_imag=te[j].imag,
                        stress_proxy_real=tg[j].real, stress_proxy_imag=tg[j].imag))
            print("completed Ward regulator", hi, "seed", seed_index, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "ward_exact_seed_rows.csv", index=False, float_format="%.17g")
    pd.DataFrame(points).to_csv(ROOT / "ward_exact_probe_rows.csv", index=False, float_format="%.17g")
    pd.DataFrame(references).to_csv(ROOT / "ward_regulated_reference.csv", index=False, float_format="%.17g")
    metrics = ["exact_total_error", "exact_to_regulated_error", "gaussian_proxy_error",
               "exact_proxy_difference", "current_total_error", "regulator_error"]
    summaries = []
    for (uv, n), d in result.groupby(["lambda_uv", "n_features"]):
        row = dict(lambda_uv=float(uv), n_features=int(n), seeds=len(d))
        for metric in metrics:
            row[metric + "_mean"] = float(d[metric].mean())
            row[metric + "_se"] = float(d[metric].std(ddof=1) / np.sqrt(len(d)))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(ROOT / "ward_exact_summary.csv", index=False, float_format="%.17g")
    return summaries


if __name__ == "__main__":
    fock = fock_checks()
    quadrature = phase_quadrature_check()
    ratio = historical_ward_summary()
    results = benchmark()
    report = dict(
        phase_quadrature_gaps=quadrature,
        fock=fock, historical_paired_ratio=ratio,
        maximum_width_ward=[r for r in results if r["n_features"] == 2097152],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        historical_inputs_unchanged=True,
    )
    (ROOT / "analysis_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
