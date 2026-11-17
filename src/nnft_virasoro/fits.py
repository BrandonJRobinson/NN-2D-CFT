r"""Deterministic regressions, null-hypothesis tests, and confidence intervals.

Mission section 3: analytic **and** bootstrap intervals, explicit fitted-row
counts and exclusions, a reported ``R^2``, and theory-motivated exponents tested
as null hypotheses.  All routines are deterministic given their seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["OLSResult", "ols", "loglog_scaling_fit", "power_law_fit", "null_test"]


@dataclass
class OLSResult:
    names: list[str]
    beta: np.ndarray
    stderr: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    r_squared: float
    n_rows: int
    dof: int
    residual_sd: float
    excluded_rows: int
    boot_ci_low: np.ndarray | None = None
    boot_ci_high: np.ndarray | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def coef(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])

    def se(self, name: str) -> float:
        return float(self.stderr[self.names.index(name)])

    def interval(self, name: str) -> tuple[float, float]:
        i = self.names.index(name)
        return float(self.ci_low[i]), float(self.ci_high[i])


def _t_crit(dof: int, alpha: float = 0.05) -> float:
    from scipy import stats

    return float(stats.t.ppf(1.0 - alpha / 2.0, dof)) if dof > 0 else float("nan")


def ols(design: np.ndarray, y: np.ndarray, names: list[str], *,
        n_boot: int = 10000, seed: int = 20260730, alpha: float = 0.05,
        excluded_rows: int = 0, notes: str = "") -> OLSResult:
    X = np.asarray(design, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    dof = n - p
    if dof <= 0:
        raise ValueError(f"OLS underdetermined: {n} rows, {p} parameters")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    sigma2 = rss / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    tc = _t_crit(dof, alpha)
    lo, hi = beta - tc * se, beta + tc * se

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, p))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot[b], *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover
            boot[b] = np.nan
    blo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    bhi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)

    return OLSResult(
        names=list(names), beta=beta, stderr=se, ci_low=lo, ci_high=hi,
        r_squared=float(1.0 - rss / tss) if tss > 0 else float("nan"),
        n_rows=n, dof=dof, residual_sd=float(np.sqrt(sigma2)),
        excluded_rows=int(excluded_rows), boot_ci_low=blo, boot_ci_high=bhi,
        notes=notes,
    )


def loglog_scaling_fit(n_features: np.ndarray, mode_cutoff: np.ndarray,
                       y: np.ndarray, *, regulator_labels: np.ndarray | None = None,
                       seed: int = 20260730, n_boot: int = 10000,
                       notes: str = "") -> OLSResult:
    r"""``log Y = c + b log N + a log Lambda + regulator fixed effects``.

    Non-positive or non-finite ``y`` rows are excluded and counted (a log axis
    cannot represent them); the count is reported, never silently dropped.
    """
    n_features = np.asarray(n_features, dtype=float)
    mode_cutoff = np.asarray(mode_cutoff, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(y) & (y > 0) & np.isfinite(n_features) & (n_features > 0)
    excluded = int((~keep).sum())
    cols = [np.ones(keep.sum()), np.log(n_features[keep]), np.log(mode_cutoff[keep])]
    names = ["intercept", "log_n_features", "log_mode_cutoff"]
    if regulator_labels is not None:
        labels = np.asarray(regulator_labels)[keep]
        uniq = sorted(set(labels.tolist()))
        for lab in uniq[1:]:  # first level absorbed in the intercept
            cols.append((labels == lab).astype(float))
            names.append(f"regulator[{lab}]")
    X = np.column_stack(cols)
    return ols(X, np.log(y[keep]), names, seed=seed, n_boot=n_boot,
               excluded_rows=excluded, notes=notes)


def power_law_fit(x: np.ndarray, y: np.ndarray, *, seed: int = 20260730,
                 n_boot: int = 10000, notes: str = "") -> OLSResult:
    """``log y = c + s log x``; ``s`` is named ``exponent``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    excluded = int((~keep).sum())
    X = np.column_stack([np.ones(keep.sum()), np.log(x[keep])])
    res = ols(X, np.log(y[keep]), ["intercept", "exponent"], seed=seed,
              n_boot=n_boot, excluded_rows=excluded, notes=notes)
    return res


def null_test(estimate: float, stderr: float, null_value: float,
              dof: int) -> dict:
    """Two-sided t-test of ``H0: coefficient == null_value``."""
    from scipy import stats

    if not np.isfinite(stderr) or stderr <= 0 or dof <= 0:
        return dict(null_value=float(null_value), t_statistic=float("nan"),
                    p_value=float("nan"), dof=int(dof), reject_at_5pct=None)
    t = (estimate - null_value) / stderr
    p = float(2.0 * stats.t.sf(abs(t), dof))
    return dict(null_value=float(null_value), t_statistic=float(t), p_value=p,
                dof=int(dof), reject_at_5pct=bool(p < 0.05))
