from pathlib import Path
import json
import sympy as s

R = Path(__file__).resolve().parents[1]
A = Path(__file__).resolve().parent / "rerun"
A.mkdir(exist_ok=True)
checks = {}

# The central contraction is a finite sum, independent of any sampling law.
m = s.symbols("m", positive=True, integer=True)
p = s.symbols("p", integer=True)
central_gap = s.simplify(s.summation(p * (m - p), (p, 1, m - 1)) / 2
                         - m * (m**2 - 1) / 12)
checks["oscillator_central_sum_gap"] = str(central_gap)
assert central_gap == 0

# Compute the cutoff remainder as an exact differential-operator identity
# in the Bargmann representation, with an arbitrary charge and test function.
x, y, charge = s.symbols("x y a")
f = s.Function("f")(x, y)
def lp(v):
    return charge * s.diff(v, x) + 2 * x * s.diff(v, y)
def lm(v):
    return charge * x * v + y * s.diff(v, x)
def lzero(v):
    return charge**2 * v / 2 + x * s.diff(v, x) + 2 * y * s.diff(v, y)
remainder = s.expand(lp(lm(f)) - lm(lp(f)) - 2 * lzero(f))
assert s.simplify(remainder + 6 * y * s.diff(f, y)) == 0
checks["cutoff_remainder_L2_m1_nminus1"] = str(remainder)
# Its action vanishes on every level <=1 polynomial and is nonzero at level 2.
assert all(s.expand(lp(lm(v)) - lm(lp(v)) - 2 * lzero(v)) == 0
           for v in [s.Integer(1), x])
assert s.expand(lp(lm(y)) - lm(lp(y)) - 2 * lzero(y)) == -6 * y
checks["cutoff_remainder_filtered_domain_verified"] = True

# Normal ordering before the infrared subtraction gives an overall factor.
# Verify cancellation for an arbitrary neutral four-charge configuration.
a, b, c, ir = s.symbols("a b c C")
charges = [a, b, c, -a - b - c]
pair_sum = sum(charges[i] * charges[j] for i in range(4) for j in range(i + 1, 4))
sum_squares = sum(z**2 for z in charges)
assert s.expand(pair_sum + sum_squares / 2) == 0
exponent = s.expand(-ir * pair_sum - ir * sum_squares / 2)
assert exponent == 0
checks["neutral_vertex_ir_exponent_after_subtraction"] = str(exponent)

# The logarithmic Bessel difference can be Abel regulated and then undamped.
# Differentiate its closed form against the Laplace transform of J_0.
eps, r, rstar = s.symbols("epsilon r rstar", positive=True)
abel_difference = 2 * s.log((eps + s.sqrt(eps**2 + rstar**2)) /
                             (eps + s.sqrt(eps**2 + r**2)))
derivative_gap = s.simplify(s.diff(abel_difference, eps)
                          - 2 * (1 / s.sqrt(eps**2 + rstar**2)
                                 - 1 / s.sqrt(eps**2 + r**2)))
assert derivative_gap == 0
assert s.simplify(s.limit(abel_difference, eps, s.oo)) == 0
assert s.simplify(s.limit(abel_difference, eps, 0)
                  + 2 * s.log(r / rstar)) == 0
checks["scalar_bessel_difference_limit"] = "-2 log(r/rstar)"

# Point splitting of the Ramond kernel against the NS pole fixes the shift.
z = s.symbols("z", positive=True)
delta = s.symbols("delta", real=True)
ramond = (2 * z + delta) / (2 * s.sqrt(z * (z + delta)) * (-delta))
regular = s.series(ramond + 1 / delta, delta, 0, 2).removeO()
stress = s.simplify(-s.diff(regular, delta).subs(delta, 0) / 2)
assert s.simplify(stress - 1 / (16 * z**2)) == 0
checks["ramond_point_split_stress"] = str(stress)
pauli1 = s.Matrix([[0, 1], [1, 0]])
bzero = pauli1 / s.sqrt(2)
gzero = charge * bzero
central_charge = s.Rational(3, 2)
ground_lzero = charge**2 / 2 + s.Rational(1, 16)
assert gzero**2 == (ground_lzero - central_charge / 24) * s.eye(2)
checks["ramond_ground_supercharge_identity"] = True

# Recheck the archived November algorithms without rerunning the benchmark.
import importlib.util
import csv
import math
spec = importlib.util.spec_from_file_location("phase_analysis", R / "evidence/major_revision/analyze_revision.py")
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)
phase.ROOT = A
fock = phase.fock_checks()
checks["fock_readouts_checked"] = sum(row["samples"] for row in fock)
checks["raw_fock_admissible"] = sum(row["raw_exact_admissible"] for row in fock)
checks["projected_fock_positive"] = sum(row["projected_positive"] for row in fock)
assert checks["fock_readouts_checked"] == 20
assert checks["raw_fock_admissible"] == 0
assert checks["projected_fock_positive"] == 19
checks["exact_phase_quadrature_gaps"] = phase.phase_quadrature_check()
# The four-point fit uses the unchanged saved block means.
rows = [row for row in csv.DictReader((R / "evidence/four_point_width/finite_width_cumulant.csv").open())
        if row["regulator_class"] == "tapered_log_window" and row["in_fit_range"] == "True"]
x = [math.log(float(row["n_features"])) for row in rows]
y = [math.log(abs(float(row["cumulant4"]))) for row in rows]
xm, ym = sum(x)/len(x), sum(y)/len(y)
sxx = sum((v-xm)**2 for v in x)
slope = sum((u-xm)*(v-ym) for u,v in zip(x,y))/sxx
intercept = ym-slope*xm
se = math.sqrt(sum((v-intercept-slope*u)**2 for u,v in zip(x,y))/(len(x)-2)/sxx)
assert abs(slope-float(rows[0]["fit_exponent"])) < 1e-10
assert abs(se-float(rows[0]["fit_exponent_stderr"])) < 1e-10
checks["saved_four_point_exponent"] = slope
checks["saved_four_point_exponent_standard_error"] = se

checks["all_passed"] = True
checks["method"] = "Exact symbolic identities; no new numerical evidence study"
(A / "analytical_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
