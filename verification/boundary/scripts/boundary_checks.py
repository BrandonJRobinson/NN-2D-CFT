"""Reproducible checks of the reflected scalar and doubled fermion kernels.

Scalar error bars concern independent frequency/angle batches after exact
phase averaging. They are not estimates of finite-ensemble Wick errors.
The fermion panel is a deterministic Cauchy-series calculation.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import j0
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/boundary'
FIG = ROOT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED, N, BLOCKS = 20261107, 16384, 32
IR, UV, TAU = 1e-3, 1e3, .15
LOG = np.log(UV/IR)
LW = (1-TAU)*LOG
rng = np.random.default_rng(SEED)

def window(u):
    u = np.asarray(u)
    w = np.ones_like(u)
    low, high = u < TAU, u > 1-TAU
    w[low] = .5*(1-np.cos(np.pi*u[low]/TAU))
    w[high] = .5*(1-np.cos(np.pi*(1-u[high])/TAU))
    return np.where((u >= 0) & (u <= 1), w, 0.)

def sample_momenta(size):
    accepted = []
    count = 0
    while count < size:
        u = rng.random(int((size-count)/(1-TAU)*1.1)+16)
        take = u[rng.random(u.size) < window(u)]
        accepted.append(take)
        count += take.size
    u = np.concatenate(accepted)[:size]
    angle = rng.uniform(0, 2*np.pi, size)
    k = IR*np.exp(LOG*u)
    return k*np.cos(angle), k*np.sin(angle)

height = np.geomspace(.1, 1.6, 17)
yref = .4
batches = []
for _ in range(BLOCKS):
    kx, ky = sample_momenta(N)
    batches.append(2*LW*np.mean(
        np.cos(2*ky[:, None]*height)-np.cos(2*ky[:, None]*yref), axis=0))
batches = np.asarray(batches)
mean = batches.mean(axis=0)
sem = batches.std(axis=0, ddof=1)/np.sqrt(BLOCKS)

def radial_image_difference(y, epsabs, spacing_factor=1.):
    # Split at taper joins and at oscillation-scale intervals in k.
    edges = [IR, IR*np.exp(TAU*LOG), IR*np.exp((1-TAU)*LOG), UV]
    maxdist = max(2*y, 2*yref)
    cuts = np.unique(np.r_[edges, np.arange(IR, UV, spacing_factor*np.pi/maxdist)])
    def integrand(k):
        u = np.log(k/IR)/LOG
        weight = float(window(u))
        return 2*weight*(j0(2*k*y)-j0(2*k*yref))/k
    return sum(quad(integrand, a, b, epsabs=epsabs/(len(cuts)-1),
                    epsrel=1e-11, limit=100)[0]
               for a, b in zip(cuts[:-1], cuts[1:]))

reference = np.array([radial_image_difference(y, 1e-8) for y in height])
reference_tight = np.array([radial_image_difference(y, 1e-10, .5) for y in height])
target = -2*np.log(height/yref)
design = np.c_[np.ones(height.size), np.log(height/yref)]
fit_blocks = np.linalg.lstsq(design, batches.T, rcond=None)[0][1]
slope = float(fit_blocks.mean())
slope_sem = float(fit_blocks.std(ddof=1)/np.sqrt(BLOCKS))
slope_reg = float(np.linalg.lstsq(design, reference, rcond=None)[0][1])

# Exact feature constraints use one and the same realization at reflected
# points. Independent bulk draws would not impose a boundary condition.
kx, ky = sample_momenta(2048)
gamma = rng.uniform(0, 2*np.pi, kx.size)
x = np.linspace(-1, 1, 11)
phase = kx[:, None]*x+gamma[:, None]
bulk = np.sqrt(4*LW/kx.size)*np.cos(phase).sum(axis=0)
dx = -np.sqrt(4*LW/kx.size)*(kx[:, None]*np.sin(phase)).sum(axis=0)
dy = -np.sqrt(4*LW/kx.size)*(ky[:, None]*np.sin(phase)).sum(axis=0)
dirichlet_residual = float(np.max(np.abs((bulk-bulk)/np.sqrt(2))))
neumann_residual = float(np.max(np.abs((dy-dy)/np.sqrt(2))))
gluing = {}
for sigma in [-1, 1]:
    bx = (dx+sigma*dx)/np.sqrt(2)
    by = (dy-sigma*dy)/np.sqrt(2)
    J = .5j*(bx-1j*by)
    Jbar = .5j*(bx+1j*by)
    gluing[str(sigma)] = float(np.max(np.abs(J-sigma*Jbar)))

def cauchy_ordered(u, v, length):
    if abs(u) < abs(v):
        return -cauchy_ordered(v, u, length)
    return (1-(v/u)**length)/(u-v)

def spinor(z, w, eta, length):
    return np.array([
        [cauchy_ordered(z,w,length), eta*cauchy_ordered(z,w.conjugate(),length)],
        [eta*cauchy_ordered(z.conjugate(),w,length),
         cauchy_ordered(z.conjugate(),w.conjugate(),length)]])

exchange_residual = max(float(np.max(np.abs(
    spinor(1.2+.15j,.4+.2j,eta,32)
    +spinor(.4+.2j,1.2+.15j,eta,32).T))) for eta in [-1,1])
fermion_gluing = max(float(np.max(np.abs(
    spinor(1.2+0j,.4+.2j,eta,32)[0]
    -eta*spinor(1.2+0j,.4+.2j,eta,32)[1]))) for eta in [-1,1])
separation = np.geomspace(.1, 3, 60)
epsilon, length = .05, 32
z = separation+1j*epsilon
wbar = -1j*epsilon
mixed = (1-(wbar/z)**length)/(z-wbar)
mixed_target = 1/(separation+2j*epsilon)
mixed_relerr = float(np.max(np.abs(mixed/mixed_target-1)))

report = dict(seed=SEED, features_per_batch=N, independent_batches=BLOCKS,
    IR=IR, UV=UV, tau=TAU, scalar_normalization='K=-log|z-w|^2',
    scalar_statistic='phase-averaged image covariance difference, yref=0.4',
    measured_log_slope=slope, log_slope_standard_error=slope_sem,
    regulated_log_slope=slope_reg, continuum_log_slope=-2,
    quadrature_tightening_max_abs=float(np.max(np.abs(reference-reference_tight))),
    scalar_largest_pointwise_standardized_difference=float(np.max(
        np.abs((mean-reference)[sem>0]/sem[sem>0]))),
    scalar_continuum_max_abs_bias=float(np.max(np.abs(reference-target))),
    dirichlet_feature_residual=dirichlet_residual,
    neumann_derivative_residual=neumann_residual,
    scalar_current_gluing_residuals=gluing,
    fermion_exchange_residual=exchange_residual,
    fermion_boundary_gluing_residual=fermion_gluing,
    fermion_series_terms=length, fermion_epsilon=epsilon,
    fermion_max_relative_truncation_error=mixed_relerr,
    scope='Construction checks and two-point kernels; no annulus/Cardy or independent supersymmetry measurement.')
assert report['quadrature_tightening_max_abs'] < 1e-7
assert report['scalar_largest_pointwise_standardized_difference'] < 5
assert abs(slope-slope_reg) < 5*slope_sem
assert max(dirichlet_residual,neumann_residual,*gluing.values(),
           exchange_residual,fermion_gluing) < 1e-12
assert mixed_relerr < 1e-10
(OUT/'boundary_checks.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(OUT/'boundary_batches.npz',height=height,yref=yref,
    batches=batches,mean=mean,sem=sem,regulated=reference,continuum=target,
    separation=separation,mixed=mixed,mixed_target=mixed_target)
np.savetxt(OUT/'scalar_image.csv',np.c_[height,mean,sem,reference,target],
    delimiter=',',header='height,mean,standard_error,regulated,continuum',comments='')

plt.rcParams.update({'font.family':'serif','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig, axes = plt.subplots(1,2,figsize=(9,3.25),constrained_layout=True)
ax=axes[0]
ax.errorbar(height,mean,yerr=sem,fmt='o',ms=3.3,color='#0072B2',
    capsize=2,label='Reflected features')
ax.plot(height,reference,color='#D55E00',lw=1.5,label='Regulated kernel')
ax.plot(height,target,'--',color='.4',lw=1,label=r'$-2\log(y/y_0)$')
ax.set(xscale='log',xlabel=r'Height $y$',ylabel=r'$K(2iy)-K(2iy_0)$',
    title='(a) Neumann image contribution')
ax.legend(frameon=False,fontsize=8)
ax=axes[1]
ax.loglog(separation,np.abs(mixed_target),color='#0072B2',lw=1.5,
    label=r'$1/\sqrt{x^2+4\varepsilon^2}$')
ax.loglog(separation[::4],np.abs(mixed[::4]),'o',mfc='white',ms=4,
    color='#D55E00',label='32-term Cauchy series')
ax.loglog(separation,1/separation,'--',color='.4',lw=1,label=r'$1/x$')
ax.set(xlabel=r'Horizontal separation $x$',ylabel=r'$|S_{LR}|$',
    title=r'(b) Mixed fermion kernel, $\varepsilon=0.05$')
ax.legend(frameon=False,fontsize=8)
fig.savefig(FIG/'boundary_kernels.pdf',bbox_inches='tight')
fig.savefig(OUT/'boundary_kernels.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print(json.dumps(report,indent=2))
