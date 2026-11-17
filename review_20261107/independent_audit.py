"""Independent checks of the manuscript; does not rerun network simulations."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import jv
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-root',type=Path,default=SNAP)
args=parser.parse_args()
ORIGIN=args.evidence_root.resolve()
OUT = ROOT / 'review_20261107' / 'rerun'
OUT.mkdir(exist_ok=True,parents=True)
SNAP.mkdir(exist_ok=True)
for name in ('src','scripts','tests','configs','results','figures_jhep_frozen',
             'figures_jhep_submission_hardening','tables_jhep_frozen',
             'tables_jhep_submission_hardening'):
    if ORIGIN!=SNAP.resolve():
        shutil.copytree(ORIGIN/name, SNAP/name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','.DS_Store'))
for name in ('Virasoro_JHEP_Frozen.bib','Draft-JHEP-Frozen.tex'):
    if ORIGIN!=SNAP.resolve(): shutil.copy2(ORIGIN/name,SNAP/name)

manifest = json.loads((SNAP/'results/jhep_submission_hardening/manifest.json').read_text())
entries = manifest['artifacts'] if 'artifacts' in manifest else manifest['files']
integrity=[]
for e in entries:
    p=SNAP/e['path']
    actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    integrity.append(dict(path=e['path'],expected=e['sha256'],actual=actual,
                          passed=actual==e['sha256']))
assert all(e['passed'] for e in integrity)
(OUT/'release_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')

# Recompute TT inference from the unchanged independent block vectors.
blocks = pd.read_csv(SNAP/'results/jhep_submission_hardening/gate03_q_ratio_blocks.csv')
results=[]
for variant,d in blocks.groupby('estimator_variant'):
    X=d.pivot(index='seed_name',columns='separation_index',values='q').to_numpy()
    assert X.shape==(12,4) and np.isfinite(X).all()
    n,p=X.shape
    mean=X.mean(0); S=np.cov(X,rowvar=False,ddof=1)
    inv=np.linalg.inv(S); one=np.ones(p)
    qhat=float(one@inv@mean/(one@inv@one))
    se=float((n*(one@inv@one))**-.5)
    delta=mean-qhat
    Q=float(n*delta@inv@delta)
    # Three fixed geometry contrasts remove the unknown common mean.
    C=np.column_stack((np.eye(p-1),-np.ones(p-1)))
    Y=X@C.T; v=Y.mean(0); Sy=np.cov(Y,rowvar=False,ddof=1)
    Tcontrast=float(n*v@np.linalg.solve(Sy,v))
    assert np.isclose(Q,Tcontrast,rtol=1e-10)
    k=p-1
    F=(n-k)*Tcontrast/(k*(n-1))
    T2=float(n*(mean-one)@inv@(mean-one))
    jointF=(n-p)*T2/(p*(n-1))
    # Sample covariance also makes the fitted GLS weights random. The old
    # one-dimensional t-test for qhat is retained only as legacy output.
    crit=np.sqrt(p*(n-1)/(n-p)*stats.f.ppf(.95,p,n-p))
    simultaneous_ci=[qhat-crit*se,qhat+crit*se]
    jack=[]
    for i in range(n):
        Xi=np.delete(X,i,axis=0); si=np.cov(Xi,rowvar=False,ddof=1)
        a=np.linalg.solve(si,one)
        jack.append(float(a@Xi.mean(0)/(a@one)))
    results.append(dict(estimator=variant,n_blocks=n,n_geometries=p,
        geometry_means=mean.tolist(),geometry_se=np.sqrt(S.diagonal()/n).tolist(),
        gls_common_ratio=qhat,gls_plugin_se=se,
        legacy_scalar_t_p=float(2*stats.t.sf(abs((qhat-1)/se),n-1)),
        residual_Q=Q,legacy_chi2_p=float(stats.chi2.sf(Q,k)),
        contrast_T2=Tcontrast,contrast_F=F,contrast_dof=[k,n-k],
        contrast_F_p=float(stats.f.sf(F,k,n-k)),
        all_geometry_T2=T2,all_geometry_F=jointF,
        all_geometry_F_p=float(stats.f.sf(jointF,p,n-p)),
        conservative_95pct_common_mean_interval_from_hotelling_projection=simultaneous_ci,
        leave_one_block_out_qhat_range=[min(jack),max(jack)],
        assumptions='Independent identically distributed approximately Gaussian block vectors; exact F calibration under multivariate normality. No geometry or block excluded.'))
(OUT/'tt_finite_block_audit.json').write_text(json.dumps(results,indent=2)+'\n')

# Independent direct J2 integral, J0 integration-by-parts formula, and bound.
nodes,weights=leggauss(24)
rows=[]
for T in (2.,4.,6.,8.):
  for tau in (.1,.15,.3):
    lo,hi=np.exp(-T),np.exp(T); ell=2*T
    km,kp=np.exp(-T+tau*ell),np.exp(-T+(1-tau)*ell)
    for r in (.25,.7,1.6):
      edges=np.unique(np.r_[lo,km,kp,hi,np.geomspace(lo,hi,150),np.linspace(lo,hi,int(np.ceil((hi-lo)*r*4/np.pi))+1)])
      a,b=edges[:-1,None],edges[1:,None]
      k=(a+b)/2+(b-a)/2*nodes; w=(b-a)/2*weights
      u=np.log(k/lo)/ell
      W=np.ones_like(k); Wu=np.zeros_like(k); Wuu=np.zeros_like(k)
      for mask,t,sgn in ((u<tau,u,1),(u>1-tau,1-u,-1)):
        arg=np.pi*t[mask]/tau
        W[mask]=.5*(1-np.cos(arg))
        Wu[mask]=sgn*np.pi/(2*tau)*np.sin(arg)
        Wuu[mask]=np.pi**2/(2*tau**2)*np.cos(arg)
      direct=.5*np.sum(w*k*W*jv(2,k*r))
      bracket=(2*Wu/ell+Wuu/ell**2)/k
      ibp=.5/r**2*np.sum(w*bracket*jv(0,k*r))
      M=2+np.pi/(tau*ell)
      bound=M*(km**2/8+2/(2*r**2.5*np.sqrt(kp)))
      error=abs(direct-r**-2)
      assert abs(direct-ibp)<2e-8
      if r*kp>=1: assert error<=bound+1e-8
      # Native NS first-derivative kernel and analogous separated-point bound.
      native=np.sum(w*W*jv(1,k*r))
      ns_ibp=np.sum(w*Wu/(ell*k)*jv(0,k*r))/r
      ns_bound=km**2*r/4+2/(r**1.5*np.sqrt(kp))
      assert abs(native-ns_ibp)<2e-8
      if r*kp>=1: assert abs(native-1/r)<=ns_bound+1e-8
      rows.append(dict(T=T,tau=tau,r=r,direct=direct,ibp=ibp,
        identity_error=abs(direct-ibp),kernel_error=error,bound=bound,
        bound_eligible=r*kp>=1,native_ns=native,ns_ibp=ns_ibp,
        ns_error=abs(native-1/r),ns_bound=ns_bound))
pd.DataFrame(rows).to_csv(OUT/'independent_bessel_checks.csv',index=False)

# Noncommuting D and G exercise the stated principal Fock canonicalizer.
rng=np.random.default_rng(20261107)
fock=[]
for L in (1,2,4,8):
  for repeat in range(20):
    Z=rng.normal(size=(L,L))+1j*rng.normal(size=(L,L))
    G=Z@Z.conj().T+np.eye(L)
    val,U=np.linalg.eigh(G)
    M=np.diag(np.sqrt(np.arange(1,L+1)))@(U*val**-.5)@U.conj().T
    err=np.linalg.norm(M@G@M.conj().T-np.diag(np.arange(1,L+1)))
    assert err<1e-10
    fock.append(err)

summary=dict(release_files_checked=len(integrity),release_files_passed=sum(e['passed'] for e in integrity),
    bessel_cases=len(rows),max_current_identity_error=max(r['identity_error'] for r in rows),
    max_native_identity_error=max(abs(r['native_ns']-r['ns_ibp']) for r in rows),
    eligible_bounds=int(sum(r['bound_eligible'] for r in rows)),
    fock_cases=len(fock),max_fock_error=max(fock),tt=results)
(OUT/'independent_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
