"""Independent symbolic Wick calculations and finite-mode consistency checks.

These verify the restored local formulas; they are not new Monte Carlo,
BRST, full bosonization, or modular-consistency measurements.
"""
from pathlib import Path
import json
import math
import numpy as np
import sympy as s
from scipy.integrate import quad
from scipy.special import jv

ROOT=Path(__file__).resolve().parents[1]
z,w,lam=s.symbols('z w lambda',positive=True)
delta=z-w
A,B=1-lam,-lam

def contraction(left,right,kappa):
    field_l,deriv_l=left
    field_r,deriv_r=right
    coefficient=kappa if (field_l,field_r)==('F','G') else (
        1 if (field_l,field_r)==('G','F') else 0)
    return s.diff(coefficient/delta,z,deriv_l,w,deriv_r)

stress_terms=[(A,('F',1),('G',0)),(B,('F',0),('G',1))]
ghosts={}
for parity,kappa,name in [(1,1,'bc'),(0,-1,'beta_gamma')]:
    tt=0
    for ca,l1,l2 in stress_terms:
        for cb,r1,r2 in stress_terms:
            # Normal-ordering excludes the two internal contractions.
            cross=(-1)**parity*contraction(l1,r1,kappa)*contraction(l2,r2,kappa)
            cross+=contraction(l1,r2,kappa)*contraction(l2,r1,kappa)
            tt+=ca*cb*cross
    central=s.factor(2*tt*delta**4)
    expected=(-1 if name=='bc' else 1)*2*(6*lam**2-6*lam+1)
    assert s.simplify(central-expected)==0
    # Derive the field weights and derivative residues by graded single
    # contractions, retaining the Taylor term of each uncontracted field.
    ward={}
    for ext in ['F','G']:
        h=0;derivative=0
        for coefficient,left,right in stress_terms:
            for i,(contracted,remaining) in enumerate([(left,right),(right,left)]):
                sign=(-1)**(parity*(1-i))
                value=s.simplify(sign*coefficient*contraction(contracted,(ext,0),kappa))
                if remaining[1]==0:
                    h+=s.simplify(value*delta**2)
                    derivative+=s.simplify(value*delta**2)
                else:
                    derivative+=s.simplify(value*delta)
        h=s.simplify(h);derivative=s.simplify(derivative)
        assert h==(lam if ext=='F' else 1-lam)
        assert derivative==1
        ward[ext]={'weight':str(h),'derivative_pole':str(derivative)}
    ghosts[name]={'central_charge':str(central),'ward':ward,
        'at_critical_weight':str(central.subs(lam,2 if name=='bc' else s.Rational(3,2)))}

# One Majorana stress tensor has two graded cross-contraction channels.
K=1/delta
tt_majorana=s.Rational(1,4)*(-K*s.diff(K,z,w)+s.diff(K,w)*s.diff(K,z))
assert s.simplify(tt_majorana-1/(4*delta**4))==0
# Independent species in i:psi_1 psi_2: have one crossed contraction.
dirac_jj=s.I**2*(-1)*K*K
assert s.simplify(dirac_jj-1/delta**2)==0

# The native feature map contracts to the audited momentum kernel.
x,y,a=s.symbols('x y a')
native=s.expand_trig((a*s.cos(x))*(-a*s.sin(y))-(-a*s.sin(x))*(a*s.cos(y)))
assert s.trigsimp(native-a*a*s.sin(x-y))==0

# The Ramond kernel, its short-distance normalization, and point-split
# stress tensor are checked without using an NS multiplicative twist.
SR=(z+w)/(2*s.sqrt(z*w)*(z-w))
series=(s.Rational(1,2)+(w/z)/(1-w/z))/s.sqrt(z*w)
assert s.simplify(SR-series)==0
assert s.simplify(SR+SR.xreplace({z:w,w:z}))==0
residue=s.limit(delta*SR,z,w)
assert residue==1
ramond_T=s.simplify(-s.Rational(1,2)*s.limit(s.diff(SR,w)-1/delta**2,w,z))
assert s.simplify(ramond_T-1/(16*z*z))==0
old_twist_residue=s.simplify(s.limit(delta/(s.sqrt(z*w)*delta),w,z))
assert old_twist_residue==1/z

def kron_all(items):
    out=np.array([[1.]])
    for item in items:out=np.kron(out,item)
    return out

identity=np.eye(2);sz=np.diag([1.,-1.]);sx=np.array([[0.,1.],[1.,0.]])
ann=np.array([[0.,1.],[0.,0.]])
fermions=[kron_all([sz]*j+[ann]+[identity]*(2-j-1)) for j in range(2)]
P=kron_all([sz,sz])
b={0:np.kron(sx,P)/np.sqrt(2)}
for j,f in enumerate(fermions,1):
    b[j]=np.kron(identity,f);b[-j]=b[j].T
eye=np.eye(8)
max_car=max(float(np.linalg.norm(b[i]@b[j]+b[j]@b[i]-(eye if i+j==0 else 0)))
            for i in b for j in b)
total_parity=np.kron(sz,P)
max_parity=max(float(np.linalg.norm(total_parity@v+v@total_parity)) for v in b.values())
assert max_car<2e-15 and max_parity<2e-15

# Check the angular Fourier integrals in several spin/orientation regimes.
angular_errors=[]
for kr in [.05,.8,4.,13.]:
    for theta in [0.,.42,1.8]:
        for spin in [1,2]:
            def integrand(angle):
                wave=math.sin(kr*math.cos(angle-theta)) if spin==1 else math.cos(kr*math.cos(angle-theta))
                return np.exp(-1j*spin*angle)*wave/(2*np.pi)
            value=quad(lambda t:integrand(t).real,0,2*np.pi,epsabs=1e-12,limit=150)[0]
            value+=1j*quad(lambda t:integrand(t).imag,0,2*np.pi,epsabs=1e-12,limit=150)[0]
            target=(1 if spin==1 else -1)*np.exp(-1j*spin*theta)*jv(spin,kr)
            angular_errors.append(float(abs(value-target)))
assert max(angular_errors)<1e-12

report=dict(ghost_wick_derivations=ghosts,
    majorana_stress_central_charge='1/2',dirac_current_level=1,
    native_majorana_feature_contraction='exact symbolic identity',
    ramond_kernel='exact geometric-series identity',ramond_pole_residue=str(residue),
    ramond_stress_expectation=str(ramond_T),ramond_ground_weight='1/16',
    multiplicative_old_NS_twist_pole_residue=str(old_twist_residue),
    ramond_finite_mode_CAR_max_error=max_car,
    ramond_parity_max_error=max_parity,
    angular_quadrature_comparisons=len(angular_errors),
    angular_max_abs_error=max(angular_errors),
    scope='Symbolic local formulas and finite-mode consistency; no new Monte Carlo, full bosonization, BRST, or modular-consistency claims.')
(ROOT/'audit/recovered_physics_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
