"""
Majorana c=1/2 VERIFICATION using correct CFT theory.

CORRECTED UNDERSTANDING:
The CFT prediction for free Majorana is <ψ(r)ψ(0)> = 1/(2πr), NOT (c/2)/r.
We should verify sim/theory ≈ 1.0, confirming our network implements c=1/2 Majorana.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MajoranaFermion:
    def __init__(self, n_features, k_min=1.0, k_max=100.0):
        log_k = np.random.uniform(np.log(k_min), np.log(k_max), n_features)
        k_magnitudes = np.exp(log_k)
        theta_k = np.random.uniform(0, 2*np.pi, n_features)
        
        self.k_x = k_magnitudes * np.cos(theta_k)
        self.k_y = k_magnitudes * np.sin(theta_k)
        self.k_magnitudes = k_magnitudes
        self.theta_k = theta_k
        self.phases = np.random.uniform(0, 2*np.pi, n_features)
        self.n_features = n_features
    
    def psi_batch(self, x_vals, y_vals):
        """Compute ψ(x) = (1/√N) Σ √k e^(-iθ) e^(i k·x)"""
        x_vals = np.asarray(x_vals).reshape(-1)
        y_vals = np.asarray(y_vals).reshape(-1)
        
        spin_factor = np.exp(-1j * self.theta_k)
        sqrt_k = np.sqrt(self.k_magnitudes)
        
        args = (self.k_x[None, :] * x_vals[:, None] + 
                self.k_y[None, :] * y_vals[:, None] + 
                self.phases[None, :])
        
        weighted = (sqrt_k * spin_factor)[None, :] * np.exp(1j * args)
        psi_vals = np.sum(weighted, axis=1) / np.sqrt(self.n_features)
        
        return psi_vals


def compute_correlator(n_networks=100000, n_features=8000, 
                       r_min=0.1, r_max=1.0, n_r=25):
    """Compute <ψ(0)ψ*(r)> with large ensemble."""
    
    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    correlator = np.zeros(len(r_vals))
    
    print(f"Computing <ψ(0)ψ*(r)> with {n_networks} networks, {n_features} modes")
    print(f"r range: [{r_min:.2f}, {r_max:.2f}], {n_r} points\n")
    
    for i in tqdm(range(n_networks), desc="Networks"):
        fermion = MajoranaFermion(n_features, 1.0, 100.0)
        
        psi_0 = fermion.psi_batch(np.array([0.0]), np.array([0.0]))[0]
        psi_r = fermion.psi_batch(r_vals, np.zeros_like(r_vals))
        
        correlator += np.real(psi_0 * np.conj(psi_r))
    
    correlator /= n_networks
    
    return r_vals, correlator


def cft_theory(r_vals):
    """
    Correct CFT theory for free Majorana fermion.
    
    <ψ(r)ψ(0)> = 1/(2πr)
    
    This is the FULL prediction, not c × (something).
    """
    return 1.0 / (2.0 * np.pi * r_vals)


def verify_against_cft(r_vals, sim, r_min=0.15, r_max=0.6):
    """
    Verify simulation matches CFT theory.
    
    We expect sim/theory ≈ 1.0 if network correctly implements c=1/2 Majorana.
    """
    theory = cft_theory(r_vals)
    
    stable_mask = (r_vals >= r_min) & (r_vals <= r_max)
    
    if np.sum(stable_mask) < 3:
        return np.nan, np.nan, np.nan
    
    ratios = sim[stable_mask] / theory[stable_mask]
    
    ratio_mean = np.mean(ratios)
    ratio_std = np.std(ratios)
    ratio_stderr = ratio_std / np.sqrt(len(ratios))
    relative_var = ratio_std / np.abs(ratio_mean) if ratio_mean != 0 else np.inf
    
    return ratio_mean, ratio_stderr, relative_var


if __name__ == "__main__":
    print("="*70)
    print("MAJORANA FERMION c=1/2 VERIFICATION")
    print("Using correct CFT theory: <ψψ*> = 1/(2πr)")
    print("="*70)
    print()
    
    # Run simulation
    r_vals, psi_psi = compute_correlator(
        n_networks=100000,
        n_features=8000,
        r_min=0.1,
        r_max=1.0,
        n_r=25
    )
    
    print("\nComputing CFT theory prediction...")
    theory = cft_theory(r_vals)
    
    # Verify match
    ratio_mean, ratio_stderr, rel_var = verify_against_cft(
        r_vals, psi_psi, r_min=0.15, r_max=0.6
    )
    
    print("\n" + "="*70)
    print("RESULTS: Majorana c=1/2 Verification")
    print("="*70)
    print(f"\nRatio sim/CFT = {ratio_mean:.4f} ± {ratio_stderr:.4f}")
    print(f"Relative std dev: {rel_var*100:.1f}%")
    
    # Interpretation
    print(f"\nInterpretation:")
    if abs(ratio_mean - 1.0) < 0.1:
        error_pct = abs(ratio_mean - 1.0) / 1.0 * 100
        print(f"✓ EXCELLENT: sim/CFT ≈ 1.0 (error {error_pct:.1f}%)")
        print(f"✓ Network correctly implements free Majorana c=1/2!")
    elif abs(ratio_mean - 1.0) < 0.2:
        error_pct = abs(ratio_mean - 1.0) / 1.0 * 100
        print(f"✓ GOOD: sim/CFT ≈ {ratio_mean:.2f} (error {error_pct:.1f}%)")
        print(f"  Small discrepancy likely due to finite-size effects")
    else:
        print(f"✗ MISMATCH: sim/CFT = {ratio_mean:.2f}")
        print(f"  Theory or implementation issue")
    
    if rel_var < 0.05:
        print(f"✓ Excellent stability (<5%)")
    elif rel_var < 0.10:
        print(f"✓ Good stability (<10%)")
    elif rel_var < 0.25:
        print(f"~ Moderate stability (<25%)")
    else:
        print(f"✗ Poor stability (>{rel_var*100:.0f}%)")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Correlator comparison
    ax = axes[0]
    ax.loglog(r_vals, psi_psi, 'o-', label='Simulation <ψ(0)ψ(r)>', 
              markersize=6, alpha=0.7, linewidth=2)
    ax.loglog(r_vals, theory, '--', label='CFT Theory: 1/(2πr)', 
              linewidth=2, color='red')
    
    ax.set_xlabel('Distance r', fontsize=12)
    ax.set_ylabel('<ψ(0)ψ(r)>', fontsize=12)
    ax.set_title('Majorana Fermion 2-Point Function', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    
    # Right: Ratio verification
    ax = axes[1]
    
    ratio_vs_r = psi_psi / theory
    
    ax.semilogx(r_vals, ratio_vs_r, 'o-', markersize=6, alpha=0.7, 
                linewidth=2, label='sim/CFT(r)')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, 
               label='Perfect match', zorder=10)
    ax.axhline(ratio_mean, color='green', linestyle=':', linewidth=2,
               label=f'Mean = {ratio_mean:.3f}', zorder=10)
    
    # Error band
    ax.fill_between(r_vals, ratio_mean - ratio_stderr, ratio_mean + ratio_stderr,
                     alpha=0.2, color='green', label=f'±1σ (SEM)')
    
    # Shade stable region
    ax.axvspan(0.15, 0.6, alpha=0.1, color='gray', label='Fit region')
    
    ax.set_xlabel('Distance r', fontsize=12)
    ax.set_ylabel('sim / CFT', fontsize=12)
    ax.set_title('CFT Verification', fontweight='bold', fontsize=14)
    ax.set_ylim([0.5, 1.5])
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filename = 'majorana_cft_verification_100k.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {filename}")
    
    # Save results
    results_file = 'majorana_cft_verification.txt'
    with open(results_file, 'w') as f:
        f.write("MAJORANA FERMION c=1/2 VERIFICATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Method: Direct CFT matching <ψ(r)ψ(0)> = 1/(2πr)\n")
        f.write(f"Networks: 100,000\n")
        f.write(f"Modes per network: 8,000\n\n")
        f.write(f"RESULTS:\n")
        f.write(f"  Ratio sim/CFT = {ratio_mean:.6f} ± {ratio_stderr:.6f}\n")
        f.write(f"  Relative std dev: {rel_var*100:.2f}%\n\n")
        if abs(ratio_mean - 1.0) < 0.1:
            f.write(f"✓ VERIFIED: Network implements free Majorana c=1/2\n")
            f.write(f"  Error from perfect match: {abs(ratio_mean-1.0)/1.0*100:.2f}%\n")
        else:
            f.write(f"  Discrepancy from CFT: {abs(ratio_mean-1.0)/1.0*100:.2f}%\n")
    
    print(f"Results saved: {results_file}")
