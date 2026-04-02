import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import j0
from tqdm import tqdm

class ChiralBosonBatched:
    def __init__(self, n_features, k_min=1.0, k_max=100.0):
        self.n_features = n_features
        self.k_min = k_min
        self.k_max = k_max
        
        # --- PRECISE CONTINUUM NORMALIZATION ---
        # We define the field such that in the continuum limit, 
        # the structure function D(r) scales with a prefactor of 2.0 * log(r).
        # A_squared / Volume = 2.0
        
        self.log_vol = np.log(k_max / k_min)
        self.norm = np.sqrt(2 * self.log_vol) 
        self.amp = (1.0 / np.sqrt(n_features)) * self.norm

    def process_batch(self, batch_size, r_points, alphas):
        # 1. Sample Wavevectors (Log-Uniform)
        log_k = np.random.uniform(np.log(self.k_min), np.log(self.k_max), (batch_size, self.n_features))
        k_mags = np.exp(log_k)
        
        # 2. Random Directions & Phases
        k_angles = np.random.uniform(0, 2*np.pi, (batch_size, self.n_features))
        phases = np.random.uniform(0, 2*np.pi, (batch_size, self.n_features))
        
        # 3. Compute Field Difference phi(0) - phi(r) directly
        # We assume points are along x-axis for simplicity (isotropic)
        # phi(0) = sum cos(phases)
        # phi(r) = sum cos(k*r*cos(theta) + phases)
        
        # k_x projection
        kx = k_mags * np.cos(k_angles)
        
        # We compute diff for each r point
        # Shape: (Batch, N_Feats, N_R)
        r_exp = r_points.reshape(1, 1, -1)
        kx_exp = kx[:, :, np.newaxis]
        ph_exp = phases[:, :, np.newaxis]
        
        # phi(0) - phi(r)
        # term(0) = cos(ph)
        # term(r) = cos(kx*r + ph)
        # diff = cos(ph) - cos(kx*r + ph)
        #      = -2 sin(kx*r/2 + ph + ...) ... let's just compute directly
        
        val_0 = np.cos(ph_exp)
        val_r = np.cos(kx_exp * r_exp + ph_exp)
        
        # Field Difference
        diff = self.amp * np.sum(val_0 - val_r, axis=1) # Sum over features -> (Batch, N_R)
        
        results = {}
        for alpha in alphas:
            # Correlator < V_a(0) V_-a(r) > = < exp(i * alpha * (phi(0) - phi(r))) >
            # We take Real part (expectation is real)
            op = np.exp(1j * alpha * diff)
            results[alpha] = np.mean(np.real(op), axis=0)
            
        return results

def exact_structure_function(r_vals, k_min, k_max):
    """
    Computes the exact integral for the two-point structure function
    with finite cutoffs.
    D(r) = 2 * Int_{k_min}^{k_max} (1 - J0(kr))/k dk
    """
    D_vals = []
    # FIX: Removed division by log(vol). 
    # The simulation produces D(r) = 2 * Integral(dk/k ...).
    # Normalizing by log(vol) would remove the logarithmic growth we want to test.
    norm_factor = 2.0 
    
    for r in r_vals:
        # Integrand: (1 - J0(k*r)) / k
        def integrand(k):
            return (1.0 - j0(k*r)) / k
        
        val, err = quad(integrand, k_min, k_max)
        D_vals.append(norm_factor * val)
        
    return np.array(D_vals)

def run_precision_test():
    print("==================================================")
    print("      VERTEX OPERATOR PRECISION TEST")
    print("==================================================")
    
    # Configuration
    N_NETS = 5000000 # Increased for better SNR
    BATCH_SIZE = 5000
    N_FEATS = 4000 
    K_MIN = 1.0
    K_MAX = 100.0
    
    # Alphas to test
    # p = alpha^2. For alpha=1.5, p=2.25 (very fast decay)
    alphas = [0.5, 1.0, 1.414] 
    
    # Points (Avoid extreme UV/IR to keep signal measurable)
    r_vals = np.logspace(-1.5, -0.2, 20) 
    
    # Initialize
    model = ChiralBosonBatched(N_FEATS, K_MIN, K_MAX)
    
    # Accumulators
    accumulators = {a: np.zeros_like(r_vals) for a in alphas}
    
    # Simulation Loop
    n_batches = N_NETS // BATCH_SIZE
    print(f"Simulating {N_NETS} networks...")
    
    for _ in tqdm(range(n_batches)):
        batch_res = model.process_batch(BATCH_SIZE, r_vals, alphas)
        for a in alphas:
            accumulators[a] += batch_res[a]
            
    # Averaging
    final_corrs = {a: accumulators[a] / n_batches for a in alphas}
    
    # --- Exact Theory Calculation ---
    print("\nComputing Exact Finite-Cutoff Theory...")
    D_exact = exact_structure_function(r_vals, K_MIN, K_MAX)
    
    # Noise Floor Estimate (Standard Error of Mean)
    # Correlation is average of complex exponentials magnitude 1.
    # sigma ~ 1 / sqrt(N)
    noise_floor = 1.0 / np.sqrt(N_NETS)
    print(f"Monte Carlo Noise Floor (1/sqrt(N)): {noise_floor:.5f}")
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    for alpha in alphas:
        # Simulation Data
        sim_data = final_corrs[alpha]
        
        # Exact Theory: <VV> = exp(-0.5 * alpha^2 * D(r))
        theory_data = np.exp(-0.5 * alpha**2 * D_exact)
        
        # --- Smart Error Calculation ---
        # Only compare where signal is distinguishable from noise (> 2 sigma)
        valid_mask = theory_data > (3 * noise_floor)
        
        if np.sum(valid_mask) > 0:
            rel_error = np.mean(np.abs((sim_data[valid_mask] - theory_data[valid_mask]) / theory_data[valid_mask])) * 100
            print(f"Alpha {alpha:.3f}: Mean Error (Trust Region) = {rel_error:.4f}% ({np.sum(valid_mask)} points)")
        else:
            print(f"Alpha {alpha:.3f}: Signal below noise floor. Increase N_NETS.")
        
        plt.loglog(r_vals, sim_data, 'o', alpha=0.6, label=f'Sim $\\alpha={alpha}$')
        plt.loglog(r_vals, theory_data, '--', color='black', alpha=0.7)
        
    plt.plot([], [], 'k--', label='Exact Finite-Cutoff Integral')
    plt.axhline(noise_floor, color='red', linestyle=':', label='Noise Floor')
    
    plt.xlabel('|z|')
    plt.ylabel(r'Correlation $\langle V_\alpha(0) V_{-\alpha}(z) \rangle$')
    plt.title('Vertex Operator: Simulation vs Exact Bessel Integral')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('vertex_precision.png')
    print("Plot saved to vertex_precision.png")

if __name__ == "__main__":
    run_precision_test()