import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from optimization_config import M_BOUNDS, P_BOUNDS, T_BOUNDS, ALPHA_BOUNDS, is_feasible
from demo_airfoil import run_xfoil_single_alpha

# Settings
N_INIT = 8
N_ITER = 50
RE = 1_000_000  # Realistic Reynolds number for windsurf foil

class ParameterScaler:
    """Handles scaling and unscaling of parameters to/from unit cube."""
    def __init__(self, bounds):
        self.bounds = torch.tensor(bounds, dtype=torch.double)
        self.lower = self.bounds[0]
        self.upper = self.bounds[1]
        self.range = self.upper - self.lower

    def scale(self, x):
        """Scale parameters to unit cube [0,1]."""
        return (x - self.lower) / self.range

    def unscale(self, x):
        """Unscale parameters from unit cube [0,1] to original bounds."""
        return x * self.range + self.lower

# Initialize parameter scaler with all bounds
PARAM_BOUNDS = [
    [M_BOUNDS[0], P_BOUNDS[0], T_BOUNDS[0], ALPHA_BOUNDS[0]],
    [M_BOUNDS[1], P_BOUNDS[1], T_BOUNDS[1], ALPHA_BOUNDS[1]]
]
scaler = ParameterScaler(PARAM_BOUNDS)

def naca_code(m, p, t):
    m_digit = int(round(m * 100))
    p_digit = int(round(p * 10))
    t_digits = int(round(t * 100))
    return f"NACA{m_digit}{p_digit}{t_digits:02d}"

def evaluate_design(m, p, t, alpha):
    code = naca_code(m, p, t)
    cl, cd = run_xfoil_single_alpha(code, RE, alpha)
    clcd = cl / cd if cd > 0 else np.nan
    return clcd

def sample_initial_data(n_samples):
    # Sample in original space
    m = np.random.uniform(*M_BOUNDS, n_samples)
    p = np.random.uniform(*P_BOUNDS, n_samples)
    t = np.random.uniform(*T_BOUNDS, n_samples)
    alpha = np.random.uniform(*ALPHA_BOUNDS, n_samples)
    X = np.stack([m, p, t, alpha], axis=1)
    # Scale to unit cube
    X_scaled = scaler.scale(torch.tensor(X, dtype=torch.double)).numpy()
    return X_scaled

def bayesian_optimization(n_init=N_INIT, n_iter=N_ITER):
    # Initial data
    X_init = []
    Y_init = []
    while len(X_init) < n_init:
        X_scaled = sample_initial_data(1)[0]
        # Unscale for evaluation
        X_unscaled = scaler.unscale(torch.tensor(X_scaled, dtype=torch.double)).numpy()
        m, p, t, alpha = X_unscaled
        y = evaluate_design(m, p, t, alpha)
        if not np.isnan(y):
            X_init.append(X_scaled)
            Y_init.append([y])
            print(f"Init {len(X_init)}: m={m:.4f}, p={p:.4f}, t={t:.4f}, alpha={alpha:.4f}, Cl/Cd={y:.4f}")
    
    # Convert lists to numpy arrays before creating tensors
    X_init_array = np.array(X_init)
    Y_init_array = np.array(Y_init)
    X = torch.tensor(X_init_array, dtype=torch.double)
    Y = torch.tensor(Y_init_array, dtype=torch.double)

    for iteration in range(n_iter):
        # Fit GP
        model = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Acquisition function
        EI = LogExpectedImprovement(model, best_f=Y.max())

        # Optimize acquisition function in unit cube
        bounds = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], 
                            dtype=torch.double, device=X.device)
        candidate, _ = optimize_acqf(
            EI, bounds=bounds, q=1, num_restarts=5, raw_samples=20
        )
        
        # Unscale candidate for evaluation
        candidate_unscaled = scaler.unscale(candidate)
        m, p, t, alpha = candidate_unscaled.detach().cpu().numpy().flatten()
        y = evaluate_design(m, p, t, alpha)

        if not np.isnan(y):
            X = torch.cat([X, candidate], dim=0)
            Y = torch.cat([Y, torch.tensor([[y]], dtype=torch.double, device=Y.device)], dim=0)
            print(f"Iter {iteration+1}: m={m:.4f}, p={p:.4f}, t={t:.4f}, alpha={alpha:.4f}, Cl/Cd={y:.4f}")
        else:
            print(f"Iter {iteration+1}: m={m:.4f}, p={p:.4f}, t={t:.4f}, alpha={alpha:.4f}, Cl/Cd=NaN (skipped)")

    # Return best found
    best_idx = torch.argmax(Y)
    best_params_scaled = X[best_idx].numpy()
    best_params = scaler.unscale(torch.tensor(best_params_scaled, dtype=torch.double)).numpy()
    best_value = Y[best_idx].item()
    best_naca = naca_code(*best_params[:3])  # Only use m, p, t for NACA code
    print(f"\nBest found: {best_naca} (m={best_params[0]:.4f}, p={best_params[1]:.4f}, t={best_params[2]:.4f}, alpha={best_params[3]:.4f}), Cl/Cd={best_value:.4f}")

    # Output XFOIL script for best result
    xfoil_script = f"""
{best_naca}
OPER
VISC {RE}
ITER
200
PACC
polar_output_best.txt

ALFA {best_params[3]}
"""
    with open("xfoil-script.txt", "w") as f:
        f.write(xfoil_script.strip() + "\n")
    print("\nXFOIL script for best result written to xfoil-script.txt")

    # Store all points in original space for plotting
    all_points = {
        'init': {
            'X': scaler.unscale(torch.tensor(X_init_array, dtype=torch.double)).numpy(),
            'Y': Y_init_array
        },
        'iter': {
            'X': scaler.unscale(X).numpy(),
            'Y': Y.numpy()
        }
    }

    return all_points, best_params, best_value

if __name__ == "__main__":
    all_points, best_params, best_value = bayesian_optimization()
    # Save results to a file
    np.savez('optimization_results.npz',
             init_X=all_points['init']['X'],
             init_Y=all_points['init']['Y'],
             iter_X=all_points['iter']['X'],
             iter_Y=all_points['iter']['Y'],
             best_params=best_params,
             best_value=best_value)
    print("\nOptimization results saved to optimization_results.npz") 