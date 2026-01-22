import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime
from main import sample_episode
import numpy as np
import matplotlib.pyplot as plt

# from scipy.special import theta
import numpy as np

# --- Jacobi theta3 helper: theta(3, z, q) ---
try:
    # If your SciPy has it, this will work in your environment
    from scipy.special import theta as _theta

    def theta3(z, q):
        # SciPy version: vectorized
        return _theta(3, z, q)

except Exception:
    # Fallback: mpmath (scalar; we convert to float)
    import mpmath as mp

    def theta3(z, q):
        return float(mp.jtheta(3, z, q))


def heat_kernel_S1(t, x, y, wrap=True):
    """
    Heat kernel on S^1 using theta3.
    x,y are angles in radians (typically in [0, 2π)).
    wrap=True wraps differences into (-π, π] for numerical stability.
    """
    d = x - y
    if wrap:
        d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap to (-π, π]
    q = np.exp(-t)
    return (1 / (2 * np.pi)) * theta3(d / 2, q)


def heat_kernel_torus_2d(t, x_vec, y_vec, wrap=True):
    """
    Product kernel on S^1 x S^1 (2D torus).
    x_vec, y_vec are length-2 arrays: [angle1, angle2].
    """
    return heat_kernel_S1(t, x_vec[0], y_vec[0], wrap=wrap) * \
           heat_kernel_S1(t, x_vec[1], y_vec[1], wrap=wrap)

def gram_matrix_heat_torus_2d(X, t, wrap=True):
    """
    Build Gram matrix K where K[i,j] = heat_kernel_torus_2d(t, X[i], X[j]).
    X: shape (n, 2)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (n, 2) for 2D inputs.")

    n = X.shape[0]
    K = np.empty((n, n), dtype=float)

    # Fill symmetrically
    for i in range(n):
        for j in range(i, n):
            v = heat_kernel_torus_2d(t, X[i], X[j], wrap=wrap)
            K[i, j] = v
            K[j, i] = v
    return K


def solve_alphas(K, y, lam):
    """
    Solve (K + lam I) alpha = y without forming an explicit inverse.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = K.shape[0]
    if K.shape != (n, n):
        raise ValueError("K must be square.")
    if y.shape[0] != n:
        raise ValueError("y must have length n.")
    if lam <= 0:
        raise ValueError("lam should be > 0 for stable regularization.")

    # A = K + lam * np.eye(n)
    alpha = np.linalg.solve(K @ K + lam * K, K @ y)

    # # Cholesky is best when A is SPD (lam>0 typically ensures this)
    # try:
    #     L = np.linalg.cholesky(A)
    #     alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    # except np.linalg.LinAlgError:
    #     # Fallback if numerical issues arise
    #     alpha = np.linalg.solve(A, y)

    return alpha


def fit_kernel_machine_heat_torus(X, y, t, lam, wrap=True):
    """
    Convenience: compute Gram matrix + alpha.
    """
    K = gram_matrix_heat_torus_2d(X, t=t, wrap=wrap)
    alpha = solve_alphas(K, y, lam=lam)
    return K, alpha


def predict_heat_torus(X_train, alpha, X_test, t, wrap=True):
    """
    f(x) = sum_i alpha_i k(x, x_i)
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test, dtype=float)
    alpha   = np.asarray(alpha, dtype=float).reshape(-1)

    n = X_train.shape[0]
    if alpha.shape[0] != n:
        raise ValueError("alpha length must match number of training points.")

    yhat = np.empty(X_test.shape[0], dtype=float)
    for m, x in enumerate(X_test):
        s = 0.0
        for i in range(n):
            s += alpha[i] * heat_kernel_torus_2d(t, x, X_train[i], wrap=wrap)
        yhat[m] = s
    return yhat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
p = torch.tensor([np.pi, np.pi], device=device)

def periodic_func(x,p):
    return np.cos(x[..., 0]) *(np.sin(x[..., 1])**2)


from scipy.spatial.distance import cdist
def SoftMaxKernel(X, t=1/2):
    real_X = np.column_stack([np.cos(X[...,0]), np.sin(X[...,0]), np.cos(X[...,1]), np.sin(X[...,1])])
    dist_mat =  cdist(real_X,real_X)**2
    exp_x = np.exp(-dist_mat/(4*t))
    return exp_x/np.sum(exp_x, axis=1)[:, np.newaxis]

def fit_kernel_machine_softmax(X, y, t, lam, wrap=True):
    """
    Convenience: compute Gram matrix + alpha.
    """
    K = SoftMaxKernel(X, t=t)
    alpha = solve_alphas(K, y, lam=lam)
    return K, alpha

def exp_dist_kernel(X, t=1/2):
    real_X = np.column_stack([np.cos(X[...,0]), np.sin(X[...,0]), np.cos(X[...,1]), np.sin(X[...,1])])
    dist_mat =  cdist(real_X,real_X)**2
    exp_x = np.exp(-dist_mat/(4*t))
    return exp_x

def fit_kernel_machine_exp(X, y, t, lam, wrap=True):
    """
    Convenience: compute Gram matrix + alpha.
    """
    K = exp_dist_kernel(X, t=t)
    alpha = solve_alphas(K, y, lam=lam)
    return K, alpha

def softmax_corr(X, t=1/2):
    real_X = np.column_stack([np.cos(X[...,0]), np.sin(X[...,0]), np.cos(X[...,1]), np.sin(X[...,1])])

    exp_x = np.exp(real_X @real_X.T/(t))
    return exp_x/np.sum(exp_x, axis=1)[:, np.newaxis]

def fit_kernel_machine_corr(X, y, t, lam, wrap=True):
    """
    Convenience: compute Gram matrix + alpha.
    """
    K = softmax_corr(X, t=t)
    alpha = solve_alphas(K, y, lam=lam)
    return K, alpha

import json

class BestConfigTracker:
    def __init__(self, save_path="best_configs.json"):
        self.best = {}
        self.save_path = save_path

    def update(self, model_name, score, t, lam, n):
        if (
            model_name not in self.best
            or score < self.best[model_name]["score"]  # Changed to < since lower error is better
        ):
            self.best[model_name] = {
                "t": t,
                "lam": lam,
                "score" : score,
                "context_length": n
            }
    
    def get_best(self, model_name):
        """Get best parameters for a model"""
        return self.best.get(model_name, {})

#### Save to CSV
from datetime import datetime
run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_path = (
    Path(__file__).parent
    / "kernels"
    / f"errors_by_model_{run_time}.csv"
)

# csv_path = Path("errors_by_model.csv")
write_header = not csv_path.exists()
print(f"Writing results to {csv_path} (write_header={write_header})")
rows = []

####
batch_size = 1
num_episodes = 20
context_lengths = 2 *np.arange(1,50,2)  # [2,4,6,...,98]
ts = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
lams = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]  # Fixed: 1e5 -> 1e-5
trackers = []
models = {'softmax': fit_kernel_machine_softmax, 'heat': fit_kernel_machine_heat_torus, 'exp': fit_kernel_machine_exp}
min_for_model = {model: 0 for model in models.keys()}
means_for_models = {model: np.array([]) for model in models.keys()}
stds_for_models = {model: np.array([]) for model in models.keys()}

# Store best parameters for each episode
best_params_per_episode = {model: [] for model in models.keys()}

for K_context in context_lengths:    
    tracker = BestConfigTracker()
    this_vals_for_model = {model: np.zeros(num_episodes) for model in models.keys()}
    # Track best params for each model at this K_context
    episode_params = {model: [] for model in models.keys()}
    
    for rep in range(num_episodes):  # multiple samples to average over
        min_for_model = {model: 100 for model in models.keys()}
        best_t_for_model = {model: None for model in models.keys()}
        best_lam_for_model = {model: None for model in models.keys()}

        tokens, y_q = sample_episode(batch_size, K_context, p, device) #, func=periodic_func)
        # Test various (t_tilde, lam_tilde) combinations
        for t_tilde in ts:
            for lam_tilde in lams:
                for model in models.keys():
                    func = models[model]
                    min_error = min_for_model[model]
                    X = tokens.cpu().numpy()[0,0:K_context,0:2]  # (K,2)
                    y = tokens.cpu().numpy()[0,0:K_context,2]  # (K,2)

                    X_test = tokens.cpu().numpy()[0,-1:,0:2]
                
                    try:
                        K, alpha = func(X, y, t=t_tilde, lam=K_context*lam_tilde)
                        y_pred = predict_heat_torus(X, alpha, X_test, t=t_tilde)
                        error = np.abs(y_pred - y_q.cpu().numpy())[0]
                        
                        if error < min_error:
                            min_for_model[model] = error
                            best_t_for_model[model] = t_tilde
                            best_lam_for_model[model] = lam_tilde
                            tracker.update(model, error, t_tilde, lam_tilde, K_context)

                    except np.linalg.LinAlgError:
                        pass
                        
        # Store the minimum error and best parameters for this episode
        for model in models.keys():
            this_vals_for_model[model][rep] = min_for_model[model]
            episode_params[model].append({
                'episode': rep,
                'K_context': K_context,
                'error': min_for_model[model],
                't': best_t_for_model[model],
                'lam': best_lam_for_model[model]
            })
    
    # Store episode params for this K_context
    for model in models.keys():
        best_params_per_episode[model].extend(episode_params[model])
    
    # Calculate statistics and get best parameters
    for model in models.keys():
        mean_error = np.mean(this_vals_for_model[model])
        std_error = np.std(this_vals_for_model[model])
        means_for_models[model] = np.append(means_for_models[model], mean_error)
        stds_for_models[model] = np.append(stds_for_models[model], std_error)
        
        # Get best parameters for this model at this K_context
        best_config = tracker.get_best(model)
        
        rows.append({
            "model": model,
            "K_context": K_context,
            "mean_error": mean_error,
            "std_error": std_error,
            "best_t": best_config.get("t", None),
            "best_lam": best_config.get("lam", None),
            "best_score": best_config.get("score", None)
        })
        print(f"{model=} K_context:{K_context}   Mean error over {num_episodes} episodes: {mean_error} ± {std_error}   Best: t={best_config.get('t')}, lam={best_config.get('lam')}")
    
    trackers.append(tracker)

# Write main results to CSV
if rows:
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# Write detailed per-episode parameters to a separate CSV
episode_csv_path = (
    Path(__file__).parent
    / "kernels"
    / f"episode_params_{run_time}.csv"
)

episode_rows = []
for model in models.keys():
    for params in best_params_per_episode[model]:
        episode_rows.append({
            "model": model,
            "K_context": params['K_context'],
            "episode": params['episode'],
            "error": params['error'],
            "best_t": params['t'],
            "best_lam": params['lam']
        })

if episode_rows:
    with episode_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)
    print(f"Per-episode parameters saved to {episode_csv_path}")

for model in models.keys():
    means = means_for_models[model]
    stds = stds_for_models[model]
    print(f'{model=} {list(zip(context_lengths,means))=}')
    plt.errorbar(context_lengths, means, yerr=stds, fmt='-o', label=model.capitalize())

plt.xlabel('K_context')
plt.ylabel('Mean Absolute Error ± StdDev')
plt.yscale('log')
plt.title('Different Kernel Machine on Heat Kernel over 2D Torus')
plt.xticks(context_lengths)
plt.grid(True)
plt.legend()
plt.savefig("kernel_machine_heat_torus_kernels.png")