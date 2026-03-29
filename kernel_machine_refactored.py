import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
from typing import Optional, Callable, Tuple
from datetime import datetime
from main import sample_episode
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# --- Jacobi theta3 helper: theta(3, z, q) ---
try:
    from scipy.special import theta as _theta

    def theta3(z, q):
        return _theta(3, z, q)

except Exception:
    import mpmath as mp

    def theta3(z, q):
        return float(mp.jtheta(3, z, q))


# ============================================================
# Kernel definitions (core math unchanged)
# ============================================================

def heat_kernel_S1(t, x, y, wrap=True):
    """Heat kernel on S^1 using theta3."""
    d = x - y
    if wrap:
        d = (d + np.pi) % (2 * np.pi) - np.pi
    q = np.exp(-t)
    return (1 / (2 * np.pi)) * theta3(d / 2, q)


def heat_kernel_torus_2d(t, x_vec, y_vec, wrap=True):
    """Product kernel on S^1 x S^1 (2D torus)."""
    return (heat_kernel_S1(t, x_vec[0], y_vec[0], wrap=wrap) *
            heat_kernel_S1(t, x_vec[1], y_vec[1], wrap=wrap))


# --- Gram matrix builders ---

def gram_matrix_heat_torus_2d(X, t, wrap=True):
    """Gram matrix K[i,j] = heat_kernel_torus_2d(t, X[i], X[j])."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            v = heat_kernel_torus_2d(t, X[i], X[j], wrap=wrap)
            K[i, j] = v
            K[j, i] = v
    return K


def _embed_torus(X):
    """Map angles to R^4 embedding: (cos θ1, sin θ1, cos θ2, sin θ2)."""
    return np.column_stack([np.cos(X[..., 0]), np.sin(X[..., 0]),
                            np.cos(X[..., 1]), np.sin(X[..., 1])])


def _sq_dist_matrix(A, B=None):
    """Squared Euclidean distance matrix between rows of A and B."""
    if B is None:
        return cdist(A, A, metric='sqeuclidean')
    return cdist(A, B, metric='sqeuclidean')


def gram_softmax(X, t=0.5, wrap=True):
    real_X = _embed_torus(X)
    dist_mat = _sq_dist_matrix(real_X)
    exp_x = np.exp(-dist_mat / (4 * t))
    return exp_x / np.sum(exp_x, axis=1)[:, np.newaxis]


def gram_exp(X, t=0.5, wrap=True):
    real_X = _embed_torus(X)
    dist_mat = _sq_dist_matrix(real_X)
    return np.exp(-dist_mat / (4 * t))


def gram_euclidean_heat(X, t=0.5, wrap=True):
    real_X = _embed_torus(X)
    dist_mat = _sq_dist_matrix(real_X)
    return np.exp(-dist_mat / (4 * t)) / (4 * np.pi * t) ** (real_X.shape[1] / 2)


# --- Kernel vector builders (for prediction: k(x_test, x_train)) ---

def kernel_vector_heat(x_test, X_train, t, wrap=True):
    """Returns array of shape (n,) with k(x_train[i], x_test) for each i."""
    n = X_train.shape[0]
    kv = np.empty(n, dtype=float)
    for i in range(n):
        kv[i] = heat_kernel_torus_2d(t, X_train[i], x_test, wrap=wrap)
    return kv


def kernel_vector_softmax(x_test, X_train, t, wrap=True):
    """
    Softmax kernel vector: n * exp(-d^2/4t) / Z
    where Z = sum over training points + the test point itself.
    Note: for prediction consistency, we embed [X_train; x_test] and
    return the last column of the softmax matrix restricted to training rows.
    """
    X_all = np.vstack([X_train, x_test[np.newaxis]])
    real_all = _embed_torus(X_all)
    n = X_train.shape[0]
    # Distances from each training point to all points (including test)
    dist_to_test = _sq_dist_matrix(real_all[:n], real_all[-1:]).ravel()  # (n,)
    dist_all = _sq_dist_matrix(real_all)  # (n+1, n+1)
    # Row normalization over all n+1 points for each training row
    exp_all = np.exp(-dist_all[:n] / (4 * t))  # (n, n+1)
    row_sums = exp_all.sum(axis=1)  # (n,)
    exp_to_test = np.exp(-dist_to_test / (4 * t))  # (n,)
    return exp_to_test / row_sums


def kernel_vector_exp(x_test, X_train, t, wrap=True):
    real_train = _embed_torus(X_train)
    real_test = _embed_torus(x_test[np.newaxis])
    dist = _sq_dist_matrix(real_train, real_test).ravel()
    return np.exp(-dist / (4 * t))


def kernel_vector_euclidean_heat(x_test, X_train, t, wrap=True):
    real_train = _embed_torus(X_train)
    real_test = _embed_torus(x_test[np.newaxis])
    d = real_train.shape[1]
    dist = _sq_dist_matrix(real_train, real_test).ravel()
    return np.exp(-dist / (4 * t)) / (t) ** (d / 2)


# ============================================================
# Solver (unchanged)
# ============================================================

def solve_alphas(K, y, lam):
    """Solve via (K^T K + lam K^T) alpha = K^T y."""
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = K.shape[0]
    if lam <= 0:
        raise ValueError("lam should be > 0 for stable regularization.")
    return np.linalg.solve(K.T @ K + lam * K.T, K.T @ y)


# ============================================================
# Prediction (vectorized, no wasteful kernel rebuilds)
# ============================================================

def predict(X_train, alpha, X_test, t, kernel_vector_fn, wrap=True):
    """f(x) = sum_i alpha_i * k(x_train[i], x)"""
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    yhat = np.empty(X_test.shape[0], dtype=float)
    for m, x in enumerate(X_test):
        kv = kernel_vector_fn(x, X_train, t, wrap=wrap)
        yhat[m] = alpha @ kv
    return yhat


# ============================================================
# Model registry
# ============================================================

@dataclass
class KernelModel:
    name: str
    gram_fn: Callable       # (X, t, wrap) -> K
    kv_fn: Callable          # (x_test, X_train, t, wrap) -> vector

    def fit(self, X, y, t, lam, wrap=True):
        K = self.gram_fn(X, t=t, wrap=wrap)
        alpha = solve_alphas(K, y, lam=lam)
        return K, alpha

    def predict(self, X_train, alpha, X_test, t, wrap=True):
        return predict(X_train, alpha, X_test, t, self.kv_fn, wrap=wrap)


MODELS = {
    'softmax': KernelModel('softmax', gram_softmax, kernel_vector_softmax),
    'heat': KernelModel('heat', gram_matrix_heat_torus_2d, kernel_vector_heat),
    'euclidean_heat': KernelModel('euclidean_heat', gram_euclidean_heat, kernel_vector_euclidean_heat),
}


# ============================================================
# Tracking & I/O (unchanged logic)
# ============================================================

class BestConfigTracker:
    def __init__(self):
        self.best = {}

    def update(self, model_name, score, t, lam, n):
        if (model_name not in self.best
                or score < self.best[model_name]["score"]):
            self.best[model_name] = {
                "t": t, "lam": lam, "score": score, "context_length": n
            }

    def get_best(self, model_name):
        return self.best.get(model_name, {})


# ============================================================
# Main experiment loop
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
p = torch.tensor([np.pi, np.pi], device=device)
task = 'distance' #'periodic'  # or 'distance'
run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(__file__).parent / "kernels"
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / f"errors_by_model_{run_time}_{task}.csv"
episode_csv_path = out_dir / f"episode_params_{run_time}_{task}.csv"
plot_path = out_dir / f"kernel_machine_heat_torus_kernels_{run_time}_{task}.png"

print(f"Writing results to {csv_path}")

# --- Hyperparameters ---
batch_size = 1
num_episodes = 20
context_lengths = 2 * np.arange(1, 40, 2)

lams = np.logspace(-10, 3, 8).tolist()

# --- Accumulators ---
rows = []
episode_rows = []
means_for_models = {name: [] for name in MODELS}
stds_for_models = {name: [] for name in MODELS}

for K_context in context_lengths:
    tracker = BestConfigTracker()
    errors_by_model = {name: np.zeros(num_episodes) for name in MODELS}
    episode_params = {name: [] for name in MODELS}

    for rep in range(num_episodes):
        min_error = {name: 100.0 for name in MODELS}
        best_t = {name: None for name in MODELS}
        best_lam = {name: None for name in MODELS}

        tokens, y_q = sample_episode(batch_size, K_context, p, device, task=task)

        # Extract numpy arrays once per episode (not inside the triple loop)
        tok_np = tokens.cpu().numpy()[0]
        X = tok_np[:K_context, :2]
        y = tok_np[:K_context, 2]
        X_test = tok_np[-1:, :2]
        y_q_np = y_q.cpu().numpy().item()
        ts = [c * 4 * np.pi**2 / K_context for c in [0.1, 0.25, 0.5, 1.0, 2.0]]
        for t_tilde in ts:
            for lam_tilde in lams:
                for name, model in MODELS.items():
                    try:
                        K, alpha = model.fit(X, y, t=t_tilde, lam=lam_tilde)
                        y_pred = model.predict(X, alpha, X_test, t=t_tilde)
                        error = np.abs(y_pred[0] - y_q_np)

                        if error < min_error[name]:
                            min_error[name] = error
                            best_t[name] = t_tilde
                            best_lam[name] = lam_tilde
                            tracker.update(name, error, t_tilde, lam_tilde, K_context)

                    except np.linalg.LinAlgError:
                        pass

        for name in MODELS:
            errors_by_model[name][rep] = min_error[name]
            episode_params[name].append({
                'episode': rep, 'K_context': int(K_context),
                'error': min_error[name],
                't': best_t[name], 'lam': best_lam[name]
            })

    for name in MODELS:
        mean_err = np.mean(errors_by_model[name])
        std_err = np.std(errors_by_model[name])
        means_for_models[name].append(mean_err)
        stds_for_models[name].append(std_err)

        best_cfg = tracker.get_best(name)
        rows.append({
            "model": name, "K_context": int(K_context),
            "mean_error": mean_err, "std_error": std_err,
            "best_t": best_cfg.get("t"), "best_lam": best_cfg.get("lam"),
            "best_score": best_cfg.get("score"),
        })
        for ep in episode_params[name]:
            episode_rows.append({"model": name, **ep})

        print(f"{name:>16s}  K={K_context:3.0f}  "
              f"err={mean_err:.4f}±{std_err:.4f}  "
              f"t*={best_cfg.get('t')}  λ*={best_cfg.get('lam')}")

# --- Write CSVs ---
if rows:
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

if episode_rows:
    with episode_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        w.writeheader()
        w.writerows(episode_rows)
    print(f"Per-episode parameters saved to {episode_csv_path}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
for name in MODELS:
    means = np.array(means_for_models[name])
    stds = np.array(stds_for_models[name])
    ax.errorbar(context_lengths, means, yerr=stds, fmt='-o', label=name.capitalize())

ax.set_xlabel('K_context')
ax.set_ylabel('Mean Absolute Error ± StdDev')
ax.set_yscale('log')
ax.set_title('Different Kernel Machines on Heat Kernel over 2D Torus')
ax.set_xticks(context_lengths)
ax.tick_params(axis='x', rotation=45)
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(plot_path)
print(f"Plot saved to {plot_path}")