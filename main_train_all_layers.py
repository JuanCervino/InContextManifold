import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np

TWOPI = 2.0 * math.pi

# -------------------------
# Config
# -------------------------
@dataclass
class RunConfig:
    steps: int = 2000
    batch_size: int = 64
    K: int = 32
    d_model: int = 5  # Fixed for bilinear tokenization
    d_ff: int = 256
    n_layers: int = 2  # NEW: Number of attention layers
    lr: float = 0.001  # Now with training
    attn_nonlinearity: str = "softmax"
    T: float = 0.01

    @staticmethod
    def from_json(path: str | Path) -> "RunConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return RunConfig(**data)

    def to_json(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
class SimpleCSVLogger:
    def __init__(self, log_dir: str | Path, config: RunConfig, run_name: str = "run", extra_params: Optional[dict] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / f"{run_name}_metrics.csv"
        self.json_path = self.log_dir / f"{run_name}_params.json"

        params = asdict(config)
        if extra_params:
            params.update(extra_params)
        
        # Convert ALL params to JSON-serializable types (not just extra_params)
        params = self._make_serializable(params)

        with open(self.json_path, "w") as f:
            json.dump(params, f, indent=2, sort_keys=True)

        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=["step", "loss", "mae"])
        self._writer.writeheader()
        self._csv_file.flush()

    def _make_serializable(self, obj):
        """Convert numpy/torch types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle PyTorch scalars
            return obj.item()
        else:
            return obj

    def log(self, step: int, loss: float, mae: float):
        self._writer.writerow({"step": int(step), "loss": float(loss), "mae": float(mae)})
        self._csv_file.flush()

    def close(self):
        if getattr(self, "_csv_file", None) is not None:
            self._csv_file.close()
            self._csv_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
# class SimpleCSVLogger:
#     def __init__(self, log_dir: str | Path, config: RunConfig, run_name: str = "run", extra_params: Optional[dict] = None):
#         self.log_dir = Path(log_dir)
#         self.log_dir.mkdir(parents=True, exist_ok=True)

#         self.csv_path = self.log_dir / f"{run_name}_metrics.csv"
#         self.json_path = self.log_dir / f"{run_name}_params.json"

#         params = asdict(config)
#         if extra_params:
#             params.update(extra_params)

#         with open(self.json_path, "w") as f:
#             json.dump(params, f, indent=2, sort_keys=True)

#         self._csv_file = open(self.csv_path, "w", newline="")
#         self._writer = csv.DictWriter(self._csv_file, fieldnames=["step", "loss", "mae"])
#         self._writer.writeheader()
#         self._csv_file.flush()

#     def log(self, step: int, loss: float, mae: float):
#         self._writer.writerow({"step": int(step), "loss": float(loss), "mae": float(mae)})
#         self._csv_file.flush()

#     def close(self):
#         if getattr(self, "_csv_file", None) is not None:
#             self._csv_file.close()
#             self._csv_file = None

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc, tb):
#         self.close()


# ---------- Torus distance target ----------
def torus_delta(a, b, period=TWOPI):
    diff = (a - b).abs()
    return torch.minimum(diff, period - diff)

def torus_distance(xy, pxy, period=TWOPI, eps=1e-12):
    dx = torus_delta(xy[..., 0], pxy[..., 0], period)
    dy = torus_delta(xy[..., 1], pxy[..., 1], period)
    return torch.sqrt(dx*dx + dy*dy + eps)


# ---------- Episode sampler ----------
@torch.no_grad()
def sample_episode(batch_size, K, p, device):
    """
    Returns:
      tokens: (B, K+1, 4) where each token is [x1, x2, y, is_query]
      y_query: (B,) the ground-truth y for the query token
    """
    x_ctx = torch.rand(batch_size, K, 2, device=device) * TWOPI
    y_ctx = torus_distance(x_ctx, p).unsqueeze(-1)
    is_query_ctx = torch.zeros(batch_size, K, 1, device=device)
    ctx_tokens = torch.cat([x_ctx, y_ctx, is_query_ctx], dim=-1)

    x_q = torch.rand(batch_size, 1, 2, device=device) * TWOPI
    y_q = torus_distance(x_q, p).squeeze(-1).squeeze(-1)
    y_q_hidden = torch.zeros(batch_size, 1, 1, device=device)
    is_query_q = torch.ones(batch_size, 1, 1, device=device)
    q_token = torch.cat([x_q, y_q_hidden, is_query_q], dim=-1)

    tokens = torch.cat([ctx_tokens, q_token], dim=1)
    return tokens, y_q


class Attention(nn.Module):
    """
    Fixed attention mechanism with specific weight matrices from Lemma A.6.
    Implements: Attn_h(Z; V, B, C) = Z + V Z h(BZ, CZ)
    Only V is trainable.
    """
    def __init__(self, d_model=5, attn_nonlinearity="softmax"):
        super().__init__()
        self.d_model = d_model

        self.B = nn.Linear(d_model, d_model)
        self.C = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        # V is trainable, initialized as identity
        # self.V = nn.Parameter(torch.eye(d_model))
        self.attn_nonlinearity = attn_nonlinearity 
        
    
    def _apply_attn_nonlinearity(self, scores):
        name = self.attn_nonlinearity.lower()
        if name == "softmax":
            attn = F.softmax(scores, dim=-1)
        elif name == "exp":
            attn = torch.exp(scores)
            attn = attn / (attn.sum(dim=-2, keepdim=True) + 1e-8)
        elif name == "relu":
            attn = F.relu(scores)
            attn = attn / (attn.sum(dim=-2, keepdim=True) + 1e-8)
        elif name == "identity":
            attn = scores
        else:
            raise ValueError(f"Unknown attn_nonlinearity: {name}")

        return attn
    
    def forward(self, x, mask=None):
        """
        x: (B, L, d_model)
        Computes: Z + V Z h(BZ, CZ) where h is the attention nonlinearity
        """
        B, L, D = x.shape
        
        BZ = self.B(x)
        CZ = self.C(x)
        
        scores = torch.bmm(BZ, CZ.transpose(-2, -1))
        
        attn = self._apply_attn_nonlinearity(scores)
        VZ = self.V(x)
        out = torch.bmm(attn, VZ) 
        
        return x + out ## This seems like the residual stream
 
class BilinearTokenization(nn.Module):
    """
    Implements the bilinear tokenization scheme from Lemma A.6.
    bilin(Z) = [1, x1, x2, ||x1||^2, ||x2||^2, 0, ..., 0, y]^T
    """
    def __init__(self, d_model=5):
        super().__init__()
        self.d_model = d_model
        self.n_padding = max(0, d_model - 5)
        
    def forward(self, tokens):
        """
        tokens: (B, L, 4) = [x1, x2, y, is_query]
        returns: (B, L, d_model) with structure [1, cos(x1), sin(x1), cos(x2), sin(x2), ||x||^2, 0..., y]
        """
        B, L, _ = tokens.shape
        device = tokens.device
        
        x1 = tokens[..., 0:1]
        x2 = tokens[..., 1:2]
        y = tokens[..., 2:3]
        is_query = tokens[..., 3:4]
        
        x_norm_sq  = torch.cos(x1) ** 2 + torch.sin(x1) ** 2 + torch.cos(x2) ** 2 + torch.sin(x2) ** 2
        ones = torch.ones(B, L, 1, device=device)
        y_masked = y * (1 - is_query)
        
        components = [ones, torch.cos(x1), torch.sin(x1), torch.cos(x2), torch.sin(x2), x_norm_sq]
        
        # if self.n_padding > 0:
        #     zeros = torch.zeros(B, L, self.n_padding, device=device)
        #     components.append(zeros)
        
        components.append(y_masked)
        embedded = torch.cat(components, dim=-1)
        
        return embedded


class FixedAttention(nn.Module):
    """
    Fixed attention mechanism with specific weight matrices from Lemma A.6.
    Implements: Attn_h(Z; V, B, C) = Z + V Z h(BZ, CZ)
    Only V is trainable.
    """
    def __init__(self, d_model=5, attn_nonlinearity="softmax", T=0.01):
        super().__init__()
        self.d_model = d_model
        self.T = T

        self.register_buffer('B', self._construct_B())
        self.register_buffer('C', self._construct_C())
        # V is trainable, initialized as identity
        # self.V = nn.Parameter(torch.eye(d_model))
        V = torch.zeros(d_model, d_model)
        V[-1, -1] = torch.randn(1).item() * 0.01  # Small random initialization
        self.V = nn.Parameter(V)
        self.attn_nonlinearity = attn_nonlinearity 
    
    def _apply_attn_nonlinearity(self, scores):
        name = self.attn_nonlinearity.lower()
        if name == "softmax":
            attn = F.softmax(scores, dim=-1)
        elif name == "exp":
            attn = torch.exp(scores)
            attn = attn / (attn.sum(dim=-2, keepdim=True) + 1e-8)
        elif name == "relu":
            attn = F.relu(scores)
            attn = attn / (attn.sum(dim=-2, keepdim=True) + 1e-8)
        elif name == "identity":
            attn = scores
        else:
            raise ValueError(f"Unknown attn_nonlinearity: {name}")
        return attn
    
    def forward(self, x, mask=None):
        """
        x: (B, L, d_model)
        Computes: Z + V Z h(BZ, CZ) where h is the attention nonlinearity
        """
        B, L, D = x.shape
        
        BZ = x @ self.B.T
        CZ = x @ self.C.T
        
        scores = torch.bmm(BZ, CZ.transpose(-2, -1))
        

        attn = self._apply_attn_nonlinearity(scores)
        VZ = x @ self.V.T
        out = torch.bmm(attn, VZ) 
        
        return x + out ## This seems like the residual stream
class TorusRegressor(nn.Module):
    """
    Multi-layer architecture with bilinear tokenization and stacked attention layers.
    All matrices trainable
    """
    def __init__(self, d_token=4, d_model=5, n_layers=2, attn_nonlinearity="softmax", T=0.01):

        super().__init__()
        
        # Bilinear tokenization (no parameters)
        self.tokenizer = BilinearTokenization(d_model=d_model)
        
        # Stack of attention layers, each with trainable V
        self.layers = nn.ModuleList([
            Attention(d_model=d_model, attn_nonlinearity=attn_nonlinearity)
            for _ in range(n_layers)
        ])
        
        # Fixed output projection
        self.register_buffer('out_proj', self._construct_output_projection(d_model))
        
    def _construct_output_projection(self, d_model):
        proj = torch.zeros(1, d_model)
        proj[0, -1] = 1.0  # Extract from last dimension (y values)
        return proj
        
    def forward(self, tokens, attn_mask=None):
        """
        tokens: (B, L, 4) = [x1, x2, y_or_0, is_query]
        returns: yhat_all: (B, L)
        """
        # Tokenize once using bilinear scheme
        h = self.tokenizer(tokens)
        
        # Apply multiple attention layers
        for layer in self.layers:
            h = layer(h, mask=attn_mask)
        
        # Project to output
        yhat = (h @ self.out_proj.T).squeeze(-1)
        
        return yhat


class FixedTorusRegressor(nn.Module):
    """
    Multi-layer architecture with bilinear tokenization and stacked attention layers.
    Only V matrices are trainable.
    """
    def __init__(self, d_token=4, d_model=5, n_layers=2, attn_nonlinearity="softmax", T=0.01):

        super().__init__()
        
        # Bilinear tokenization (no parameters)
        self.tokenizer = BilinearTokenization(d_model=d_model)
        
        # Stack of attention layers, each with trainable V
        self.layers = nn.ModuleList([
            FixedAttention(d_model=d_model, attn_nonlinearity=attn_nonlinearity, T=T)
            for _ in range(n_layers)
        ])
        
        # Fixed output projection
        self.register_buffer('out_proj', self._construct_output_projection(d_model))
        
    def _construct_output_projection(self, d_model):
        proj = torch.zeros(1, d_model)
        proj[0, -1] = 1.0  # Extract from last dimension (y values)
        return proj
        
    def forward(self, tokens, attn_mask=None):
        """
        tokens: (B, L, 4) = [x1, x2, y_or_0, is_query]
        returns: yhat_all: (B, L)
        """
        # Tokenize once using bilinear scheme
        h = self.tokenizer(tokens)
        
        # Apply multiple attention layers
        for layer in self.layers:
            h = layer(h, mask=attn_mask)
        
        # Project to output
        yhat = (h @ self.out_proj.T).squeeze(-1)
        
        return yhat


def train_architecture(
    config: RunConfig,
    log_dir: str | Path = "logs/torus_trainable_v",
    run_name: str = "trainable_v",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train the architecture with trainable V matrices.
    """
    p = torch.tensor([1.0, 2.0], device=device)

    model = TorusRegressor(
        d_token=6, 
        d_model=config.d_model, 
        n_layers=config.n_layers,
        attn_nonlinearity=config.attn_nonlinearity,
        T=config.T,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total params: {total_params}, Trainable: {trainable_params}")
    print(f"Architecture: {config.n_layers}-layer with trainable V matrices")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    logger = SimpleCSVLogger(
        log_dir,
        config,
        run_name=run_name,
        extra_params={
            "total_params": total_params, 
            "trainable_params": trainable_params,
            "architecture": f"{config.n_layers}_layer_trainable_v"
        },
    )

    model.train()
    for step in range(1, config.steps + 1):
        tokens, y_q = sample_episode(config.batch_size, config.K, p, device)
        
        # Forward pass
        yhat_all = model(tokens)
        yhat_q = yhat_all[:, -1]
        
        # Compute loss
        loss = F.mse_loss(yhat_q, y_q)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            mae = (yhat_q - y_q).abs().mean().item()
        
        if step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.6f} | MAE {mae:.4f}")
        
        if step % 10 == 0:
            logger.log(step, loss.item(), mae)
    
    logger.close()
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Training Multi-Layer Architecture with Trainable V Matrices")
    print("Based on Lemma A.6 construction")
    print("=" * 60)
    ts = [0.001]
    # ts = [0.15, 0.2, 0.25]

    context_lengths = 2 *np.arange(10,20,2)  # [2,4,6,...,98]
    layers = [9,15]  # Different number of layers to try
    
    for _ in range(1):  # 3 runs per config
        for thisLayers in layers:
            for thisK in context_lengths:
                for T in ts:
                    config = RunConfig(
                        steps=20000,
                        batch_size=64,
                        K=thisK,
                        d_model=7,
                        d_ff=0,
                        n_layers=thisLayers,  
                        lr=1e-3,
                        attn_nonlinearity="softmax",
                        T=T,
                    )

                    run_name = f"K_{thisK}_L_{config.n_layers}_T_{T}" + datetime.now().strftime("%Y%m%d_%H%M%S")
                    model = train_architecture(config, log_dir="logs/torus_trainable_T_layers_", run_name=run_name)
                    
                    print(f"\nTraining complete for K={thisK} with {config.n_layers} layers!")
                    print("-" * 60)