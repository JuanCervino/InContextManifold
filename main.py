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




# -------------------------
# Config
# -------------------------
@dataclass
class RunConfig:
    steps: int = 2000
    batch_size: int = 64
    K: int = 32
    d_model: int = 128
    d_ff: int = 256
    lr: float = 1e-3
    attn_nonlinearity: str = "softmax"

    @staticmethod
    def from_json(path: str | Path) -> "RunConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return RunConfig(**data)

    def to_json(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)


class SimpleCSVLogger:
    """
    Writes:
      - metrics.csv with columns: step, loss, mae
      - params.json with run hyperparams (and optional extras)
    """
    def __init__(self, log_dir: str | Path, config: RunConfig, run_name: str = "run", extra_params: Optional[dict] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / f"{run_name}_metrics.csv"
        self.json_path = self.log_dir / f"{run_name}_params.json"

        params = asdict(config)
        if extra_params:
            params.update(extra_params)

        # Write params once
        with open(self.json_path, "w") as f:
            json.dump(params, f, indent=2, sort_keys=True)

        # Create CSV + header
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=["step", "loss", "mae"])
        self._writer.writeheader()
        self._csv_file.flush()

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

TWOPI = 2.0 * math.pi

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
              - context tokens have is_query=0 and correct y
              - query token has is_query=1 and y is set to 0 (hidden)
      y_query: (B,) the ground-truth y for the query token
    """
    # context
    x_ctx = torch.rand(batch_size, K, 2, device=device) * TWOPI
    y_ctx = torus_distance(x_ctx, p).unsqueeze(-1)  # (B,K,1)

    is_query_ctx = torch.zeros(batch_size, K, 1, device=device)

    ctx_tokens = torch.cat([x_ctx, y_ctx, is_query_ctx], dim=-1)  # (B,K,4)

    # query
    x_q = torch.rand(batch_size, 1, 2, device=device) * TWOPI
    y_q = torus_distance(x_q, p).squeeze(-1).squeeze(-1)          # (B,)
    y_q_hidden = torch.zeros(batch_size, 1, 1, device=device)
    is_query_q = torch.ones(batch_size, 1, 1, device=device)

    q_token = torch.cat([x_q, y_q_hidden, is_query_q], dim=-1)    # (B,1,4)

    tokens = torch.cat([ctx_tokens, q_token], dim=1)              # (B,K+1,4)
    return tokens, y_q

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model, attn_nonlinearity="softmax"):
        super().__init__()
        self.d_model = d_model

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        # string or callable
        self.attn_nonlinearity = attn_nonlinearity

    def _apply_attn_nonlinearity(self, scores):
        """
        scores: (B, L, L)
        returns attn weights (B, L, L)
        """
        nl = self.attn_nonlinearity
        if callable(nl):
            attn = nl(scores)
        else:
            name = str(nl).lower()
            if name == "softmax":
                attn = F.softmax(scores, dim=-1)
            elif name == "relu":
                attn = F.relu(scores)  # NOTE: not normalized
            elif name == "relu_norm":
                attn = F.relu(scores)
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            elif name == "sigmoid":
                attn = torch.sigmoid(scores)
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            elif name == "tanh":
                attn = torch.tanh(scores)
                # tanh can be negative; make it nonnegative then normalize
                attn = F.relu(attn)
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            elif name == "identity":
                attn = scores  # not normalized
            else:
                raise ValueError(f"Unknown attn_nonlinearity: {nl}")

        return attn

    def forward(self, x, mask=None):
        """
        x: (B, L, d_model)
        mask (optional):
          - (B, L) key padding mask with 1=keep, 0=mask, OR
          - (B, L, L) attention mask with 1=keep, 0=mask
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x (B,L,D), got {tuple(x.shape)}")

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_model)  # (B,L,L)

        # masking before nonlinearity is standard
        if mask is not None:
            if mask.dim() == 2:  # (B,L)
                scores = scores.masked_fill(mask[:, None, :] == 0, float("-inf"))
            elif mask.dim() == 3:  # (B,L,L)
                scores = scores.masked_fill(mask == 0, float("-inf"))
            else:
                raise ValueError(f"mask must be dim 2 or 3, got {mask.dim()}")

        # IMPORTANT: if you use non-softmax nonlinearities, -inf can break things.
        # Replace -inf with a large negative number for non-softmax modes.
        if not (callable(self.attn_nonlinearity) or str(self.attn_nonlinearity).lower() == "softmax"):
            scores = torch.nan_to_num(scores, neginf=-1e4, posinf=1e4)

        attn = self._apply_attn_nonlinearity(scores)     # (B,L,L)
        out = attn @ V                                   # (B,L,D)
        return out
# ---------- In-context regressor ----------
class InContextTorusRegressor(nn.Module):
    def __init__(self, d_token=4, d_model=128, d_ff=256, attn_nonlinearity="softmax"):
        super().__init__()
        self.in_proj = nn.Linear(d_token, d_model)
        self.attn = SimpleSelfAttention(d_model, attn_nonlinearity=attn_nonlinearity)
        self.ln1 = nn.LayerNorm(d_model)

        # FFN (you can swap in GLU here if you want)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_ff),
            nn.GLU(dim=-1),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

        self.out = nn.Linear(d_model, 1)

    def forward(self, tokens, attn_mask=None):
        """
        tokens: (B, L, 4) = [x1, x2, y_or_0, is_query]
        returns: yhat_all: (B, L)
        """
        h = self.in_proj(tokens)                    # (B,L,d)
        h = self.ln1(h + self.attn(h, attn_mask))   # residual
        h = self.ln2(h + self.ff(h))                # residual
        yhat = self.out(h).squeeze(-1)              # (B,L)
        return yhat


def train_in_context(
    config: RunConfig,
    log_dir: str | Path = "logs/torus_ic",
    run_name: str = "exp1",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    p = torch.tensor([1.0, 2.0], device=device)

    model = InContextTorusRegressor(d_token=4, d_model=config.d_model, d_ff=config.d_ff, attn_nonlinearity=config.attn_nonlinearity).to(device)
    print("Number of params", total_params := sum(p.numel() for p in model.parameters()),
          "trainable", trainable_params := sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger = SimpleCSVLogger(
        log_dir,
        config,
        run_name=run_name,
        extra_params={"total_params": total_params, "trainable_params": trainable_params},
    )
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Optional: causal mask if you want "read context then query" strictly.
    # Here we *allow full attention*, which is standard for set-based in-context regression.
    causal = False

    for step in range(1, config.steps + 1):
        tokens, y_q = sample_episode(config.batch_size, config.K, p, device)
        L = config.K + 1

        attn_mask = None
        if causal:
            # query can attend to context, context cannot attend to query if you want strict ordering
            m = torch.tril(torch.ones(L, L, device=device)).unsqueeze(0).repeat(config.batch_size, 1, 1)
            attn_mask = m

        yhat_all = model(tokens, attn_mask=attn_mask)
        yhat_q = yhat_all[:, -1]            # last token is query
        loss = F.mse_loss(yhat_q, y_q)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            with torch.no_grad():
                mae = (yhat_q - y_q).abs().mean().item()
            print(f"step {step:4d} | loss {loss.item():.6f} | MAE {mae:.4f}")
            logger.log(step, loss.item(), mae)

    return model

# Run
if __name__ == "__main__":
    
    # Ks = [8, 16, 32, 64]
    # d_models = [64, 128, 256]
    # attn_nonlinearities = ["softmax", "relu", "relu_norm", "sigmoid", "tanh", "identity"]    
    
    Ks = [8]
    d_models = [64]
    attn_nonlinearities = ["relu"]    
    
    for K in Ks:
        for d_model in d_models:
            for attn_nonlinearity in attn_nonlinearities:
                d_ff = d_model * 2
                print(f"Running K={K}, d_model={d_model}, d_ff={d_ff}")
                config = RunConfig(
                    steps=10000,
                    batch_size=16,
                    K=K,
                    d_model=d_model,
                    d_ff=d_ff,
                    lr=1e-3,
                    attn_nonlinearity=attn_nonlinearity,
                )
                run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
                model = train_in_context(config, log_dir="logs/torus_ic", run_name=run_name)
    # config = RunConfig(
    #     steps=2000,
    #     batch_size=64,
    #     K=32,
    #     d_model=128,
    #     d_ff=256,
    #     lr=1e-3,
    #     attn_nonlinearity="softmax",
    # )
    # run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # model = train_in_context(config, log_dir="logs/torus_ic", run_name=run_name)

