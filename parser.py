import json
import csv
from pathlib import Path
import pandas as pd

import math

def safe_float(x):
    if x is None:
        return float("nan")
    if isinstance(x, str):
        x = x.strip()
        if x == "" or x.lower() == "nan":
            return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")



def load_experiments_df(log_dir="logs/torus_trainable_T_layers_", include_history=False) -> pd.DataFrame:
    log_dir = Path(log_dir)
    rows = []

    for params_path in sorted(log_dir.glob("*_params.json")):
        run_name = params_path.name.replace("_params.json", "")
        metrics_path = log_dir / f"{run_name}_metrics.csv"

        # ---- params ----
        # with open(params_path, "r") as f:
        #     params = json.load(f)
        try:
            with open(params_path, "r") as f:
                params = json.load(f)
        except json.JSONDecodeError as e:
            print("\nBAD JSON FILE:", params_path)
            # show the first few lines with visible whitespace
            txt = params_path.read_text(errors="replace")
            print("---- first 200 chars ----")
            print(repr(txt[:200]))
            print("---- json error ----")
            print(e)
            continue


        # ---- metrics ----
        history = []
        if metrics_path.exists():
            with open(metrics_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    
                    print(run_name)
                    # history.append({
                    #     "step": int(r["step"]),
                    #     "loss": float(r["loss"]),
                    #     "mae": float(r["mae"]),
                    # })
                    
                    step = safe_float(r.get("step"))
                    loss = safe_float(r.get("loss"))
                    mae  = safe_float(r.get("mae"))

                    # Skip rows that are totally broken
                    if math.isnan(step):
                        continue

                    history.append({
                    "step": int(step),
                    "loss": loss,
                    "mae": mae,
                    })

        last = history[-1] if history else None
        best_loss = min(history, key=lambda r: r["loss"]) if history else None
        best_mae = min(history, key=lambda r: r["mae"]) if history else None
        avg_mae_3 = (
            sum(r["mae"] for r in history[-3:]) / len(history[-3:])
            if history else None
        )
        avg_mae_5 = (
            sum(r["mae"] for r in history[-5:]) / len(history[-5:])
            if history else None
        )
        avg_mae_7 = (
            sum(r["mae"] for r in history[-7:]) / len(history[-7:])
            if history else None
        )
        
        row = {
            "run_name": run_name,
            **params,  # columns: steps,batch_size,K,d_model,d_ff,lr,attn_nonlinearity,total_params,...
            "n_logged": len(history),
            "last_step": last["step"] if last else None,
            "last_loss": last["loss"] if last else None,
            "last_mae": last["mae"] if last else None,
            "best_loss": best_loss["loss"] if best_loss else None,
            "best_loss_step": best_loss["step"] if best_loss else None,
            "best_mae": best_mae["mae"] if best_mae else None,
            "best_mae_step": best_mae["step"] if best_mae else None,
            "avg_last_3_mae": avg_mae_3,
            "avg_last_5_mae": avg_mae_5,
            "avg_last_7_mae": avg_mae_7,
        }

        if include_history:
            row["history"] = history  # list of dicts

        rows.append(row)

    df = pd.DataFrame(rows)

    # nice default sort
    if not df.empty and "best_mae" in df.columns:
        df = df.sort_values(["best_mae", "best_loss"], ascending=[True, True], na_position="last").reset_index(drop=True)

    return df


# # usage
df = load_experiments_df("logs/torus_trainable_T_layers_")
print(df.head())

from pathlib import Path

df = load_experiments_df("logs/torus_trainable_T_layers_")  # assuming you already built df

out_dir = Path("logs/torus_trainable_T_layers_")
out_dir.mkdir(parents=True, exist_ok=True)

# Save as CSV
csv_path = out_dir / "experiments_summary.csv"
df.to_csv(csv_path, index=False)

# Save as Parquet (recommended if you have lots of runs / want types preserved)
parquet_path = out_dir / "experiments_summary.parquet"
df.to_parquet(parquet_path, index=False)

print(f"Saved:\n  {csv_path}\n  {parquet_path}")
