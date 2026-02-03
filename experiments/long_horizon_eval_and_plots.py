#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Long-horizon eval + plots (multi-model, per-model outputs)
==========================================================

- Procesa una lista de modelos (--models ...).
- Para cada modelo crea:  <output-root>/<model_name>/
  y guarda:
    - long_horizon_<testset>_detailed.csv
    - long_horizon_<testset>_agg.csv
    - long_horizon_<testset>_agg_by_gamma.csv
    - plots_overlay/ (figuras)
- Modo --max-run: usa 1000 trayectorias tanto para evaluar como para dibujar.     

python long_horizon_multimodel.py \
  --models ../training/results/neural_ode_physics_hybrid_complete_seed42_20260122_203659/best_model.pt ../training/results/pinode_v3_g0.5_auto0.1_seed42_20260119_113549/best_model.pt ../training/results/models/neural_ode_complete_seed42_20251204_073546/best_model.pt 
  --data-dir ../data/datasets \
  --test-set test_id \
  --horizons 45,60,90,120 \
  --t-max 90 \
  --n-trajectories-eval 1000 \
  --n-trajectories-plot 5 \
  --output-root ./lh_runs

Notas:
- GT a largo plazo: integra Bloch completo por RK4 (extendido hasta --t-max).
- PINN/PINNFourier: usan t_norm=t/30 (extrapolación => t_norm>1).
- neural_ode_physics: se evalúa con t REAL (0..t_max).
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------
# PATH del proyecto
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import NeuralODE, PINN, PINODE, NeuralODEConfig, PINNConfig, PINODEConfig
from models.pinode_symplectic import PINODESymplectic, PINODESymplecticConfig
from models.pinn_fourier import PINNFourier, PINNFourierConfig

# ✅ Neural ODE Physics
try:
    from models.neural_ode_physics import NeuralODEPhysics, NeuralODEPhysicsConfig
    NEURAL_ODE_PHYSICS_AVAILABLE = True
except ImportError:
    NEURAL_ODE_PHYSICS_AVAILABLE = False

try:
    from models.pinode_v3_optimized import PINODEv3, PINODEv3Config
    PINODE_V3_AVAILABLE = True
except ImportError:
    PINODE_V3_AVAILABLE = False

try:
    from models.pinode_v4 import PINODEv4, PINODEv4Config
    PINODE_V4_AVAILABLE = True
except ImportError:
    PINODE_V4_AVAILABLE = False


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
def load_test_data(data_dir: Path, dataset: str = "test_id") -> List[Dict]:
    if dataset == "test_id":
        csv_path = data_dir / "test" / "test_id.csv"
    elif dataset == "test_resonance":
        csv_path = data_dir / "test" / "test_resonance.csv"
    elif dataset == "val":
        csv_path = data_dir / "val" / "val.csv"
    else:
        csv_path = Path(dataset)

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró: {csv_path}")

    df = pd.read_csv(csv_path)
    trajectories = []

    for traj_id, group in df.groupby("trajectory_id"):
        group = group.sort_values("t")
        t = torch.tensor(group["t"].values, dtype=torch.float32)

        trajectories.append({
            "id": int(traj_id),
            "t": t,                       # (T,)
            "t_norm": t / 30.0,           # (T,)
            "states": torch.tensor(group[["u", "v", "w"]].values, dtype=torch.float32),  # (T,3)
            "params": torch.tensor([
                group["Omega_R"].iloc[0],
                group["detuning"].iloc[0],
                group["gamma"].iloc[0],
            ], dtype=torch.float32),      # (3,)
            "Omega": float(group["Omega_R"].iloc[0]),
            "Delta": float(group["detuning"].iloc[0]),
            "gamma": float(group["gamma"].iloc[0]),
        })

    return trajectories


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
def load_model(model_path: Path, device: str = "cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_path_str = str(model_path).lower()

    # ✅ MUY IMPORTANTE: neural_ode_physics antes que neural_ode
    if "neural_ode_physics" in model_path_str:
        if not NEURAL_ODE_PHYSICS_AVAILABLE:
            raise RuntimeError("No se pudo importar NeuralODEPhysics en este entorno.")

        def _get(k, default=None):
            if isinstance(config, dict):
                return config.get(k, default)
            return getattr(config, k, default)

        model_config = NeuralODEPhysicsConfig(
            hidden_dim=int(_get("hidden_dim", 256)),
            num_layers=int(_get("num_layers", 4)),
            activation=str(_get("activation", "silu")),
            use_layer_norm=bool(_get("use_layer_norm", True)),
            dropout=float(_get("dropout", 0.0)),
            solver=str(_get("solver", "rk4")),
            loss_mode=str(_get("loss_mode", "hybrid")),
            physics_target=str(_get("physics_target", "complete")),
            lambda_physics=float(_get("lambda_physics", 1.0)),
            lambda_data=float(_get("lambda_data", 1.0)),
            n_collocation=int(_get("n_collocation", 2000)),
            collocation_strategy=str(_get("collocation_strategy", _get("collocation_source", "uniform"))),
        )
        model = NeuralODEPhysics(model_config).to(device)
        if hasattr(model, "set_device"):
            model.set_device(device)
        model_type = "neural_ode_physics"

    elif "pinode_v3" in model_path_str:
        if not PINODE_V3_AVAILABLE:
            raise RuntimeError("PINODEv3 no disponible en este entorno.")
        model_config = PINODEv3Config(
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4),
            activation=config.get("activation", "silu"),
            use_layer_norm=config.get("use_layer_norm", True),
            gate_value=config.get("gate_value", 0.5),
        )
        model = PINODEv3(model_config).to(device)
        model_type = "pinode_v3"

    elif "pinode_v4" in model_path_str:
        if not PINODE_V4_AVAILABLE:
            raise RuntimeError("PINODEv4 no disponible en este entorno.")
        model_config = PINODEv4Config(
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4),
            activation=config.get("activation", "silu"),
            use_layer_norm=config.get("use_layer_norm", True),
            dropout=config.get("dropout", 0.0),
            gate_value=config.get("gate_value", 0.5),
            lambda_cons=config.get("lambda_cons", 0.0),
            lambda_cons_auto=config.get("lambda_cons_auto", False),
            lambda_cons_scale=config.get("lambda_cons_scale", 0.1),
        )
        model = PINODEv4(model_config).to(device)
        model_type = "pinode_v4"

    elif "pinode_symplectic" in model_path_str:
        model_config = PINODESymplecticConfig(
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_layers=int(config.get("num_layers", 4)),
            activation=str(config.get("activation", "tanh")),
            physics_mode=str(config.get("physics_mode", "complete")),
            use_omega_eff=bool(config.get("use_omega_eff", False)),
            g_min=float(config.get("g_min", 0.05)),
            conservation_gamma_scale=float(config.get("conservation_gamma_scale", 20.0)),
        )
        model = PINODESymplectic(model_config).to(device)
        model_type = "pinode_symplectic"

    elif "neural_ode" in model_path_str:
        model = NeuralODE(NeuralODEConfig()).to(device)
        model_type = "neural_ode"

    elif "pinn_fourier" in model_path_str:
        model_config = PINNFourierConfig(
            num_frequencies=config.get("num_frequencies", 32),
            freq_scale=config.get("freq_scale", 10.0),
            hidden_dims=config.get("hidden_dims", [256, 256, 256, 256]),
            physics_mode=config.get("physics_mode", "complete"),
        )
        model = PINNFourier(model_config).to(device)
        model_type = "pinn_fourier"

    elif "pinn" in model_path_str:
        model = PINN(PINNConfig(physics_mode=config.get("physics_mode", "complete"))).to(device)
        model_type = "pinn"

    elif "pinode" in model_path_str:
        model = PINODE(PINODEConfig(physics_mode=config.get("physics_mode", "complete"))).to(device)
        model_type = "pinode"

    else:
        raise ValueError(f"No se pudo detectar el tipo de modelo: {model_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_name = model_path.parent.name
    return model, model_type, model_name, config


@torch.no_grad()
def predict_trajectory(model, model_type: str, t: torch.Tensor, params: torch.Tensor, y0: torch.Tensor, device: str) -> np.ndarray:
    t = t.to(device)
    t_norm = (t / 30.0).to(device)

    p = params.unsqueeze(0).to(device)     # (1,3)
    y0 = y0.unsqueeze(0).to(device)        # (1,3)

    # ✅ neural_ode_physics usa t real
    if model_type in ["neural_ode_physics", "pinode_v3", "pinode_v4"]:
        pred = model(y0, t, p)
        return pred.squeeze(1).cpu().numpy()

    # resto ODEs como tus scripts (t_norm)
    if model_type in ["neural_ode", "pinode", "pinode_symplectic"]:
        pred = model(y0, t_norm, p)
        return pred.squeeze(1).cpu().numpy()  # (T,3)

    if model_type in ["pinn", "pinn_fourier"]:
        n = len(t_norm)
        x = torch.zeros(n, 4, device=device)
        x[:, 0] = t_norm
        x[:, 1] = p[0, 0]
        x[:, 2] = p[0, 1]
        x[:, 3] = p[0, 2]
        return model(x).cpu().numpy()

    raise ValueError(f"Tipo no soportado: {model_type}")


# ---------------------------------------------------------------------
# GT extendido (Bloch completo, RK4)
# ---------------------------------------------------------------------
def bloch_rhs_complete_np(y: np.ndarray, params: np.ndarray) -> np.ndarray:
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    Omega = params[:, 0:1]
    Delta = params[:, 1:2]
    gamma = params[:, 2:3]

    du = -gamma * u + Delta * v
    dv = -gamma * v - Delta * u - Omega * w
    dw = -2.0 * gamma * (w + 1.0) + Omega * v

    return np.concatenate([du, dv, dw], axis=-1)


def rk4_step_np(y: np.ndarray, params: np.ndarray, dt: float) -> np.ndarray:
    k1 = bloch_rhs_complete_np(y, params)
    k2 = bloch_rhs_complete_np(y + 0.5 * dt * k1, params)
    k3 = bloch_rhs_complete_np(y + 0.5 * dt * k2, params)
    k4 = bloch_rhs_complete_np(y + dt * k3, params)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def generate_extended_gt(t: np.ndarray, y0: np.ndarray, params: np.ndarray) -> np.ndarray:
    T = len(t)
    y = y0.reshape(1, 3).copy()
    p = params.reshape(1, 3).copy()
    out = np.zeros((T, 3), dtype=np.float64)
    out[0] = y[0]
    for i in range(T - 1):
        dt = float(t[i+1] - t[i])
        y = rk4_step_np(y, p, dt)
        out[i+1] = y[0]
    return out.astype(np.float32)


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    denom = np.maximum(np.linalg.norm(y_true, axis=1), 1e-8)
    rel = float(np.mean(np.linalg.norm(err, axis=1) / denom))

    norm_pred = np.linalg.norm(y_pred, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_error = float(np.mean(np.abs(norm_pred - norm_true)))
    norm_violation = float(np.mean(np.maximum(0.0, norm_pred - 1.0)))
    max_norm = float(np.max(norm_pred))

    final_error = float(np.linalg.norm(y_pred[-1] - y_true[-1]))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rel_error": rel,
        "norm_error": norm_error,
        "norm_violation": norm_violation,
        "max_norm": max_norm,
        "final_error": final_error,
    }


# ---------------------------------------------------------------------
# PLOT (un modelo por figura en este script)
# ---------------------------------------------------------------------
def plot_long_horizon_overlay(
    t: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    params: np.ndarray,
    traj_id: int,
    model_name: str,
    horizon: float,
    save_path: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    omega, delta, gamma = params.tolist()

    def _plot_component(ax, idx, name):
        ax.plot(t, y_true[:, idx], linewidth=2, label="GT")
        ax.plot(t, y_pred[:, idx], linestyle="--", linewidth=1.2, label=model_name)
        ax.set_xlabel("t")
        ax.set_ylabel(name)
        ax.set_title(f"{name}(t)")
        ax.grid(True, alpha=0.3)

    _plot_component(axes[0, 0], 0, "u")
    _plot_component(axes[0, 1], 1, "v")
    _plot_component(axes[1, 0], 2, "w")

    ax = axes[1, 1]
    ax.plot(t, np.linalg.norm(y_true, axis=1), linewidth=2, label="GT norm")
    ax.plot(t, np.linalg.norm(y_pred, axis=1), linestyle="--", linewidth=1.2, label=f"{model_name} norm")
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5, label="||B||=1")
    ax.set_xlabel("t")
    ax.set_ylabel("||B||")
    ax.set_title("Norma del vector de Bloch")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.35])

    fig.suptitle(
        f"{model_name} | Trayectoria {traj_id} | H={horizon:g}\nΩ={omega:.2f}, Δ={delta:.2f}, γ={gamma:.4f}",
        fontsize=12
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--test-set", type=str, default="test_id", choices=["test_id", "test_resonance", "val"])
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--horizons", type=str, default="30,45,60,90")
    parser.add_argument("--t-max", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=None)

    parser.add_argument("--n-trajectories-eval", type=int, default=None)
    parser.add_argument("--n-trajectories-plot", type=int, default=5)
    parser.add_argument("--plot-selection", type=str, default="random", choices=["random"])

    parser.add_argument("--max-run", action="store_true")
    parser.add_argument("--output-root", type=str, default="./long_horizon_outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    horizons = sorted(set(float(x.strip()) for x in args.horizons.split(",") if x.strip()))
    if args.t_max < max(horizons):
        raise ValueError("t-max debe ser >= max(horizons)")

    if args.max_run:
        args.n_trajectories_eval = 1000
        args.n_trajectories_plot = 1000

    data_dir = Path(args.data_dir)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    trajectories = load_test_data(data_dir, args.test_set)
    trajectories_eval = trajectories[:args.n_trajectories_eval] if args.n_trajectories_eval is not None else trajectories

    print(f"Trayectorias disponibles: {len(trajectories)} | Evaluación: {len(trajectories_eval)}")

    global_rows = []

    for model_path_str in args.models:
        model_path = Path(model_path_str)
        model, model_type, model_name, config = load_model(model_path, device=args.device)

        model_out = out_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        plots_dir = model_out / "plots_overlay"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # plots aleatorios (full eval)
        n_plot = min(args.n_trajectories_plot or 0, len(trajectories_eval))
        plot_indices = list(range(n_plot))
        if n_plot > 0:
            idxs = np.arange(len(trajectories_eval))
            np.random.shuffle(idxs)
            plot_indices = idxs[:n_plot].tolist()
        plot_set = set(plot_indices)

        rows = []

        for i_traj, traj in enumerate(tqdm(trajectories_eval, desc=f"Evaluando {model_name}")):
            traj_id = traj["id"]
            params = traj["params"].numpy().astype(np.float32)
            y0 = traj["states"][0].numpy().astype(np.float32)

            # dt
            t0 = float(traj["t"][0].item())
            if args.dt is None:
                dt = float((traj["t"][1] - traj["t"][0]).item()) if len(traj["t"]) > 1 else 0.05
            else:
                dt = float(args.dt)

            t_ext = np.arange(t0, float(args.t_max) + 1e-9, dt, dtype=np.float32)
            y_true_ext = generate_extended_gt(t_ext, y0, params)

            t_ext_torch = torch.tensor(t_ext, dtype=torch.float32)
            y_pred_ext = predict_trajectory(
                model, model_type,
                t_ext_torch,
                torch.tensor(params, dtype=torch.float32),
                torch.tensor(y0, dtype=torch.float32),
                args.device
            )

            for H in horizons:
                mask = t_ext <= (H + 1e-9)
                met = compute_metrics(y_true_ext[mask], y_pred_ext[mask])

                row = {
                    "trajectory_id": traj_id,
                    "Omega": float(params[0]),
                    "Delta": float(params[1]),
                    "gamma": float(params[2]),
                    "horizon": float(H),
                    "model": model_name,
                    **met,
                }
                rows.append(row)
                global_rows.append(row)

                # plot solo si está seleccionado
                if i_traj in plot_set:
                    save_path = plots_dir / f"traj_{traj_id:04d}_H{int(H):03d}.png"
                    plot_long_horizon_overlay(
                        t=t_ext[mask],
                        y_true=y_true_ext[mask],
                        y_pred=y_pred_ext[mask],
                        params=params,
                        traj_id=traj_id,
                        model_name=model_name,
                        horizon=H,
                        save_path=save_path,
                    )

        df = pd.DataFrame(rows)
        detailed_csv = model_out / f"long_horizon_{args.test_set}_detailed.csv"
        df.to_csv(detailed_csv, index=False)

        df["gamma_group"] = np.where(df["gamma"] < 1e-6, "gamma_0", "gamma_pos")

        agg = df.groupby(["model", "horizon"]).agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rel_mean=("rel_error", "mean"),
            rel_std=("rel_error", "std"),
            norm_violation_mean=("norm_violation", "mean"),
            final_error_mean=("final_error", "mean"),
        ).reset_index()
        agg_csv = model_out / f"long_horizon_{args.test_set}_agg.csv"
        agg.to_csv(agg_csv, index=False)

        agg_gamma = df.groupby(["model", "horizon", "gamma_group"]).agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rel_mean=("rel_error", "mean"),
            rel_std=("rel_error", "std"),
            norm_violation_mean=("norm_violation", "mean"),
            final_error_mean=("final_error", "mean"),
        ).reset_index()
        agg_gamma_csv = model_out / f"long_horizon_{args.test_set}_agg_by_gamma.csv"
        agg_gamma.to_csv(agg_gamma_csv, index=False)

        print(f"\n✅ Guardado {model_name}:")
        print(f" - {detailed_csv}")
        print(f" - {agg_csv}")
        print(f" - {agg_gamma_csv}")
        print(f" - Plots: {plots_dir} ({n_plot} trayectorias x {len(horizons)} horizontes)")

    global_df = pd.DataFrame(global_rows)
    global_csv = out_root / f"ALL_MODELS_long_horizon_{args.test_set}_detailed.csv"
    global_df.to_csv(global_csv, index=False)

    print("\n" + "=" * 90)
    print("✅ FIN")
    print(f"Consolidado multi-modelo: {global_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()