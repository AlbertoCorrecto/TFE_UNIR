#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualización Completa de Trayectorias de Test
==============================================

Genera gráficas individuales para TODAS las trayectorias de test
de un modelo entrenado.

Uso:
    python visualize_all.py --model-path ../training/results/models/neural_ode_complete_seed42_20251204_073546/best_model.pt --data-dir ../data/datasets --test-set test_extrap_params
    python visualize_all.py --model-path ../training/results/models/pinode_v3_g0.5_auto0.1_seed42_20260119_113549/best_model.pt --data-dir ../data/datasets --test-set test_extrap_params
    python visualize_all.py --model-path ../training/results/neural_ode_physics_hybrid_complete_seed42_20260122_203659/best_model.pt --data-dir ../data/datasets
    python visualize_all.py --model-path ... --data-dir ... --experiment-name E1_neural_ode
    python visualize_all.py --model-path ... --data-dir ... --test-set test_resonance
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


# =============================================================================
# PATH DEL PROYECTO
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# IMPORTS DE MODELOS
# =============================================================================
MODELS_AVAILABLE = True
PINODE_V3_AVAILABLE = True
PINODE_V4_AVAILABLE = True

try:
    from models import NeuralODE, PINN, PINODE, NeuralODEConfig, PINNConfig, PINODEConfig
    from models.pinode_symplectic import PINODESymplectic, PINODESymplecticConfig
    from models.pinn_fourier import PINNFourier, PINNFourierConfig
except Exception:
    MODELS_AVAILABLE = False
    print("⚠️ No se pudieron importar algunos modelos base. Revisa tu PYTHONPATH.")

try:
    from models.pinode_v3_optimized import PINODEv3, PINODEv3Config
except Exception:
    PINODE_V3_AVAILABLE = False
    print("⚠️ PINODEv3 no disponible.")

try:
    from models.pinode_v4 import PINODEv4, PINODEv4Config
except Exception:
    PINODE_V4_AVAILABLE = False
    print("⚠️ PINODEv4 no disponible.")


# =============================================================================
# CARGA DE DATOS
# =============================================================================
def load_test_data(data_dir: Path, dataset: str = "test_id") -> List[Dict]:
    """Carga TODAS las trayectorias de test"""
    if dataset == "test_id":
        csv_path = data_dir / "test" / "test_id.csv"
    elif dataset == "test_resonance":
        csv_path = data_dir / "test" / "test_resonance.csv"
    elif dataset == "test_extrap_params":
        csv_path = data_dir / "test" / "test_extrap_params.csv"
    elif dataset == "val":
        csv_path = data_dir / "val" / "val.csv"
    else:
        raise ValueError(f"Dataset desconocido: {dataset}")

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró: {csv_path}")

    print(f"   📂 Cargando: {csv_path}")
    df = pd.read_csv(csv_path)

    trajectories = []
    grouped = df.groupby("trajectory_id")

    for traj_id, group in grouped:
        group = group.sort_values("t")
        t = group["t"].values.astype(np.float32)

        trajectories.append({
            "id": int(traj_id) if str(traj_id).isdigit() else traj_id,
            "t": torch.tensor(t, dtype=torch.float32),
            "t_norm": torch.tensor(t / 30.0, dtype=torch.float32),
            "states": torch.tensor(group[["u", "v", "w"]].values.astype(np.float32), dtype=torch.float32),
            "params": torch.tensor([
                float(group["Omega_R"].iloc[0]),
                float(group["detuning"].iloc[0]),
                float(group["gamma"].iloc[0]),
            ], dtype=torch.float32),
        })

    return trajectories


# =============================================================================
# CARGA DE MODELOS
# =============================================================================
def load_model(model_path: Path, device: str = "cpu"):
    """Carga un modelo desde checkpoint y detecta su tipo por el path."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {}) or {}
    model_path_str = str(model_path).lower()

    # 1) Neural ODE Physics (antes que 'neural_ode')
    if "neural_ode_physics" in model_path_str:
        from models.neural_ode_physics import NeuralODEPhysics, NeuralODEPhysicsConfig

        model_config = NeuralODEPhysicsConfig(
            hidden_dim=int(config.get("hidden_dim", 256)),
            num_layers=int(config.get("num_layers", 4)),
            activation=str(config.get("activation", "silu")),
            use_layer_norm=bool(config.get("use_layer_norm", True)),
            dropout=float(config.get("dropout", 0.0)),
            solver=str(config.get("solver", "rk4")),

            loss_mode=str(config.get("loss_mode", "hybrid")),
            physics_target=str(config.get("physics_target", "complete")),
            lambda_physics=float(config.get("lambda_physics", 1.0)),
            lambda_data=float(config.get("lambda_data", 1.0)),
            n_collocation=int(config.get("n_collocation", 2000)),
            collocation_strategy=str(config.get("collocation_strategy", "uniform")),
        )

        model = NeuralODEPhysics(model_config).to(device)
        model_type = "neural_ode_physics"

    # 2) PINODE v3
    elif "pinode_v3" in model_path_str:
        if not PINODE_V3_AVAILABLE:
            raise ImportError("PINODEv3 no está disponible en tu entorno.")
        from models.pinode_v3_optimized import PINODEv3, PINODEv3Config

        model_config = PINODEv3Config(
            hidden_dim=int(config.get("hidden_dim", 256)),
            num_layers=int(config.get("num_layers", 4)),
            activation=str(config.get("activation", "silu")),
            use_layer_norm=bool(config.get("use_layer_norm", True)),
            gate_value=float(config.get("gate_value", 0.5)),
        )
        model = PINODEv3(model_config).to(device)
        model_type = "pinode_v3"

    # 3) PINODE v4
    elif "pinode_v4" in model_path_str:
        if not PINODE_V4_AVAILABLE:
            raise ImportError("PINODEv4 no está disponible en tu entorno.")
        from models.pinode_v4 import PINODEv4, PINODEv4Config

        model_config = PINODEv4Config(
            hidden_dim=int(config.get("hidden_dim", 256)),
            num_layers=int(config.get("num_layers", 4)),
            activation=str(config.get("activation", "silu")),
            use_layer_norm=bool(config.get("use_layer_norm", True)),
            dropout=float(config.get("dropout", 0.0)),
            gate_value=float(config.get("gate_value", 0.5)),
            lambda_cons=float(config.get("lambda_cons", 0.0)),
            lambda_cons_auto=bool(config.get("lambda_cons_auto", False)),
            lambda_cons_scale=float(config.get("lambda_cons_scale", 0.1)),
        )
        model = PINODEv4(model_config).to(device)
        model_type = "pinode_v4"

    # 4) PINODE Symplectic
    elif "pinode_symplectic" in model_path_str:
        model_config = PINODESymplecticConfig(
            hidden_dim=int(config.get("hidden_dim", 128)),
            num_layers=int(config.get("num_layers", 4)),
            physics_mode=str(config.get("physics_mode", "complete")),
            use_omega_eff=bool(config.get("use_omega_eff", True)),
            g_min=float(config.get("g_min", 0.05)),
            g_init=float(config.get("g_init", 0.30)),
            g_decay=float(config.get("g_decay", 0.999)),
            lambda_residual=float(config.get("lambda_residual", 0.0)),
            lambda_conservation=float(config.get("lambda_conservation", 1.0)),
            conservation_gamma_scale=float(config.get("conservation_gamma_scale", 20.0)),
        )
        model = PINODESymplectic(model_config).to(device)
        model_type = "pinode_symplectic"

    # 5) Neural ODE
    elif "neural_ode" in model_path_str:
        model = NeuralODE(NeuralODEConfig()).to(device)
        model_type = "neural_ode"

    # 6) PINODE original
    elif "pinode" in model_path_str:
        physics_mode = str(config.get("physics_mode", "complete"))
        model = PINODE(PINODEConfig(physics_mode=physics_mode)).to(device)
        model_type = "pinode"

    # 7) PINN Fourier
    elif "pinn_fourier" in model_path_str:
        physics_mode = str(config.get("physics_mode", "complete"))
        num_frequencies = int(config.get("num_frequencies", 32))
        freq_scale = float(config.get("freq_scale", 10.0))
        hidden_dims = config.get("hidden_dims", [256, 256, 256, 256])

        model_config = PINNFourierConfig(
            num_frequencies=num_frequencies,
            freq_scale=freq_scale,
            hidden_dims=hidden_dims,
            physics_mode=physics_mode,
        )
        model = PINNFourier(model_config).to(device)
        model_type = "pinn_fourier"

    # 8) PINN original
    elif "pinn" in model_path_str:
        physics_mode = str(config.get("physics_mode", "complete"))
        model = PINN(PINNConfig(physics_mode=physics_mode)).to(device)
        model_type = "pinn"

    else:
        raise ValueError(f"No se pudo detectar el tipo de modelo: {model_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, model_type, config, checkpoint


# =============================================================================
# PREDICCIÓN (CLAVE: elegir t o t_norm correctamente)
# =============================================================================
def _choose_time(model_type: str, traj: Dict, device: str):
    t = traj["t"].to(device)
    t_norm = traj["t_norm"].to(device)

    # ODEs que usan tiempo raw
    if model_type in ["neural_ode_physics", "pinode_v3", "pinode_v4"]:
        return t, "t"

    # PINNs -> normalizado
    if model_type in ["pinn", "pinn_fourier"]:
        return t_norm, "t_norm"

    # Por defecto (mantiene tu comportamiento anterior)
    return t_norm, "t_norm"


def predict_single(model, model_type: str, traj: Dict, device: str, debug_once: bool = False) -> np.ndarray:
    """Genera predicción para una trayectoria."""
    with torch.no_grad():
        t_used, time_label = _choose_time(model_type, traj, device)
        params = traj["params"].unsqueeze(0).to(device)
        initial_state = traj["states"][0].unsqueeze(0).to(device)

        if debug_once:
            print(f"   ⏱️  Tiempo usado para {model_type}: {time_label} | "
                  f"tT={traj['t'][-1].item():.3f}, t_normT={traj['t_norm'][-1].item():.3f}")

        if model_type in ["neural_ode", "neural_ode_physics", "pinode", "pinode_symplectic", "pinode_v3", "pinode_v4"]:
            pred = model(initial_state, t_used, params)
            pred = pred.squeeze(1).cpu().numpy()  # (T, 3)

        elif model_type in ["pinn", "pinn_fourier"]:
            n_points = len(t_used)
            x = torch.zeros(n_points, 4, device=device)
            x[:, 0] = t_used
            x[:, 1] = params[0, 0]
            x[:, 2] = params[0, 1]
            x[:, 3] = params[0, 2]
            pred = model(x).cpu().numpy()

        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    return pred


# =============================================================================
# VISUALIZACIÓN
# =============================================================================
def compute_trajectory_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    y_true_norm = np.maximum(np.linalg.norm(y_true, axis=1), 1e-8)
    rel_error = float(np.mean(np.linalg.norm(errors, axis=1) / y_true_norm))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    norm_pred = np.linalg.norm(y_pred, axis=1)
    norm_violation = float(np.mean(np.maximum(0, norm_pred - 1.0)))

    return {
        "mse": mse,
        "rmse": rmse,
        "rel_error": rel_error,
        "r2": r2,
        "norm_violation": norm_violation,
    }


def plot_trajectory_full(
    t: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    params: np.ndarray,
    traj_id: int,
    model_name: str,
    save_path: Path
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    omega, delta, gamma = params

    # u(t)
    ax = axes[0, 0]
    ax.plot(t, y_true[:, 0], label="Ground Truth", linewidth=2)
    ax.plot(t, y_pred[:, 0], "--", label="Predicción", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    ax.set_title("Componente u")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # v(t)
    ax = axes[0, 1]
    ax.plot(t, y_true[:, 1], label="Ground Truth", linewidth=2)
    ax.plot(t, y_pred[:, 1], "--", label="Predicción", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("v")
    ax.set_title("Componente v")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # w(t)
    ax = axes[1, 0]
    ax.plot(t, y_true[:, 2], label="Ground Truth", linewidth=2)
    ax.plot(t, y_pred[:, 2], "--", label="Predicción", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("w")
    ax.set_title("Componente w")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Norma Bloch
    ax = axes[1, 1]
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)
    ax.plot(t, norm_true, label="GT Norm", linewidth=2)
    ax.plot(t, norm_pred, "--", label="Pred Norm", linewidth=1.5)
    ax.axhline(y=1.0, linestyle=":", alpha=0.6, label="||B||=1")
    ax.set_xlabel("t")
    ax.set_ylabel("||B||")
    ax.set_title("Norma del vector de Bloch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.3])

    # Métricas
    m = compute_trajectory_metrics(y_true, y_pred)
    fig.suptitle(
        f"{model_name} - Trayectoria {traj_id}\n"
        f"Ω={omega:.2f}, Δ={delta:.2f}, γ={gamma:.3f} | "
        f"MSE={m['mse']:.2e}, RelErr={m['rel_error']*100:.1f}%",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Visualización completa de trayectorias de test")
    parser.add_argument("--model-path", type=str, required=True, help="Path al modelo .pt")
    parser.add_argument("--data-dir", type=str, required=True, help="Directorio de datos")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Nombre del experimento (default: derivado del modelo)")
    parser.add_argument("--test-set", type=str, default="test_id",
                        choices=["test_id", "test_resonance", "val", "test_extrap_params"],
                        help="Conjunto de test a usar")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo")
    parser.add_argument("--max-traj", type=int, default=None, help="Máximo de trayectorias (default: todas)")
    parser.add_argument("--gamma-min", type=float, default=None, help="Filtrar trayectorias con gamma >= este valor")
    parser.add_argument("--gamma-max", type=float, default=None, help="Filtrar trayectorias con gamma <= este valor")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)

    exp_name = args.experiment_name if args.experiment_name else model_path.parent.name

    # Output dir
    test_set_name = args.test_set
    if args.gamma_min is not None:
        test_set_name += f"_gamma_min{args.gamma_min}"
    if args.gamma_max is not None:
        test_set_name += f"_gamma_max{args.gamma_max}"

    output_dir = Path(__file__).parent / "visualizations" / exp_name / test_set_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" 📊 VISUALIZACIÓN COMPLETA DE TRAYECTORIAS")
    print("=" * 60)
    print(f" 🧠 Modelo: {model_path}")
    print(f" 📁 Datos: {data_dir}")
    print(f" 🧪 Test set: {args.test_set}")
    if args.gamma_min is not None:
        print(f" 🔬 Filtro: γ >= {args.gamma_min}")
    if args.gamma_max is not None:
        print(f" 🔬 Filtro: γ <= {args.gamma_max}")
    print(f" 📂 Output: {output_dir}")
    print("=" * 60)

    # Modelo
    print("\n🔄 Cargando modelo...")
    model, model_type, config, checkpoint = load_model(model_path, args.device)
    print(f"   Tipo: {model_type}")

    # Mostrar algo útil de validación (varía según trainer)
    val_msg = checkpoint.get("val_loss", None)
    if val_msg is None:
        val_msg = checkpoint.get("val_mse", None)
    if val_msg is None:
        val_msg = checkpoint.get("val_traj_mse", None)
    if val_msg is None:
        val_msg = checkpoint.get("val_deriv_mse", None)
    print(f"   Val metric (train): {val_msg if val_msg is not None else 'N/A'}")

    if "physics_target" in config:
        print(f"   Physics target: {config.get('physics_target')} | Loss mode: {config.get('loss_mode')}")

    # Datos
    print("\n🔄 Cargando datos de test...")
    trajectories = load_test_data(data_dir, args.test_set)

    if args.max_traj:
        trajectories = trajectories[:args.max_traj]

    # Filtrar por gamma
    if args.gamma_min is not None or args.gamma_max is not None:
        filtered = []
        for traj in trajectories:
            gamma = float(traj["params"][2].item())
            if args.gamma_min is not None and gamma < args.gamma_min:
                continue
            if args.gamma_max is not None and gamma > args.gamma_max:
                continue
            filtered.append(traj)
        trajectories = filtered
        print(f"   Filtradas por gamma: {len(trajectories)}")

    print(f"   Trayectorias: {len(trajectories)}")
    if len(trajectories) > 0:
        print(f"   Ejemplo tT={trajectories[0]['t'][-1].item():.3f} | t_normT={trajectories[0]['t_norm'][-1].item():.3f}")

    # Loop
    print(f"\n🎨 Generando {len(trajectories)} gráficas...")
    all_metrics = []

    debug_once = True
    for traj in tqdm(trajectories, desc="Procesando"):
        traj_id = traj["id"]

        y_pred = predict_single(model, model_type, traj, args.device, debug_once=debug_once)
        debug_once = False

        y_true = traj["states"].numpy()
        t = traj["t"].numpy()
        params = traj["params"].numpy()

        m = compute_trajectory_metrics(y_true, y_pred)
        m["trajectory_id"] = traj_id
        m["Omega"] = float(params[0])
        m["Delta"] = float(params[1])
        m["gamma"] = float(params[2])
        all_metrics.append(m)

        save_path = output_dir / f"traj_{int(traj_id):04d}.png" if str(traj_id).isdigit() else output_dir / f"traj_{traj_id}.png"
        plot_trajectory_full(
            t=t,
            y_true=y_true,
            y_pred=y_pred,
            params=params,
            traj_id=int(traj_id) if str(traj_id).isdigit() else -1,
            model_name=model_type.upper(),
            save_path=save_path,
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = output_dir / "metrics_per_trajectory.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Resumen
    print("\n" + "=" * 60)
    print(" ✅ COMPLETADO")
    print("=" * 60)
    print(f" 📊 Gráficas generadas: {len(trajectories)}")
    print(f" 📂 Ubicación: {output_dir}")
    print(f" 📄 Métricas: {metrics_csv_path}\n")

    if len(metrics_df) > 0:
        print(" 📈 RESUMEN DE MÉTRICAS:")
        print(f"    MSE medio:       {metrics_df['mse'].mean():.6e}")
        print(f"    MSE std:         {metrics_df['mse'].std():.6e}")
        print(f"    R² medio:        {metrics_df['r2'].mean():.4f}")
        print(f"    Error rel medio: {metrics_df['rel_error'].mean()*100:.2f}%")

        worst_idx = metrics_df["mse"].idxmax()
        best_idx = metrics_df["mse"].idxmin()
        print(f"    Peor MSE:        {metrics_df.loc[worst_idx, 'mse']:.6e} (traj {metrics_df.loc[worst_idx, 'trajectory_id']})")
        print(f"    Mejor MSE:       {metrics_df.loc[best_idx, 'mse']:.6e} (traj {metrics_df.loc[best_idx, 'trajectory_id']})")

    print("=" * 60)


if __name__ == "__main__":
    main()
