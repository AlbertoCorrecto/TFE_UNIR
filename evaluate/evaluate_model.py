#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluación Completa de Modelos
===============================

Evalúa modelos entrenados calculando métricas separadas para:
- γ = 0 (sin disipación)
- γ > 0 (con disipación)

Uso:
    python evaluate_model.py --model-path ../training/results/.../best_model.pt --data-dir ../data/datasets
    python evaluate_model.py --model-path ... --data-dir ... --test-set test_resonance
    python evaluate_model.py --model-path ... --data-dir ... --save-individual
    python evaluate_model.py --model-path ... --data-dir ... --max-traj 1000
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


# =============================================================================
# PATH DEL PROYECTO
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# IMPORTS DE MODELOS (según tu repo)
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
    print("⚠️ No se pudieron importar modelos del proyecto. Se usará fallback parcial.")

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
# FALLBACK MINIMAL (solo si fallan imports)
# =============================================================================
if not MODELS_AVAILABLE:
    @dataclass
    class PINNFourierConfig:
        num_frequencies: int = 32
        freq_scale: float = 10.0
        include_input: bool = True
        hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256, 256])
        activation: str = "tanh"
        use_layer_norm: bool = True
        dropout: float = 0.0
        physics_mode: str = "complete"

    class FourierFeatures(nn.Module):
        def __init__(self, num_frequencies: int = 32, freq_scale: float = 10.0, include_input: bool = True):
            super().__init__()
            self.num_frequencies = num_frequencies
            self.include_input = include_input
            frequencies = torch.exp(torch.linspace(0, math.log(freq_scale), num_frequencies))
            self.register_buffer('frequencies', frequencies)
            self.output_dim = 2 * num_frequencies + (1 if include_input else 0)

        def forward(self, t):
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            angles = 2 * math.pi * t * self.frequencies
            features = [torch.sin(angles), torch.cos(angles)]
            if self.include_input:
                features.append(t)
            return torch.cat(features, dim=-1)

    class PINNFourier(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.fourier = FourierFeatures(config.num_frequencies, config.freq_scale, config.include_input)
            input_dim = self.fourier.output_dim + 3
            layers = []
            dims = [input_dim] + list(config.hidden_dims)
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if config.use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.Tanh() if config.activation == "tanh" else nn.SiLU())
            self.backbone = nn.Sequential(*layers)
            self.output_layer = nn.Linear(config.hidden_dims[-1], 3)

        def forward(self, x):
            t_features = self.fourier(x[:, 0])
            features = torch.cat([t_features, x[:, 1:4]], dim=-1)
            return self.output_layer(self.backbone(features))


# =============================================================================
# CARGA DE DATOS
# =============================================================================
def load_test_data(data_dir: Path, dataset: str = "test_id") -> List[Dict]:
    """Carga trayectorias de test desde CSV."""
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
            "t_norm": torch.tensor(t / 30.0, dtype=torch.float32),  # útil para PINNs si quieres
            "states": torch.tensor(group[["u", "v", "w"]].values.astype(np.float32), dtype=torch.float32),
            "params": torch.tensor([
                float(group["Omega_R"].iloc[0]),
                float(group["detuning"].iloc[0]),
                float(group["gamma"].iloc[0]),
            ], dtype=torch.float32),
            "Omega": float(group["Omega_R"].iloc[0]),
            "Delta": float(group["detuning"].iloc[0]),
            "gamma": float(group["gamma"].iloc[0]),
        })

    return trajectories


# =============================================================================
# CARGA DE MODELOS
# =============================================================================
def load_model(model_path: Path, device: str = "cpu"):
    """Carga un modelo desde checkpoint + detecta el tipo por el path."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {}) or {}

    model_path_str = str(model_path).lower()

    # 1) Neural ODE Physics (IMPORTANTÍSIMO: antes que 'neural_ode')
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
            collocation_strategy=str(config.get("collocation_strategy", config.get("collocation_strategy", "uniform"))),
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
        physics_mode = str(config.get("physics_mode", "complete"))
        use_omega_eff = bool(config.get("use_omega_eff", False))

        hidden_dim = int(config.get("hidden_dim", 128))
        num_layers = int(config.get("num_layers", 4))
        activation = str(config.get("activation", "tanh"))

        g_min = float(config.get("g_min", 0.05))
        conservation_gamma_scale = float(config.get("conservation_gamma_scale", 20.0))

        model_config = PINODESymplecticConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            physics_mode=physics_mode,
            use_omega_eff=use_omega_eff,
            g_min=g_min,
            conservation_gamma_scale=conservation_gamma_scale,
        )
        model = PINODESymplectic(model_config).to(device)
        model_type = "pinode_symplectic"

    # 5) Neural ODE (genérica)
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
# PREDICCIÓN
# =============================================================================
def _choose_time(model_type: str, traj: Dict, device: str):
    """
    Decide si usar t (raw) o t_norm según el tipo de modelo.
    - ODEs tipo neural_ode_physics / pinode_v3 / pinode_v4: usar t (raw)
    - PINNs: usar t_norm
    - neural_ode / pinode / pinode_symplectic: por consistencia puedes usar t_norm,
      pero si tú entrenaste con t raw también, cámbialo aquí a t.
    """
    t = traj["t"].to(device)
    t_norm = traj["t_norm"].to(device)

    if model_type in ["neural_ode_physics", "pinode_v3", "pinode_v4"]:
        return t, "t"
    if model_type in ["pinn", "pinn_fourier"]:
        return t_norm, "t_norm"

    # Por defecto:
    return t_norm, "t_norm"


def predict_trajectory(model, model_type: str, traj: Dict, device: str, debug_once: bool = False) -> np.ndarray:
    """Genera predicción para una trayectoria."""
    with torch.no_grad():
        t_used, time_label = _choose_time(model_type, traj, device)

        params = traj["params"].unsqueeze(0).to(device)          # (1, 3)
        initial_state = traj["states"][0].unsqueeze(0).to(device) # (1, 3)

        if debug_once:
            print(f"   ⏱️  Tiempo usado para {model_type}: {time_label} | "
                  f"t0={traj['t'][0].item():.3f}, tT={traj['t'][-1].item():.3f} | "
                  f"t_normT={traj['t_norm'][-1].item():.3f}")

        # Modelos tipo ODE
        if model_type in ["neural_ode", "neural_ode_physics", "pinode", "pinode_symplectic", "pinode_v3", "pinode_v4"]:
            pred = model(initial_state, t_used, params)  # esperado: (T, batch, 3) o similar
            # En tu código: luego haces squeeze(1) y queda (T,3)
            pred = pred.squeeze(1).cpu().numpy()

        # Modelos tipo PINN
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
# MÉTRICAS
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas completas para una trayectoria."""
    errors = y_pred - y_true
    sq_errors = errors ** 2

    mse = float(np.mean(sq_errors))
    rmse = float(np.sqrt(mse))

    mse_u = float(np.mean(sq_errors[:, 0]))
    mse_v = float(np.mean(sq_errors[:, 1]))
    mse_w = float(np.mean(sq_errors[:, 2]))

    mae = float(np.mean(np.abs(errors)))

    y_true_norm = np.linalg.norm(y_true, axis=1, keepdims=True)
    y_true_norm = np.maximum(y_true_norm, 1e-8)
    rel_error = float(np.mean(np.linalg.norm(errors, axis=1) / y_true_norm.squeeze()))

    ss_res = float(np.sum(sq_errors))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)
    norm_error = float(np.mean(np.abs(norm_pred - norm_true)))
    norm_violation = float(np.mean(np.maximum(0, norm_pred - 1.0)))
    max_norm = float(np.max(norm_pred))

    final_error = float(np.linalg.norm(y_pred[-1] - y_true[-1]))

    return {
        "mse": mse,
        "rmse": rmse,
        "mse_u": mse_u,
        "mse_v": mse_v,
        "mse_w": mse_w,
        "mae": mae,
        "rel_error": rel_error,
        "r2": r2,
        "norm_error": norm_error,
        "norm_violation": norm_violation,
        "max_norm": max_norm,
        "final_error": final_error,
    }


# =============================================================================
# EVALUACIÓN PRINCIPAL
# =============================================================================
def evaluate_model(model, model_type: str, trajectories: List[Dict], device: str, verbose: bool = True) -> pd.DataFrame:
    """Evalúa modelo en todas las trayectorias."""
    all_metrics = []
    iterator = tqdm(trajectories, desc="Evaluando") if verbose else trajectories

    debug_once = True
    for traj in iterator:
        y_pred = predict_trajectory(model, model_type, traj, device, debug_once=debug_once)
        debug_once = False

        y_true = traj["states"].numpy()
        metrics = compute_metrics(y_true, y_pred)

        metrics["trajectory_id"] = traj["id"]
        metrics["Omega"] = traj["Omega"]
        metrics["Delta"] = traj["Delta"]
        metrics["gamma"] = traj["gamma"]
        all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)


def print_metrics_summary(df: pd.DataFrame, title: str = "RESUMEN"):
    print(f"\n{'═' * 60}")
    print(f" 📊 {title}")
    print(f"{'═' * 60}")
    print(f" Trayectorias: {len(df)}\n")
    print(f" {'Métrica':<20} {'Media':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f" {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    metrics_to_show = ["mse", "rmse", "mae", "rel_error", "r2", "norm_error", "final_error"]

    for metric in metrics_to_show:
        if metric not in df.columns:
            continue
        mean = df[metric].mean()
        std = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()

        if metric == "rel_error":
            print(f" {metric:<20} {mean*100:>11.2f}% {std*100:>11.2f}% {min_val*100:>11.2f}% {max_val*100:>11.2f}%")
        elif metric == "r2":
            print(f" {metric:<20} {mean:>12.4f} {std:>12.4f} {min_val:>12.4f} {max_val:>12.4f}")
        else:
            print(f" {metric:<20} {mean:>12.2e} {std:>12.2e} {min_val:>12.2e} {max_val:>12.2e}")

    print(f"{'═' * 60}")


def print_comparison_table(results: Dict[str, pd.DataFrame]):
    print(f"\n{'═' * 80}")
    print(f" 📊 COMPARACIÓN γ=0 vs γ>0")
    print(f"{'═' * 80}")

    print(f"\n {'Métrica':<15} │ {'γ=0 (Mean)':>14} │ {'γ>0 (Mean)':>14} │ {'Ratio':>10} │ {'Mejor':>8}")
    print(f" {'-'*15}─┼─{'-'*14}─┼─{'-'*14}─┼─{'-'*10}─┼─{'-'*8}")

    metrics = ["mse", "rmse", "rel_error", "norm_error", "final_error"]

    for metric in metrics:
        if "gamma_0" not in results or "gamma_pos" not in results:
            continue

        df0 = results["gamma_0"]
        dfp = results["gamma_pos"]
        if len(df0) == 0 or len(dfp) == 0:
            continue

        v0 = float(df0[metric].mean())
        vp = float(dfp[metric].mean())
        ratio = (v0 / vp) if vp != 0 else np.nan
        better = "γ=0" if v0 < vp else "γ>0"

        if metric == "rel_error":
            print(f" {metric:<15} │ {v0*100:>13.2f}% │ {vp*100:>13.2f}% │ {ratio:>10.2f} │ {better:>8}")
        else:
            print(f" {metric:<15} │ {v0:>14.2e} │ {vp:>14.2e} │ {ratio:>10.2f} │ {better:>8}")

    print(f"{'═' * 80}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluación completa de modelos")
    parser.add_argument("--model-path", type=str, required=True, help="Path al modelo .pt")
    parser.add_argument("--data-dir", type=str, required=True, help="Directorio de datos")
    parser.add_argument("--test-set", type=str, default="test_id",
                        choices=["test_id", "test_resonance", "val"],
                        help="Conjunto de test (default: test_id)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directorio de salida (default: junto al modelo)")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo")
    parser.add_argument("--save-individual", action="store_true",
                        help="Guardar métricas individuales por grupo")
    parser.add_argument("--max-traj", type=int, default=None,
                        help="Máximo de trayectorias a evaluar")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Header
    print("=" * 70)
    print(" 🔬 EVALUACIÓN DE MODELO")
    print("=" * 70)
    print(f" 🧠 Modelo: {model_path.parent.name}")
    print(f" 📁 Datos: {data_dir}")
    print(f" 🧪 Test set: {args.test_set}")
    print(f" 📂 Output: {output_dir}")
    print("=" * 70)

    # Cargar modelo
    print("\n🔄 Cargando modelo...")
    model, model_type, config, checkpoint = load_model(model_path, args.device)

    # Mostrar “modo físico” correcto según el modelo
    physics_target = config.get("physics_target", None)
    physics_mode = config.get("physics_mode", None)

    print(f"   Tipo: {model_type}")
    if physics_target is not None:
        print(f"   Physics target: {physics_target}")
    elif physics_mode is not None:
        print(f"   Physics mode: {physics_mode}")

    if "val_mse" in checkpoint:
        print(f"   Val MSE (train): {checkpoint.get('val_mse', 'N/A')}")
    elif "val_loss" in checkpoint:
        print(f"   Val loss (train): {checkpoint.get('val_loss', 'N/A')}")
    elif "val_traj_mse" in checkpoint:
        print(f"   Val traj MSE (train): {checkpoint.get('val_traj_mse', 'N/A')}")
    elif "val_deriv_mse" in checkpoint:
        print(f"   Val deriv MSE (train): {checkpoint.get('val_deriv_mse', 'N/A')}")

    # Cargar datos
    print("\n🔄 Cargando datos...")
    trajectories = load_test_data(data_dir, args.test_set)

    if args.max_traj:
        trajectories = trajectories[:args.max_traj]

    print(f"   Total trayectorias: {len(trajectories)}")
    print(f"   Ejemplo t: [{trajectories[0]['t'][0].item():.3f}, {trajectories[0]['t'][-1].item():.3f}]")
    print(f"   Ejemplo t_norm: [{trajectories[0]['t_norm'][0].item():.3f}, {trajectories[0]['t_norm'][-1].item():.3f}]")

    # Separar por gamma
    traj_gamma_0 = [t for t in trajectories if t["gamma"] <= 1e-6]
    traj_gamma_pos = [t for t in trajectories if t["gamma"] > 1e-6]

    print(f"   γ = 0: {len(traj_gamma_0)} trayectorias")
    print(f"   γ > 0: {len(traj_gamma_pos)} trayectorias")

    results: Dict[str, pd.DataFrame] = {}

    # Todas
    print("\n" + "=" * 70)
    print(" 📊 EVALUANDO TODAS LAS TRAYECTORIAS")
    print("=" * 70)
    df_all = evaluate_model(model, model_type, trajectories, args.device, verbose=True)
    results["all"] = df_all
    print_metrics_summary(df_all, "TODAS LAS TRAYECTORIAS")

    # Gamma 0
    if len(traj_gamma_0) > 0:
        print("\n" + "=" * 70)
        print(" 📊 EVALUANDO γ = 0 (sin disipación)")
        print("=" * 70)
        df_gamma_0 = evaluate_model(model, model_type, traj_gamma_0, args.device, verbose=True)
        results["gamma_0"] = df_gamma_0
        print_metrics_summary(df_gamma_0, "γ = 0 (SIN DISIPACIÓN)")

    # Gamma > 0
    if len(traj_gamma_pos) > 0:
        print("\n" + "=" * 70)
        print(" 📊 EVALUANDO γ > 0 (con disipación)")
        print("=" * 70)
        df_gamma_pos = evaluate_model(model, model_type, traj_gamma_pos, args.device, verbose=True)
        results["gamma_pos"] = df_gamma_pos
        print_metrics_summary(df_gamma_pos, "γ > 0 (CON DISIPACIÓN)")

    # Comparación
    if len(traj_gamma_0) > 0 and len(traj_gamma_pos) > 0:
        print_comparison_table(results)

    # Guardar resultados
    print("\n💾 Guardando resultados...")

    summary = {
        "model_path": str(model_path),
        "model_type": model_type,
        "physics_target": physics_target if physics_target is not None else None,
        "physics_mode": physics_mode if physics_mode is not None else None,
        "test_set": args.test_set,
        "n_trajectories_total": len(trajectories),
        "n_trajectories_gamma_0": len(traj_gamma_0),
        "n_trajectories_gamma_pos": len(traj_gamma_pos),
    }

    for group_name, df in results.items():
        for metric in ["mse", "rmse", "rel_error", "r2", "norm_error", "final_error"]:
            if metric in df.columns:
                summary[f"{group_name}_{metric}_mean"] = float(df[metric].mean())
                summary[f"{group_name}_{metric}_std"] = float(df[metric].std())

    summary_path = output_dir / f"evaluation_{args.test_set}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"   📄 Resumen: {summary_path}")

    # CSV consolidado
    csv_all_path = output_dir / f"metrics_{args.test_set}_all.csv"
    df_all.to_csv(csv_all_path, index=False)
    print(f"   📄 Métricas completas: {csv_all_path}")

    # CSVs por grupo
    if args.save_individual:
        for group_name, df in results.items():
            csv_path = output_dir / f"metrics_{args.test_set}_{group_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"   📄 Métricas {group_name}: {csv_path}")

    print("\n" + "=" * 70)
    print(" ✅ EVALUACIÓN COMPLETADA")
    print("=" * 70)
    print(f"\n 📊 RESUMEN RÁPIDO:")
    print(f"    MSE total:     {df_all['mse'].mean():.4e} ± {df_all['mse'].std():.4e}")
    print(f"    Error rel:     {df_all['rel_error'].mean()*100:.2f}% ± {df_all['rel_error'].std()*100:.2f}%")
    print(f"    R² medio:      {df_all['r2'].mean():.4f}")

    if "gamma_0" in results and "gamma_pos" in results:
        print(f"\n    MSE γ=0:       {results['gamma_0']['mse'].mean():.4e}")
        print(f"    MSE γ>0:       {results['gamma_pos']['mse'].mean():.4e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
