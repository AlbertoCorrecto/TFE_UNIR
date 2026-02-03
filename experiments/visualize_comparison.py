#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualización Comparativa de Modelos
====================================

Genera gráficas comparativas superponiendo predicciones de múltiples modelos
(Neural ODE, PINODE, Neural ODE-P) para trayectorias específicas.

Uso:
    python visualize_comparison.py \
        --neural-ode PATH_NEURAL_ODE \
        --pinode PATH_PINODE \
        --neural-ode-p PATH_NEURAL_ODE_P \
        --data-dir ../data/datasets \
        --traj-ids 264 128 42 \
        --test-set test_id

    # Solo algunos modelos:
    python visualize_comparison.py \
        --neural-ode PATH \
        --neural-ode-p PATH \
        --data-dir ../data/datasets \
        --traj-ids 264

python visualize_comparison.py --neural-ode ../training/results/models/neural_ode_complete_seed42_20251204_073546/best_model.pt --pinode ../training/results/pinode_v3_g0.5_auto0.1_seed42_20260119_113549/best_model.pt --neural-ode-p ../training/results/neural_ode_physics_hybrid_complete_seed42_20260122_203659/best_model.pt  --data-dir ../data/datasets    --traj-ids 264

"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


# =============================================================================
# PATH DEL PROYECTO
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURACIÓN DE ESTILO
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Paleta de colores distintiva
COLORS = {
    'ground_truth': '#2C3E50',      # Gris oscuro elegante
    'neural_ode': '#E74C3C',         # Rojo coral
    'pinode': '#3498DB',             # Azul brillante
    'neural_ode_physics': '#27AE60', # Verde esmeralda
}

LABELS = {
    'ground_truth': 'Ground Truth',
    'neural_ode': 'Neural ODE',
    'pinode': 'PINODE',
    'pinode_v3': 'PINODE v3',
    'pinode_v4': 'PINODE v4',
    'neural_ode_physics': 'Neural ODE-P',
}

LINE_STYLES = {
    'ground_truth': '-',
    'neural_ode': '--',
    'pinode': '-.',
    'pinode_v3': '-.',
    'pinode_v4': ':',
    'neural_ode_physics': ':',
}


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
    print("⚠️ No se pudieron importar algunos modelos base.")

try:
    from models.pinode_v3_optimized import PINODEv3, PINODEv3Config
except Exception:
    PINODE_V3_AVAILABLE = False

try:
    from models.pinode_v4 import PINODEv4, PINODEv4Config
except Exception:
    PINODE_V4_AVAILABLE = False


# =============================================================================
# CARGA DE DATOS
# =============================================================================
def load_test_data(data_dir: Path, dataset: str = "test_id") -> Dict[int, Dict]:
    """Carga trayectorias de test indexadas por ID."""
    dataset_paths = {
        "test_id": data_dir / "test" / "test_id.csv",
        "test_resonance": data_dir / "test" / "test_resonance.csv",
        "test_extrap_params": data_dir / "test" / "test_extrap_params.csv",
        "val": data_dir / "val" / "val.csv",
    }
    
    csv_path = dataset_paths.get(dataset)
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(f"No se encontró dataset: {dataset} en {csv_path}")

    print(f"   📂 Cargando: {csv_path}")
    df = pd.read_csv(csv_path)

    trajectories = {}
    grouped = df.groupby("trajectory_id")

    for traj_id, group in grouped:
        group = group.sort_values("t")
        t = group["t"].values.astype(np.float32)
        traj_id_int = int(traj_id) if str(traj_id).isdigit() else traj_id

        trajectories[traj_id_int] = {
            "id": traj_id_int,
            "t": torch.tensor(t, dtype=torch.float32),
            "t_norm": torch.tensor(t / 30.0, dtype=torch.float32),
            "states": torch.tensor(group[["u", "v", "w"]].values.astype(np.float32)),
            "params": torch.tensor([
                float(group["Omega_R"].iloc[0]),
                float(group["detuning"].iloc[0]),
                float(group["gamma"].iloc[0]),
            ], dtype=torch.float32),
        }

    return trajectories


# =============================================================================
# CARGA DE MODELOS
# =============================================================================
def load_model(model_path: Path, device: str = "cpu") -> Tuple:
    """Carga un modelo y detecta su tipo."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {}) or {}
    model_path_str = str(model_path).lower()

    # Neural ODE Physics (antes que 'neural_ode')
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

    # PINODE v3
    elif "pinode_v3" in model_path_str:
        if not PINODE_V3_AVAILABLE:
            raise ImportError("PINODEv3 no disponible")
        model_config = PINODEv3Config(
            hidden_dim=int(config.get("hidden_dim", 256)),
            num_layers=int(config.get("num_layers", 4)),
            activation=str(config.get("activation", "silu")),
            use_layer_norm=bool(config.get("use_layer_norm", True)),
            gate_value=float(config.get("gate_value", 0.5)),
        )
        model = PINODEv3(model_config).to(device)
        model_type = "pinode_v3"

    # PINODE v4
    elif "pinode_v4" in model_path_str:
        if not PINODE_V4_AVAILABLE:
            raise ImportError("PINODEv4 no disponible")
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

    # Neural ODE
    elif "neural_ode" in model_path_str:
        model = NeuralODE(NeuralODEConfig()).to(device)
        model_type = "neural_ode"

    # PINODE original
    elif "pinode" in model_path_str:
        physics_mode = str(config.get("physics_mode", "complete"))
        model = PINODE(PINODEConfig(physics_mode=physics_mode)).to(device)
        model_type = "pinode"

    else:
        raise ValueError(f"Tipo de modelo no reconocido: {model_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, model_type, config


# =============================================================================
# PREDICCIÓN
# =============================================================================
def _choose_time(model_type: str, traj: Dict, device: str):
    """Selecciona tiempo apropiado según tipo de modelo."""
    t = traj["t"].to(device)
    t_norm = traj["t_norm"].to(device)

    if model_type in ["neural_ode_physics", "pinode_v3", "pinode_v4"]:
        return t
    return t_norm


def predict_single(model, model_type: str, traj: Dict, device: str) -> np.ndarray:
    """Genera predicción para una trayectoria."""
    with torch.no_grad():
        t_used = _choose_time(model_type, traj, device)
        params = traj["params"].unsqueeze(0).to(device)
        initial_state = traj["states"][0].unsqueeze(0).to(device)

        pred = model(initial_state, t_used, params)
        pred = pred.squeeze(1).cpu().numpy()

    return pred


# =============================================================================
# MÉTRICAS
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de error."""
    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    norm_pred = np.linalg.norm(y_pred, axis=1)
    norm_violation = float(np.mean(np.abs(norm_pred - 1.0)))
    
    return {"mse": mse, "r2": r2, "norm_violation": norm_violation}


# =============================================================================
# VISUALIZACIÓN COMPARATIVA
# =============================================================================
def plot_comparison(
    t: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    params: np.ndarray,
    traj_id: int,
    save_path: Path,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Genera gráfica comparativa con todos los modelos superpuestos.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    omega, delta, gamma = params
    
    components = [('u', 0), ('v', 1), ('w', 2)]
    
    # Componentes u, v, w
    for idx, (comp_name, comp_idx) in enumerate(components):
        ax = axes[idx // 2, idx % 2] if idx < 2 else axes[1, 0]
        
        # Ground Truth
        ax.plot(t, y_true[:, comp_idx], 
                color=COLORS['ground_truth'], 
                linewidth=2.5, 
                label=LABELS['ground_truth'],
                zorder=10)
        
        # Predicciones de cada modelo
        for model_type, y_pred in predictions.items():
            color = COLORS.get(model_type, '#95A5A6')
            linestyle = LINE_STYLES.get(model_type, '--')
            label = LABELS.get(model_type, model_type)
            
            ax.plot(t, y_pred[:, comp_idx],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=label,
                    alpha=0.9)
        
        ax.set_xlabel('Tiempo $t$')
        ax.set_ylabel(f'${comp_name}(t)$')
        ax.set_title(f'Componente ${comp_name}$', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Panel de Norma de Bloch
    ax = axes[1, 1]
    
    norm_true = np.linalg.norm(y_true, axis=1)
    ax.plot(t, norm_true, 
            color=COLORS['ground_truth'], 
            linewidth=2.5, 
            label=LABELS['ground_truth'],
            zorder=10)
    
    for model_type, y_pred in predictions.items():
        norm_pred = np.linalg.norm(y_pred, axis=1)
        color = COLORS.get(model_type, '#95A5A6')
        linestyle = LINE_STYLES.get(model_type, '--')
        label = LABELS.get(model_type, model_type)
        
        ax.plot(t, norm_pred,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
                alpha=0.9)
    
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Tiempo $t$')
    ax.set_ylabel(r'$\|\mathbf{B}\|$')
    ax.set_title('Norma del Vector de Bloch', fontweight='bold')
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3)
    
    # Leyenda unificada
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.02),
               ncol=len(predictions) + 1,
               frameon=True,
               fancybox=True,
               shadow=True)
    
    # Título con parámetros y métricas
    metrics_str = []
    for model_type, y_pred in predictions.items():
        m = compute_metrics(y_true, y_pred)
        label = LABELS.get(model_type, model_type)
        metrics_str.append(f"{label}: MSE={m['mse']:.2e}, R²={m['r2']:.4f}")
    
    regime = "Conservativo" if gamma == 0 else f"Disipativo (γ={gamma:.3f})"
    
    title = (f"Comparación de Modelos — Trayectoria {traj_id}\n"
             f"$\\Omega_R={omega:.2f}$, $\\Delta={delta:.2f}$ — Régimen: {regime}")
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Subtítulo con métricas
    subtitle = " | ".join(metrics_str)
    fig.text(0.5, 0.93, subtitle, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.91])
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return {model_type: compute_metrics(y_true, y_pred) 
            for model_type, y_pred in predictions.items()}


def plot_error_comparison(
    t: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    params: np.ndarray,
    traj_id: int,
    save_path: Path,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Genera gráfica de errores absolutos por componente.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    omega, delta, gamma = params
    
    components = [('u', 0), ('v', 1), ('w', 2)]
    
    for idx, (comp_name, comp_idx) in enumerate(components):
        ax = axes[idx // 2, idx % 2] if idx < 2 else axes[1, 0]
        
        for model_type, y_pred in predictions.items():
            error = np.abs(y_pred[:, comp_idx] - y_true[:, comp_idx])
            color = COLORS.get(model_type, '#95A5A6')
            label = LABELS.get(model_type, model_type)
            
            ax.semilogy(t, error + 1e-10,  # Evitar log(0)
                        color=color,
                        linewidth=1.5,
                        label=label,
                        alpha=0.85)
        
        ax.set_xlabel('Tiempo $t$')
        ax.set_ylabel(f'$|{comp_name}_{{pred}} - {comp_name}_{{true}}|$')
        ax.set_title(f'Error Absoluto en ${comp_name}$', fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='upper left', fontsize=9)
    
    # Error total (norma)
    ax = axes[1, 1]
    for model_type, y_pred in predictions.items():
        error_norm = np.linalg.norm(y_pred - y_true, axis=1)
        color = COLORS.get(model_type, '#95A5A6')
        label = LABELS.get(model_type, model_type)
        
        ax.semilogy(t, error_norm + 1e-10,
                    color=color,
                    linewidth=1.5,
                    label=label,
                    alpha=0.85)
    
    ax.set_xlabel('Tiempo $t$')
    ax.set_ylabel(r'$\|\mathbf{y}_{pred} - \mathbf{y}_{true}\|$')
    ax.set_title('Error Total (Norma L2)', fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=9)
    
    regime = "Conservativo" if gamma == 0 else f"Disipativo (γ={gamma:.3f})"
    title = (f"Análisis de Errores — Trayectoria {traj_id}\n"
             f"$\\Omega_R={omega:.2f}$, $\\Delta={delta:.2f}$ — Régimen: {regime}")
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualización comparativa de modelos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
    python visualize_comparison.py \\
        --neural-ode path/to/neural_ode/best_model.pt \\
        --pinode path/to/pinode_v3/best_model.pt \\
        --neural-ode-p path/to/neural_ode_physics/best_model.pt \\
        --data-dir ../data/datasets \\
        --traj-ids 264 128 42
        """
    )
    
    # Modelos (todos opcionales)
    parser.add_argument("--neural-ode", type=str, default=None,
                        help="Path al modelo Neural ODE")
    parser.add_argument("--pinode", type=str, default=None,
                        help="Path al modelo PINODE (v3 o v4)")
    parser.add_argument("--neural-ode-p", type=str, default=None,
                        help="Path al modelo Neural ODE-P (neural_ode_physics)")
    
    # Datos
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directorio de datos")
    parser.add_argument("--test-set", type=str, default="test_id",
                        choices=["test_id", "test_resonance", "val", "test_extrap_params"],
                        help="Conjunto de test")
    
    # Trayectorias
    parser.add_argument("--traj-ids", type=int, nargs='+', required=True,
                        help="IDs de trayectorias a visualizar (ej: 264 128 42)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directorio de salida (default: ./comparison_plots)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-error-plots", action="store_true",
                        help="No generar gráficas de error")
    
    args = parser.parse_args()
    
    # Validar que al menos hay un modelo
    model_paths = {
        'neural_ode': args.neural_ode,
        'pinode': args.pinode,
        'neural_ode_physics': args.neural_ode_p,
    }
    model_paths = {k: v for k, v in model_paths.items() if v is not None}
    
    if not model_paths:
        print("❌ Error: Debe especificar al menos un modelo")
        parser.print_help()
        sys.exit(1)
    
    # Directorios
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("./comparison_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(" 📊 VISUALIZACIÓN COMPARATIVA DE MODELOS")
    print("=" * 70)
    print(f" 📁 Datos: {data_dir}")
    print(f" 🧪 Test set: {args.test_set}")
    print(f" 🎯 Trayectorias: {args.traj_ids}")
    print(f" 📂 Output: {output_dir}")
    print("-" * 70)
    
    # Cargar modelos
    print("\n🔄 Cargando modelos...")
    models = {}
    for model_key, model_path in model_paths.items():
        path = Path(model_path)
        print(f"   • {model_key}: {path.name}")
        model, model_type, config = load_model(path, args.device)
        models[model_type] = model
        
        # Actualizar key si es una variante (v3, v4)
        if model_type != model_key:
            print(f"     → Detectado como: {model_type}")
    
    print(f"\n   ✅ Modelos cargados: {list(models.keys())}")
    
    # Cargar datos
    print("\n🔄 Cargando datos de test...")
    trajectories = load_test_data(data_dir, args.test_set)
    print(f"   Trayectorias disponibles: {len(trajectories)}")
    
    # Verificar IDs solicitados
    missing_ids = [tid for tid in args.traj_ids if tid not in trajectories]
    if missing_ids:
        print(f"   ⚠️ IDs no encontrados: {missing_ids}")
        available_sample = list(trajectories.keys())[:20]
        print(f"   Algunos IDs disponibles: {available_sample}")
    
    valid_ids = [tid for tid in args.traj_ids if tid in trajectories]
    if not valid_ids:
        print("❌ Error: Ningún ID válido")
        sys.exit(1)
    
    # Procesar trayectorias
    print(f"\n🎨 Generando visualizaciones para {len(valid_ids)} trayectorias...")
    print("-" * 70)
    
    all_metrics = []
    
    for traj_id in valid_ids:
        traj = trajectories[traj_id]
        params = traj["params"].numpy()
        omega, delta, gamma = params
        
        print(f"\n   📈 Trayectoria {traj_id}: Ω={omega:.2f}, Δ={delta:.2f}, γ={gamma:.3f}")
        
        # Obtener predicciones
        predictions = {}
        for model_type, model in models.items():
            y_pred = predict_single(model, model_type, traj, args.device)
            predictions[model_type] = y_pred
        
        y_true = traj["states"].numpy()
        t = traj["t"].numpy()
        
        # Gráfica principal
        save_path = output_dir / f"comparison_traj_{traj_id:04d}.png"
        metrics = plot_comparison(
            t=t,
            y_true=y_true,
            predictions=predictions,
            params=params,
            traj_id=traj_id,
            save_path=save_path
        )
        print(f"      ✓ Guardada: {save_path.name}")
        
        # Gráfica de errores
        if not args.no_error_plots:
            error_path = output_dir / f"errors_traj_{traj_id:04d}.png"
            plot_error_comparison(
                t=t,
                y_true=y_true,
                predictions=predictions,
                params=params,
                traj_id=traj_id,
                save_path=error_path
            )
            print(f"      ✓ Errores: {error_path.name}")
        
        # Guardar métricas
        for model_type, m in metrics.items():
            all_metrics.append({
                'traj_id': traj_id,
                'model': model_type,
                'Omega': omega,
                'Delta': delta,
                'gamma': gamma,
                **m
            })
    
    # Resumen de métricas
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = output_dir / "comparison_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        print("\n" + "=" * 70)
        print(" 📊 RESUMEN DE MÉTRICAS")
        print("=" * 70)
        
        for model_type in models.keys():
            model_metrics = metrics_df[metrics_df['model'] == model_type]
            if len(model_metrics) > 0:
                label = LABELS.get(model_type, model_type)
                print(f"\n   {label}:")
                print(f"      MSE medio:  {model_metrics['mse'].mean():.4e}")
                print(f"      R² medio:   {model_metrics['r2'].mean():.4f}")
        
        print(f"\n   📄 Métricas guardadas: {metrics_path}")
    
    print("\n" + "=" * 70)
    print(" ✅ COMPLETADO")
    print("=" * 70)
    print(f" 📂 Visualizaciones en: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()