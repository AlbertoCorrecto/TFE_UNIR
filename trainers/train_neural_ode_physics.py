#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrenamiento de Neural ODE con Loss Física (Neural ODE-P)
===========================================================


Modos de loss:
- 'physics': Solo ||f_neural - f_Bloch||² (sin ver trayectorias)
- 'data': Solo MSE de trayectorias (Neural ODE estándar)
- 'hybrid': Combinación ponderada de ambas

Uso:
    python train_neural_ode_physics.py
    python train_neural_ode_physics.py --loss-mode physics --physics-target complete
    python train_neural_ode_physics.py --loss-mode hybrid --lambda-physics 1.0 --lambda-data 1.0

Autor: Alberto Vidal Fernández
Fecha: 2025
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Path del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.neural_ode_physics import NeuralODEPhysics, NeuralODEPhysicsConfig


# =============================================================================
# ECUACIONES DE BLOCH (TARGETS)
# =============================================================================

def bloch_rhs_complete(y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Ecuaciones de Bloch COMPLETAS"""
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Omega, Delta, gamma = params[:, 0:1], params[:, 1:2], params[:, 2:3]
    
    du = -gamma * u + Delta * v
    dv = -gamma * v - Delta * u - Omega * w
    dw = -2.0 * gamma * (w + 1.0) + Omega * v
    
    return torch.cat([du, dv, dw], dim=-1)


def bloch_rhs_incomplete(y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Ecuaciones de Bloch SIN detuning"""
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Omega, gamma = params[:, 0:1], params[:, 2:3]
    
    du = -gamma * u
    dv = -gamma * v - Omega * w
    dw = -2.0 * gamma * (w + 1.0) + Omega * v
    
    return torch.cat([du, dv, dw], dim=-1)


def bloch_rhs_minimal(y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Ecuaciones de Bloch SOLO disipación"""
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    gamma = params[:, 2:3]
    
    du = -gamma * u
    dv = -gamma * v
    dw = -2.0 * gamma * (w + 1.0)
    
    return torch.cat([du, dv, dw], dim=-1)


PHYSICS_FUNCTIONS = {
    'complete': bloch_rhs_complete,
    'incomplete': bloch_rhs_incomplete,
    'minimal': bloch_rhs_minimal,
}


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def load_data(data_dir: Path, split: str = "train"):
    """Carga datos de entrenamiento/validación"""
    
    csv_path = data_dir / split / f"{split}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    trajectories = []
    grouped = df.groupby('trajectory_id')
    
    for traj_id, group in grouped:
        group = group.sort_values('t')
        
        trajectories.append({
            't': torch.tensor(group['t'].values, dtype=torch.float32),
            'states': torch.tensor(group[['u', 'v', 'w']].values, dtype=torch.float32),
            'params': torch.tensor([
                group['Omega_R'].iloc[0],
                group['detuning'].iloc[0],
                group['gamma'].iloc[0]
            ], dtype=torch.float32),
        })
    
    return trajectories


def create_dataloader(trajectories, batch_size: int, shuffle: bool = True):
    """Crea DataLoader desde trayectorias"""
    
    states = torch.stack([t['states'] for t in trajectories])  # (N, T, 3)
    params = torch.stack([t['params'] for t in trajectories])  # (N, 3)
    t_span = trajectories[0]['t']
    
    dataset = TensorDataset(states, params)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader, t_span


# =============================================================================
# MUESTREO DE PUNTOS DE COLOCACIÓN
# =============================================================================

def sample_collocation_uniform(n_points: int, param_ranges: dict, device: str):
    """
    Muestrea puntos de colocación uniformemente en la esfera de Bloch.
    """
    # Estados en la esfera de Bloch (||y|| <= 1)
    states = []
    while len(states) < n_points:
        candidates = torch.rand(n_points * 2, 3, device=device) * 2 - 1
        norms = torch.norm(candidates, dim=-1)
        valid = candidates[norms <= 1.0]
        states.append(valid)
    
    states = torch.cat(states, dim=0)[:n_points]
    
    # Parámetros aleatorios
    Omega = torch.rand(n_points, 1, device=device) * \
            (param_ranges['Omega'][1] - param_ranges['Omega'][0]) + param_ranges['Omega'][0]
    Delta = torch.rand(n_points, 1, device=device) * \
            (param_ranges['Delta'][1] - param_ranges['Delta'][0]) + param_ranges['Delta'][0]
    gamma = torch.rand(n_points, 1, device=device) * \
            (param_ranges['gamma'][1] - param_ranges['gamma'][0]) + param_ranges['gamma'][0]
    
    params = torch.cat([Omega, Delta, gamma], dim=-1)
    
    return states, params


def sample_collocation_from_trajectories(states_batch: torch.Tensor, 
                                         params_batch: torch.Tensor,
                                         n_points: int):
    """
    Muestrea puntos de colocación a lo largo de trayectorias dadas.
    """
    batch, T, _ = states_batch.shape
    
    # Aplanar
    all_states = states_batch.reshape(-1, 3)
    params_expanded = params_batch.unsqueeze(1).expand(-1, T, -1).reshape(-1, 3)
    
    # Muestrear
    n_total = all_states.shape[0]
    indices = torch.randperm(n_total, device=all_states.device)[:n_points]
    
    return all_states[indices], params_expanded[indices]


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def compute_physics_loss(model, coll_states, coll_params, physics_fn):
    """Calcula loss de residuo físico"""
    f_pred = model.ode_func.forward_with_params(coll_states, coll_params)
    f_true = physics_fn(coll_states, coll_params)
    return ((f_pred - f_true) ** 2).mean()


def compute_data_loss(y_pred, y_true):
    """Calcula MSE de trayectorias"""
    return ((y_pred - y_true) ** 2).mean()


def compute_derivative_metrics(model, coll_states, coll_params, physics_fn):
    """Calcula métricas de derivadas para monitoreo"""
    with torch.no_grad():
        f_pred = model.ode_func.forward_with_params(coll_states, coll_params)
        f_true = physics_fn(coll_states, coll_params)
        
        mse = ((f_pred - f_true) ** 2).mean().item()
        
        # Correlaciones por componente
        correlations = []
        for i in range(3):
            pred_i = f_pred[:, i]
            true_i = f_true[:, i]
            if pred_i.std() > 1e-8 and true_i.std() > 1e-8:
                corr = torch.corrcoef(torch.stack([pred_i, true_i]))[0, 1].item()
            else:
                corr = 0.0
            correlations.append(corr)
        
        return mse, correlations


def train_epoch(model, loader, t_span, optimizer, device, config, physics_fn, param_ranges):
    """Entrena una época"""
    model.train()
    
    total_loss = 0.0
    total_physics = 0.0
    total_data = 0.0
    n_batches = 0
    
    for states, params in loader:
        states = states.to(device)
        params = params.to(device)
        t = t_span.to(device)
        
        optimizer.zero_grad()
        
        loss = torch.tensor(0.0, device=device)
        physics_loss = torch.tensor(0.0, device=device)
        data_loss = torch.tensor(0.0, device=device)
        
        # Loss física
        if config.loss_mode in ['physics', 'hybrid']:
            # Muestrear puntos de colocación
            if config.collocation_source == 'uniform':
                coll_states, coll_params = sample_collocation_uniform(
                    config.n_collocation, param_ranges, device
                )
            else:  # 'trajectory'
                coll_states, coll_params = sample_collocation_from_trajectories(
                    states, params, config.n_collocation
                )
            
            physics_loss = compute_physics_loss(model, coll_states, coll_params, physics_fn)
            loss = loss + config.lambda_physics * physics_loss
        
        # Loss de datos
        if config.loss_mode in ['data', 'hybrid']:
            y0 = states[:, 0, :]
            y_pred = model(y0, t, params)  # (T, batch, 3)
            y_pred = y_pred.permute(1, 0, 2)  # (batch, T, 3)
            
            data_loss = compute_data_loss(y_pred, states)
            loss = loss + config.lambda_data * data_loss
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Acumular
        total_loss += loss.item()
        total_physics += physics_loss.item()
        total_data += data_loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'physics': total_physics / n_batches,
        'data': total_data / n_batches,
    }


def validate(model, loader, t_span, device, config, physics_fn, param_ranges):
    """Valida el modelo"""
    model.eval()
    
    total_traj_mse = 0.0
    total_deriv_mse = 0.0
    all_correlations = []
    n_batches = 0
    
    with torch.no_grad():
        for states, params in loader:
            states = states.to(device)
            params = params.to(device)
            t = t_span.to(device)
            
            # MSE de trayectorias
            y0 = states[:, 0, :]
            y_pred = model(y0, t, params)
            y_pred = y_pred.permute(1, 0, 2)
            
            traj_mse = compute_data_loss(y_pred, states).item()
            total_traj_mse += traj_mse
            
            # MSE de derivadas (en puntos de colocación)
            coll_states, coll_params = sample_collocation_uniform(
                500, param_ranges, device
            )
            deriv_mse, correlations = compute_derivative_metrics(
                model, coll_states, coll_params, physics_fn
            )
            
            total_deriv_mse += deriv_mse
            all_correlations.append(correlations)
            n_batches += 1
    
    mean_corr = np.mean(all_correlations, axis=0)
    
    return {
        'traj_mse': total_traj_mse / n_batches,
        'deriv_mse': total_deriv_mse / n_batches,
        'corr_u': mean_corr[0],
        'corr_v': mean_corr[1],
        'corr_w': mean_corr[2],
        'corr_mean': np.mean(mean_corr),
    }


def train(
    data_dir: Path,
    output_dir: Path,
    config: NeuralODEPhysicsConfig,
    param_ranges: dict,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
    device: str = "cuda",
    seed: int = 42,
):
    """Entrenamiento completo"""
    
    # Reproducibilidad
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    train_traj = load_data(data_dir, "train")
    val_traj = load_data(data_dir, "val")
    
    print(f"   Train: {len(train_traj)} trayectorias")
    print(f"   Val: {len(val_traj)} trayectorias")
    
    train_loader, t_span = create_dataloader(train_traj, batch_size, shuffle=True)
    val_loader, _ = create_dataloader(val_traj, batch_size, shuffle=False)
    
    # Función física objetivo
    physics_fn = PHYSICS_FUNCTIONS[config.physics_target]
    
    # Modelo
    print("\n🧠 Creando modelo...")
    model = NeuralODEPhysics(config).to(device)
    model.set_device(device)
    n_params = model.count_parameters()
    print(f"   Parámetros: {n_params:,}")
    print(f"   Loss mode: {config.loss_mode}")
    print(f"   Physics target: {config.physics_target}")
    print(f"   λ_physics: {config.lambda_physics}")
    print(f"   λ_data: {config.lambda_data}")
    print(f"   N colocación: {config.n_collocation}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print("\n🚀 Iniciando entrenamiento...")
    print("=" * 80)
    
    best_val_metric = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []
    
    # Métrica principal según modo
    metric_key = 'deriv_mse' if config.loss_mode == 'physics' else 'traj_mse'
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, t_span, optimizer, device, 
            config, physics_fn, param_ranges
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, t_span, device, 
            config, physics_fn, param_ranges
        )
        
        # Scheduler
        scheduler.step(val_metrics[metric_key])
        
        # Historia
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_physics': train_metrics['physics'],
            'train_data': train_metrics['data'],
            'val_traj_mse': val_metrics['traj_mse'],
            'val_deriv_mse': val_metrics['deriv_mse'],
            'val_corr_u': val_metrics['corr_u'],
            'val_corr_v': val_metrics['corr_v'],
            'val_corr_w': val_metrics['corr_w'],
            'val_corr_mean': val_metrics['corr_mean'],
            'lr': optimizer.param_groups[0]['lr'],
        })
        
        # Check best
        current_metric = val_metrics[metric_key]
        is_best = current_metric < best_val_metric
        if is_best:
            best_val_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Guardar mejor modelo
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_traj_mse': val_metrics['traj_mse'],
                'val_deriv_mse': val_metrics['deriv_mse'],
                'val_correlations': [val_metrics['corr_u'], val_metrics['corr_v'], val_metrics['corr_w']],
                'config': model.get_config(),
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1
        
        # Print
        marker = " ← BEST" if is_best else ""
        print(f"  Epoch {epoch:3d} | "
              f"Loss: {train_metrics['loss']:.4e} | "
              f"Traj MSE: {val_metrics['traj_mse']:.4e} | "
              f"Deriv MSE: {val_metrics['deriv_mse']:.4e} | "
              f"Corr: {val_metrics['corr_mean']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}{marker}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping en epoch {epoch}")
            break
    
    # Guardar historia
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    
    # Guardar último modelo
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_traj_mse': val_metrics['traj_mse'],
        'val_deriv_mse': val_metrics['deriv_mse'],
        'config': model.get_config(),
    }, output_dir / "last_model.pt")
    
    print("=" * 80)
    print(f"\n✅ Entrenamiento completado!")
    print(f"   Mejor epoch: {best_epoch}")
    print(f"   Mejor {metric_key}: {best_val_metric:.4e}")
    print(f"   Resultados en: {output_dir}")
    
    return best_val_metric, best_epoch


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Neural ODE with Physics Loss")
    
    # Paths
    parser.add_argument('--data-dir', type=str, default='../data/datasets',
                        help='Directorio de datos')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directorio de salida')
    
    # Arquitectura
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--activation', type=str, default='silu')
    parser.add_argument('--use-layer-norm', action='store_true', default=True)
    parser.add_argument('--no-layer-norm', action='store_false', dest='use_layer_norm')
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Modo de loss
    parser.add_argument('--loss-mode', type=str, default='physics',
                        choices=['physics', 'data', 'hybrid'],
                        help='Modo de loss: physics, data, hybrid')
    parser.add_argument('--physics-target', type=str, default='complete',
                        choices=['complete', 'incomplete', 'minimal'],
                        help='Física objetivo a aprender')
    parser.add_argument('--lambda-physics', type=float, default=1.0,
                        help='Peso del loss física')
    parser.add_argument('--lambda-data', type=float, default=1.0,
                        help='Peso del loss datos')
    
    # Colocación
    parser.add_argument('--n-collocation', type=int, default=2000,
                        help='Número de puntos de colocación por batch')
    parser.add_argument('--collocation-source', type=str, default='uniform',
                        choices=['uniform', 'trajectory'],
                        help='Fuente de puntos de colocación')
    
    # Rangos de parámetros
    parser.add_argument('--omega-min', type=float, default=0.5)
    parser.add_argument('--omega-max', type=float, default=5.0)
    parser.add_argument('--delta-min', type=float, default=-2.0)
    parser.add_argument('--delta-max', type=float, default=2.0)
    parser.add_argument('--gamma-min', type=float, default=0.01)
    parser.add_argument('--gamma-max', type=float, default=0.1)
    
    # Entrenamiento
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    
    # ODE Solver
    parser.add_argument('--solver', type=str, default='rk4',
                        choices=['rk4', 'dopri5', 'euler'])
    
    args = parser.parse_args()
    
    # Config del modelo
    config = NeuralODEPhysicsConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        use_layer_norm=args.use_layer_norm,
        dropout=args.dropout,
        solver=args.solver,
        loss_mode=args.loss_mode,
        physics_target=args.physics_target,
        lambda_physics=args.lambda_physics,
        lambda_data=args.lambda_data,
        n_collocation=args.n_collocation,
        collocation_strategy=args.collocation_source,
    )
    
    # Rangos de parámetros
    param_ranges = {
        'Omega': (args.omega_min, args.omega_max),
        'Delta': (args.delta_min, args.delta_max),
        'gamma': (args.gamma_min, args.gamma_max),
    }
    
    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        name = f"neural_ode_physics_{args.loss_mode}_{args.physics_target}_seed{args.seed}_{timestamp}"
        output_dir = Path(__file__).parent / "results" / name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Header
    print("=" * 80)
    print(" 🧪 NEURAL ODE CON LOSS FÍSICA (Neural ODE-P)")
    print("=" * 80)
    print(f" Loss mode: {args.loss_mode}")
    print(f" Physics target: {args.physics_target}")
    print(f" λ_physics: {args.lambda_physics}, λ_data: {args.lambda_data}")
    print(f" N colocación: {args.n_collocation} ({args.collocation_source})")
    print(f" Hidden: {args.hidden_dim}, Layers: {args.num_layers}")
    print(f" Solver: {args.solver}")
    print(f" Output: {output_dir}")
    print("=" * 80)
    
    # Data dir
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / args.data_dir
    
    # Train
    train(
        data_dir=data_dir,
        output_dir=output_dir,
        config=config,
        param_ranges=param_ranges,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()