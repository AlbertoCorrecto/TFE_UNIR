#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Entrenamiento Unificado
==================================================

Entrena Neural ODE, PINN o PINODE para el oscilador de Rabi.

IMPORTANTE: Solo se normaliza el tiempo t ∈ [0, 30] → [0, 1].
Los parámetros físicos (Ω, Δ, γ) NO se normalizan porque PINN/PINODE
los usan directamente en las ecuaciones de Bloch.

Uso:
    # Test rápido
    python train.py --model neural_ode --quick-test --data-dir ../data/datasets
    
    # Entrenamiento real
    python train.py --model neural_ode --seed 42 --data-dir ../data/datasets
    python train.py --model pinn --seed 42 --data-dir ../data/datasets
    python train.py --model pinode --seed 42 --physics-mode complete --data-dir ../data/datasets
    python train.py --model pinode --seed 42 --physics-mode incomplete --data-dir ../data/datasets

Autor: Alberto Vidal Fernández
Fecha: 2025
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Añadir path del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import NeuralODE, PINN, PINODE, NeuralODEConfig, PINNConfig, PINODEConfig
from models.pinn import CollocationSampler


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class TrainConfig:
    """Configuración de entrenamiento"""
    # Paths
    data_dir: str = "data/datasets"
    output_dir: str = "results/models"
    
    # Training
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    lr_min: float = 1e-7
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # Scheduler
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 50
    
    # Validation
    val_freq: int = 5
    
    # Data limits
    max_train_trajectories: int = 5000
    max_val_trajectories: int = 1000
    
    # Físico
    t_max: float = 30.0  # Para normalización de t
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# DATASETS
# =============================================================================

class TrajectoryDataset(Dataset):
    """
    Dataset de trayectorias para Neural ODE y PINODE.
    
    Normaliza t ∈ [0, t_max] → [0, 1]
    NO normaliza parámetros físicos (Ω, Δ, γ) - necesarios para ecuaciones.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        t_max: float = 30.0,
        max_trajectories: Optional[int] = None
    ):
        self.trajectories = []
        self.t_max = t_max
        
        # Agrupar por trayectoria
        grouped = df.groupby('trajectory_id')
        
        for traj_id, group in grouped:
            group = group.sort_values('t')
            
            # Tiempo normalizado: [0, t_max] -> [0, 1]
            t_raw = torch.tensor(group['t'].values, dtype=torch.float32)
            t_norm = t_raw / t_max
            
            # Estados (ya en [-1, 1] por física)
            states = torch.tensor(
                group[['u', 'v', 'w']].values, 
                dtype=torch.float32
            )
            
            # Parámetros físicos SIN normalizar
            params = torch.tensor([
                group['Omega_R'].iloc[0],
                group['detuning'].iloc[0],
                group['gamma'].iloc[0]
            ], dtype=torch.float32)
            
            self.trajectories.append({
                't': t_norm,           # Normalizado
                't_raw': t_raw,        # Original (para debug)
                'states': states,
                'params': params,      # NO normalizado
                'initial_state': states[0],
            })
            
            if max_trajectories and len(self.trajectories) >= max_trajectories:
                break
        
        print(f"   Cargadas {len(self.trajectories)} trayectorias")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx]


class PINNDataset(Dataset):
    """
    Dataset para PINN (formato punto a punto).
    
    Normaliza t ∈ [0, t_max] → [0, 1]
    NO normaliza parámetros físicos.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        t_max: float = 30.0,
        max_points: Optional[int] = None
    ):
        if max_points and len(df) > max_points:
            df = df.sample(n=max_points, random_state=42)
        
        self.t_max = t_max
        
        # Input: [t_norm, Omega, Delta, gamma]
        t_norm = df['t'].values / t_max
        self.x = torch.tensor(
            np.column_stack([
                t_norm,
                df['Omega_R'].values,
                df['detuning'].values,
                df['gamma'].values
            ]),
            dtype=torch.float32
        )
        
        # Output: [u, v, w]
        self.y = torch.tensor(
            df[['u', 'v', 'w']].values,
            dtype=torch.float32
        )
        
        print(f"   Cargados {len(self.x):,} puntos")
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def collate_trajectories(batch: List[Dict]) -> Optional[Dict]:
    """Collate function para trayectorias"""
    if not batch:
        return None
    
    # Verificar longitudes
    t_lens = [traj['t'].shape[0] for traj in batch]
    if len(set(t_lens)) > 1:
        common_len = max(set(t_lens), key=t_lens.count)
        batch = [t for t in batch if t['t'].shape[0] == common_len]
    
    if not batch:
        return None
    
    return {
        't_span': batch[0]['t'],
        'initial_states': torch.stack([t['initial_state'] for t in batch]),
        'params': torch.stack([t['params'] for t in batch]),
        'targets': torch.stack([t['states'] for t in batch]),
    }


# =============================================================================
# COLLOCATION SAMPLER (para PINN)
# =============================================================================

class CollocationSamplerNormalized:
    """Genera puntos de colocación con t normalizado"""
    
    def __init__(
        self,
        t_max: float = 30.0,
        omega_range: tuple = (0.5, 5.0),
        delta_range: tuple = (-3.0, 3.0),
        gamma_values: tuple = (0.0, 0.02, 0.05),
        n_points: int = 50000,
        device: str = "cpu"
    ):
        self.t_max = t_max
        self.omega_range = omega_range
        self.delta_range = delta_range
        self.gamma_values = gamma_values
        self.n_points = n_points
        self.device = device
        self._generate()
    
    def _generate(self):
        n = self.n_points
        
        # t normalizado [0, 1]
        t_norm = torch.rand(n)
        
        # Omega log-uniform (NO normalizado)
        log_omega_min = np.log(self.omega_range[0])
        log_omega_max = np.log(self.omega_range[1])
        omega = torch.exp(torch.rand(n) * (log_omega_max - log_omega_min) + log_omega_min)
        
        # Delta uniform (NO normalizado)
        delta = torch.rand(n) * (self.delta_range[1] - self.delta_range[0]) + self.delta_range[0]
        
        # Gamma discreto (NO normalizado)
        gamma_idx = torch.randint(0, len(self.gamma_values), (n,))
        gamma = torch.tensor([self.gamma_values[i] for i in gamma_idx], dtype=torch.float32)
        
        self.points = torch.stack([t_norm, omega, delta, gamma], dim=1).to(self.device)
    
    def refresh(self):
        self._generate()
    
    def get_batch(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, self.n_points, (batch_size,))
        return self.points[idx]


# =============================================================================
# ENTRENAMIENTO: NEURAL ODE
# =============================================================================

def train_neural_ode(
    model: NeuralODE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Entrena Neural ODE"""
    
    # Neural ODE es más sensible - usar LR más bajo
    lr = min(config.lr, 3e-4)  # Máximo 3e-4 para Neural ODE
    print(f"  (Neural ODE: usando LR={lr:.1e})")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.lr_min
    )
    
    mse_loss = nn.MSELoss()
    best_val_loss = float('inf')
    no_improve = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}
    
    start_time = time.time()
    
    n_batches = len(train_loader)
    consecutive_nan = 0  # Contador de épocas con NaN
    
    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        train_losses = []
        epoch_has_nan = False
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            t_span = batch['t_span'].to(device)
            initial_states = batch['initial_states'].to(device)
            params = batch['params'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            pred = model(initial_states, t_span, params)  # (T, batch, 3)
            pred = pred.permute(1, 0, 2)  # (batch, T, 3)
            
            loss = mse_loss(pred, targets)
            
            # Detección de NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  ⚠️  NaN/Inf detectado en batch {batch_idx+1}, saltando...")
                optimizer.zero_grad()
                epoch_has_nan = True
                continue
            
            loss.backward()
            
            # Verificar gradientes
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if torch.isnan(grad_norm) or grad_norm > 100:
                print(f"\n  ⚠️  Gradiente explosivo ({grad_norm:.1f}), saltando batch...")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Progreso cada 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"\r  Epoch {epoch} | Batch {batch_idx+1}/{n_batches} | Loss: {loss.item():.4e}", end="", flush=True)
        
        print()  # Nueva línea al terminar época
        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        
        # Verificar NaN en la época
        if epoch_has_nan or np.isnan(avg_train_loss) or len(train_losses) == 0:
            consecutive_nan += 1
            print(f"  ⚠️  Época con NaN ({consecutive_nan}/5 consecutivas)")
            if consecutive_nan >= 5:
                print(f"\n❌ Demasiados NaN consecutivos. Parando entrenamiento.")
                print(f"   Mejor modelo guardado en época anterior.")
                break
        else:
            consecutive_nan = 0
        
        # Validation
        if epoch % config.val_freq == 0 or epoch == 1:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    
                    t_span = batch['t_span'].to(device)
                    initial_states = batch['initial_states'].to(device)
                    params = batch['params'].to(device)
                    targets = batch['targets'].to(device)
                    
                    pred = model(initial_states, t_span, params)
                    pred = pred.permute(1, 0, 2)
                    
                    val_losses.append(mse_loss(pred, targets).item())
            
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            scheduler.step(avg_val_loss)
            
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch:4d}/{config.epochs} | "
                  f"Train: {avg_train_loss:.4e} | Val: {avg_val_loss:.4e} | "
                  f"LR: {lr:.2e} | Time: {elapsed/60:.1f}m")
            
            history['epoch'].append(epoch)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['lr'].append(lr)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'config': model.get_config(),
                    't_max': config.t_max,
                }, output_dir / 'best_model.pt')
                
                print(f"  └─ ✅ Mejor modelo guardado!")
            else:
                no_improve += config.val_freq
                if no_improve >= config.early_stopping_patience:
                    print(f"\n⏸️  Early stopping en época {epoch}")
                    break
    
    return {
        'best_val_loss': best_val_loss,
        'total_epochs': epoch,
        'total_time_minutes': (time.time() - start_time) / 60,
        'history': history,
    }


# =============================================================================
# ENTRENAMIENTO: PINN
# =============================================================================

def train_pinn(
    model: PINN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Entrena PINN"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.lr_min
    )
    
    # Collocation sampler
    colloc_sampler = CollocationSamplerNormalized(
        t_max=config.t_max, n_points=50000, device=device
    )
    
    # Initial conditions (t=0)
    n_ic = 1000
    omega_ic = torch.exp(torch.rand(n_ic) * (np.log(5.0) - np.log(0.5)) + np.log(0.5))
    delta_ic = torch.rand(n_ic) * 6.0 - 3.0
    gamma_ic = torch.tensor([0.0, 0.02, 0.05])[torch.randint(0, 3, (n_ic,))]
    
    ic_x = torch.stack([torch.zeros(n_ic), omega_ic, delta_ic, gamma_ic], dim=1).to(device)
    ic_y = torch.zeros(n_ic, 3, device=device)
    ic_y[:, 2] = -1.0  # w₀ = -1
    
    best_val_loss = float('inf')
    no_improve = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 
               'data_loss': [], 'phys_loss': [], 'ic_loss': [], 'norm_loss': [], 'lr': []}
    
    start_time = time.time()
    
    n_batches = len(train_loader)
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = {'data': [], 'phys': [], 'ic': [], 'norm': [], 'total': []}
        
        if epoch % 20 == 0:
            colloc_sampler.refresh()
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            x_colloc = colloc_sampler.get_batch(min(4096, len(x_batch)))
            
            ic_idx = torch.randint(0, n_ic, (min(256, len(x_batch)),))
            x_ic_batch = ic_x[ic_idx]
            y_ic_batch = ic_y[ic_idx]
            
            optimizer.zero_grad()
            
            losses = model.compute_losses(x_batch, y_batch, x_colloc, x_ic_batch, y_ic_batch)
            total_loss = model.compute_total_loss(losses, update_balancer=(epoch % 50 == 0))
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            for k, v in losses.items():
                epoch_losses[k].append(v.item())
            epoch_losses['total'].append(total_loss.item())
            
            # Progreso cada 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"\r  Epoch {epoch} | Batch {batch_idx+1}/{n_batches} | Loss: {total_loss.item():.4e}", end="", flush=True)
        
        print()  # Nueva línea
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # Validation
        if epoch % config.val_freq == 0 or epoch == 1:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_pred = model(x_batch)
                    val_losses.append(torch.mean((y_pred - y_batch)**2).item())
            
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            scheduler.step(avg_val_loss)
            
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            weights = model.get_current_weights()
            
            print(f"Epoch {epoch:4d}/{config.epochs} | "
                  f"Train: {avg_losses['total']:.4e} | Val: {avg_val_loss:.4e} | "
                  f"LR: {lr:.2e} | Time: {elapsed/60:.1f}m")
            print(f"  └─ Data: {avg_losses['data']:.3e} | Phys: {avg_losses['phys']:.3e} | "
                  f"IC: {avg_losses['ic']:.3e} | Norm: {avg_losses['norm']:.3e}")
            print(f"  └─ Weights: λ_phys={weights['phys']:.0f}, λ_ic={weights['ic']:.0f}, λ_norm={weights['norm']:.0f}")
            
            history['epoch'].append(epoch)
            history['train_loss'].append(avg_losses['total'])
            history['val_loss'].append(avg_val_loss)
            history['data_loss'].append(avg_losses['data'])
            history['phys_loss'].append(avg_losses['phys'])
            history['ic_loss'].append(avg_losses['ic'])
            history['norm_loss'].append(avg_losses['norm'])
            history['lr'].append(lr)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': best_val_loss,
                    'config': model.get_config(),
                    't_max': config.t_max,
                }, output_dir / 'best_model.pt')
                
                print(f"  └─ ✅ Mejor modelo guardado!")
            else:
                no_improve += config.val_freq
                if no_improve >= config.early_stopping_patience:
                    print(f"\n⏸️  Early stopping en época {epoch}")
                    break
    
    return {
        'best_val_loss': best_val_loss,
        'total_epochs': epoch,
        'total_time_minutes': (time.time() - start_time) / 60,
        'history': history,
    }


# =============================================================================
# ENTRENAMIENTO: PINODE
# =============================================================================

def train_pinode(
    model: PINODE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Entrena PINODE"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.lr_min
    )
    
    best_val_loss = float('inf')
    no_improve = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'gate_mean': [], 'lr': []}
    
    start_time = time.time()
    
    n_batches = len(train_loader)
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        gate_values = []
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            t_span = batch['t_span'].to(device)
            initial_states = batch['initial_states'].to(device)
            params = batch['params'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            pred = model(initial_states, t_span, params)
            pred = pred.permute(1, 0, 2)
            
            losses = model.compute_loss(pred, targets)
            
            # Detección de NaN
            if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                print(f"\n  ⚠️  NaN/Inf detectado en batch {batch_idx+1}, saltando...")
                continue
            
            losses['total'].backward()
            
            # Verificar gradientes
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if torch.isnan(grad_norm) or grad_norm > 100:
                print(f"\n  ⚠️  Gradiente explosivo ({grad_norm:.1f}), saltando batch...")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            train_losses.append(losses['mse'].item())
            gate_values.append(model.gate_mean)
            
            # Progreso cada 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"\r  Epoch {epoch} | Batch {batch_idx+1}/{n_batches} | Loss: {losses['mse'].item():.4e}", end="", flush=True)
        
        print()  # Nueva línea
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_gate = np.mean(gate_values) if gate_values else 0
        
        # Validation
        if epoch % config.val_freq == 0 or epoch == 1:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    
                    t_span = batch['t_span'].to(device)
                    initial_states = batch['initial_states'].to(device)
                    params = batch['params'].to(device)
                    targets = batch['targets'].to(device)
                    
                    pred = model(initial_states, t_span, params)
                    pred = pred.permute(1, 0, 2)
                    
                    val_losses.append(torch.mean((pred - targets)**2).item())
            
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            scheduler.step(avg_val_loss)
            
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch:4d}/{config.epochs} | "
                  f"Train: {avg_train_loss:.4e} | Val: {avg_val_loss:.4e} | "
                  f"Gate: {avg_gate:.3f} | LR: {lr:.2e} | Time: {elapsed/60:.1f}m")
            
            history['epoch'].append(epoch)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['gate_mean'].append(avg_gate)
            history['lr'].append(lr)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': best_val_loss,
                    'config': model.get_config(),
                    'physics_mode': model.config.physics_mode,
                    't_max': config.t_max,
                }, output_dir / 'best_model.pt')
                
                print(f"  └─ ✅ Mejor modelo guardado!")
            else:
                no_improve += config.val_freq
                if no_improve >= config.early_stopping_patience:
                    print(f"\n⏸️  Early stopping en época {epoch}")
                    break
    
    return {
        'best_val_loss': best_val_loss,
        'total_epochs': epoch,
        'total_time_minutes': (time.time() - start_time) / 60,
        'history': history,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos para el oscilador de Rabi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Test rápido (5 épocas, 100 trayectorias)
  python train.py --model neural_ode --quick-test --data-dir ../data/datasets
  
  # Entrenamiento completo
  python train.py --model neural_ode --seed 42 --data-dir ../data/datasets
  python train.py --model pinn --seed 42 --data-dir ../data/datasets
  python train.py --model pinode --seed 42 --physics-mode complete --data-dir ../data/datasets
  python train.py --model pinode --seed 42 --physics-mode incomplete --data-dir ../data/datasets
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['neural_ode', 'pinn', 'pinode'],
                        help='Modelo a entrenar')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria (default: 42)')
    parser.add_argument('--physics-mode', type=str, default='complete',
                        choices=['complete', 'incomplete'],
                        help='Modo de física para PINN/PINODE (default: complete)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Número de épocas (default: 300)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Tamaño de batch (default: 64)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directorio de datos')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directorio de salida (default: results/models)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Modo test rápido (5 épocas, 100 trayectorias)')
    parser.add_argument('--device', type=str, default=None,
                        help='Dispositivo (cuda/cpu, default: auto)')
    
    args = parser.parse_args()
    
    # Configuración
    config = TrainConfig()
    
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Quick test mode
    if args.quick_test:
        config.epochs = 5
        config.max_train_trajectories = 100
        config.max_val_trajectories = 50
        config.val_freq = 1
        config.early_stopping_patience = 100  # Desactivar
    
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"{args.model}_{args.physics_mode}_seed{args.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Header
    print("\n" + "="*70)
    print(f" 🧪 ENTRENAMIENTO: {args.model.upper()}")
    print("="*70)
    print(f" 📍 Device: {device}")
    print(f" 🎲 Seed: {args.seed}")
    print(f" ⚙️  Physics mode: {args.physics_mode}")
    print(f" 📁 Output: {output_dir}")
    print(f" 🚀 Quick test: {args.quick_test}")
    print(f" ⏱️  Epochs: {config.epochs}")
    print(f" 📦 Batch size: {config.batch_size}")
    print("="*70)
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    data_dir = Path(config.data_dir)
    
    df_train = pd.read_csv(data_dir / "train" / "train.csv")
    df_val = pd.read_csv(data_dir / "val" / "val.csv")
    
    print(f"   Total train: {len(df_train):,} puntos")
    print(f"   Total val: {len(df_val):,} puntos")
    
    # Crear modelo y dataloaders
    if args.model == 'neural_ode':
        print("\n🧠 Creando Neural ODE...")
        model = NeuralODE(NeuralODEConfig()).to(device)
        
        train_ds = TrajectoryDataset(df_train, config.t_max, config.max_train_trajectories)
        val_ds = TrajectoryDataset(df_val, config.t_max, config.max_val_trajectories)
        
        train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                  shuffle=True, collate_fn=collate_trajectories)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                shuffle=False, collate_fn=collate_trajectories)
        
        train_fn = train_neural_ode
        
    elif args.model == 'pinn':
        print("\n🔬 Creando PINN...")
        model = PINN(PINNConfig(physics_mode=args.physics_mode)).to(device)
        
        max_points = config.max_train_trajectories * 400
        train_ds = PINNDataset(df_train, config.t_max, max_points)
        val_ds = PINNDataset(df_val, config.t_max, config.max_val_trajectories * 400)
        
        batch_size = 8192 if not args.quick_test else 1024
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        train_fn = train_pinn
        
    elif args.model == 'pinode':
        print("\n⚡ Creando PINODE...")
        model = PINODE(PINODEConfig(physics_mode=args.physics_mode)).to(device)
        
        train_ds = TrajectoryDataset(df_train, config.t_max, config.max_train_trajectories)
        val_ds = TrajectoryDataset(df_val, config.t_max, config.max_val_trajectories)
        
        train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                                  shuffle=True, collate_fn=collate_trajectories)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                                shuffle=False, collate_fn=collate_trajectories)
        
        train_fn = train_pinode
    
    n_params = model.count_parameters()
    print(f"   Parámetros: {n_params:,}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Entrenar
    print(f"\n{'='*70}")
    print("🎯 ENTRENAMIENTO")
    print(f"{'='*70}\n")
    
    results = train_fn(model, train_loader, val_loader, config, device, output_dir)
    
    # Resumen
    print(f"\n{'='*70}")
    print("✅ COMPLETADO")
    print(f"{'='*70}")
    print(f" 📊 Best Val Loss: {results['best_val_loss']:.6e}")
    print(f" 📈 Epochs: {results['total_epochs']}")
    print(f" ⏱️  Tiempo: {results['total_time_minutes']:.1f} minutos")
    print(f" 📁 Output: {output_dir}")
    
    # Guardar metadata
    meta = {
        'model': args.model,
        'physics_mode': args.physics_mode,
        'seed': args.seed,
        'device': device,
        'n_parameters': n_params,
        'config': config.to_dict(),
        'model_config': model.get_config(),
        **results,
    }
    
    with open(output_dir / 'training_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    # Guardar history como CSV
    history_df = pd.DataFrame(results['history'])
    history_df.to_csv(output_dir / 'history.csv', index=False)
    
    print(f"\n📁 Archivos guardados:")
    print(f"   - {output_dir / 'best_model.pt'}")
    print(f"   - {output_dir / 'training_meta.json'}")
    print(f"   - {output_dir / 'history.csv'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()