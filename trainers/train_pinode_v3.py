#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trainer PINODE v3 - Con lambdas configurables y diagnóstico de escalas
=======================================================================

ESTRATEGIA:
1. Física incompleta (sin detuning) como base estable
2. Red aprende solo las correcciones (términos de Δ)
3. Lambdas configurables (fijo, auto-escalado, o desactivado)
4. Gate FIJO para estabilidad
5. Diagnóstico de escalas para entender el balance de pérdidas

Uso:
    # Sin regularización (baseline)
    python train_pinode_v3_optimized.py --data-dir ../data/datasets --lambda-cons 0.0
    
    # Lambda fijo pequeño
    python train_pinode_v3_optimized.py --data-dir ../data/datasets --lambda-cons 0.01
    
    # Lambda auto-escalado (RECOMENDADO para experimentar)
    python train_pinode_v3_optimized.py --data-dir ../data/datasets --lambda-cons-auto
    
    # Test rápido
    python train_pinode_v3_optimized.py --data-dir ../data/datasets --quick-test --lambda-cons-auto

Autor: Alberto Vidal Fernández
Fecha: 2025
"""

import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Importar modelo
import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Importar desde models/
from models.pinode_v3_optimized import PINODEv3, PINODEv3Config, PhysicsOnlyBaseline


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATASET
# =============================================================================

class TrajectoryDataset(Dataset):
    """Dataset de trayectorias para PINODE"""
    
    def __init__(self, data_dir: Path, split: str = "train", 
                 max_trajectories: int = None):
        
        if split == "train":
            df = pd.read_csv(data_dir / "train" / "train.csv")
        elif split == "val":
            df = pd.read_csv(data_dir / "val" / "val.csv")
        elif split == "test":
            df = pd.read_csv(data_dir / "test" / "test_id.csv")
        else:
            raise ValueError(f"Split desconocido: {split}")
        
        self.trajectories = []
        
        grouped = df.groupby("trajectory_id")
        count = 0
        
        for traj_id, group in grouped:
            if max_trajectories and count >= max_trajectories:
                break
            
            group = group.sort_values("t")
            
            t = torch.tensor(group["t"].values, dtype=torch.float32)
            states = torch.tensor(
                group[["u", "v", "w"]].values, 
                dtype=torch.float32
            )
            params = torch.tensor([
                group["Omega_R"].iloc[0],
                group["detuning"].iloc[0],
                group["gamma"].iloc[0]
            ], dtype=torch.float32)
            
            self.trajectories.append({
                "t_span": t,
                "states": states,
                "params": params,
                "initial_state": states[0],
            })
            count += 1
        
        print(f"   Cargadas {len(self.trajectories)} trayectorias ({split})")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx]


def collate_fn(batch):
    """Collate para batches de trayectorias"""
    if not batch:
        return None
    
    # Verificar longitudes
    lengths = [b["states"].shape[0] for b in batch]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        for b in batch:
            b["t_span"] = b["t_span"][:min_len]
            b["states"] = b["states"][:min_len]
    
    return {
        "t_span": batch[0]["t_span"],
        "initial_states": torch.stack([b["initial_state"] for b in batch]),
        "targets": torch.stack([b["states"] for b in batch]),
        "params": torch.stack([b["params"] for b in batch]),
    }


# =============================================================================
# EVALUACIÓN DE BASELINE
# =============================================================================

def evaluate_baseline(val_loader, device) -> float:
    """
    Evalúa el error de la física incompleta sola.
    Esto es el "piso" que PINODE debe superar.
    """
    baseline = PhysicsOnlyBaseline()
    
    mse_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            
            t_span = batch["t_span"].to(device)
            y0 = batch["initial_states"].to(device)
            params = batch["params"].to(device)
            targets = batch["targets"].to(device)
            
            pred = baseline(y0, t_span, params).permute(1, 0, 2)
            mse = (pred - targets).pow(2).mean()
            mse_list.append(mse.item())
    
    return np.mean(mse_list)


# =============================================================================
# DIAGNÓSTICO DE ESCALAS
# =============================================================================

def print_scale_diagnostics(losses: dict, epoch: int, batch_idx: int):
    """Imprime diagnóstico de escalas de las pérdidas"""
    if 'diagnostics' not in losses:
        return
    
    d = losses['diagnostics']
    print(f"\n  📊 [ESCALAS Epoch {epoch}, Batch {batch_idx}]")
    print(f"     MSE:              {d['mse_scale']:.4e}")
    print(f"     L_cons (raw):     {d['cons_scale']:.4e}")
    print(f"     Ratio MSE/Cons:   {d['ratio_mse_cons']:.2f}")
    print(f"     λ sugerido (10%): {d['lambda_suggested_10pct']:.4e}")
    print(f"     λ sugerido (1%):  {d['lambda_suggested_1pct']:.4e}")
    print(f"     λ efectivo:       {losses['lambda_eff'].item():.4e}")
    print(f"     Cons contribución:{d['cons_contribution']:.4e}")
    print(f"     Cons % del total: {d['cons_pct_of_total']:.1f}%")
    print()


# =============================================================================
# TRAINING
# =============================================================================

def train(model, train_loader, val_loader, config, device, output_dir: Path):
    """Entrenamiento con diagnóstico de escalas"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["scheduler_patience"],
        min_lr=config["lr_min"]
    )
    
    # Evaluar baseline primero
    print("\n📊 Evaluando baseline (física incompleta sola)...")
    baseline_mse = evaluate_baseline(val_loader, device)
    print(f"   Baseline MSE: {baseline_mse:.4e}")
    print(f"   PINODE debe ser MEJOR que esto.\n")
    
    history = []
    scale_history = []  # Para guardar diagnóstico de escalas
    best_val_mse = float("inf")
    best_epoch = 0
    no_improve = 0
    
    start_time = time.time()
    n_batches = len(train_loader)
    
    # Info de configuración de lambda
    lambda_info = ""
    if model.config.lambda_cons_auto:
        lambda_info = f"AUTO (scale={model.config.lambda_cons_scale})"
    elif model.config.lambda_cons > 0:
        lambda_info = f"FIJO ({model.config.lambda_cons})"
    else:
        lambda_info = "DESACTIVADO (0.0)"
    
    print("=" * 70)
    print("🎯 ENTRENAMIENTO PINODE v3 (con diagnóstico de escalas)")
    print("=" * 70)
    print(f"   Gate fijo: {model.gate:.2f}")
    print(f"   Lambda conservation: {lambda_info}")
    print("=" * 70 + "\n")
    
    for epoch in range(1, config["epochs"] + 1):
        # ----- TRAIN -----
        model.train()
        train_losses = []
        train_mse_only = []
        train_cons_only = []
        train_lambda_eff = []
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            t_span = batch["t_span"].to(device)
            y0 = batch["initial_states"].to(device)
            params = batch["params"].to(device)
            targets = batch["targets"].to(device)
            
            optimizer.zero_grad()
            
            pred = model(y0, t_span, params).permute(1, 0, 2)
            
            # Calcular loss con diagnóstico en primeras épocas
            return_diag = (epoch <= config["diag_epochs"] and batch_idx == 0)
            losses = model.compute_loss(pred, targets, params, return_diagnostics=return_diag)
            
            loss = losses["total"]
            
            # Imprimir diagnóstico de escalas
            if return_diag:
                print_scale_diagnostics(losses, epoch, batch_idx)
                
                # Guardar para análisis posterior
                if 'diagnostics' in losses:
                    scale_history.append({
                        'epoch': epoch,
                        'batch': batch_idx,
                        **losses['diagnostics']
                    })
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config["grad_clip"]
            )
            
            if torch.isnan(grad_norm):
                continue
            
            optimizer.step()
            
            train_losses.append(loss.item())
            train_mse_only.append(losses["mse"].item())
            train_cons_only.append(losses["conservation"].item())
            train_lambda_eff.append(losses["lambda_eff"].item())
            
            # Progreso
            if (batch_idx + 1) % config["log_every"] == 0:
                lambda_str = f"λ={losses['lambda_eff'].item():.2e}" if model.config.lambda_cons > 0 or model.config.lambda_cons_auto else ""
                print(f"\r  Epoch {epoch:3d} | Batch {batch_idx+1:3d}/{n_batches} | "
                      f"MSE: {losses['mse'].item():.4e} | "
                      f"Cons: {losses['conservation'].item():.4e} {lambda_str}", 
                      end="", flush=True)
        
        print()
        
        avg_train = np.mean(train_losses) if train_losses else float("nan")
        avg_train_mse = np.mean(train_mse_only) if train_mse_only else float("nan")
        avg_train_cons = np.mean(train_cons_only) if train_cons_only else float("nan")
        avg_lambda = np.mean(train_lambda_eff) if train_lambda_eff else 0.0
        
        # ----- VALIDATION -----
        if epoch % config["val_freq"] == 0 or epoch == 1:
            model.eval()
            val_losses = []
            val_cons = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    
                    t_span = batch["t_span"].to(device)
                    y0 = batch["initial_states"].to(device)
                    params = batch["params"].to(device)
                    targets = batch["targets"].to(device)
                    
                    pred = model(y0, t_span, params).permute(1, 0, 2)
                    losses = model.compute_loss(pred, targets, params)
                    val_losses.append(losses["mse"].item())
                    val_cons.append(losses["conservation"].item())
            
            avg_val = np.mean(val_losses) if val_losses else float("inf")
            avg_val_cons = np.mean(val_cons) if val_cons else float("nan")
            
            scheduler.step(avg_val)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = (time.time() - start_time) / 60.0
            
            # Calcular mejora vs baseline
            improvement_vs_baseline = (baseline_mse - avg_val) / baseline_mse * 100
            
            print(f"Epoch {epoch:4d}/{config['epochs']} | "
                  f"Train MSE: {avg_train_mse:.4e} | Val MSE: {avg_val:.4e} | "
                  f"Val Cons: {avg_val_cons:.4e} | "
                  f"vs Baseline: {improvement_vs_baseline:+.1f}% | "
                  f"LR: {lr:.2e} | {elapsed:.1f}m")
            
            history.append({
                "epoch": epoch,
                "train_total": avg_train,
                "train_mse": avg_train_mse,
                "train_cons": avg_train_cons,
                "val_mse": avg_val,
                "val_cons": avg_val_cons,
                "baseline_mse": baseline_mse,
                "improvement_vs_baseline": improvement_vs_baseline,
                "lambda_eff_avg": avg_lambda,
                "lr": lr,
            })
            
            # Guardar mejor modelo
            if avg_val < best_val_mse:
                best_val_mse = avg_val
                best_epoch = epoch
                no_improve = 0
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mse": best_val_mse,
                    "val_cons": avg_val_cons,
                    "baseline_mse": baseline_mse,
                    "improvement_vs_baseline": improvement_vs_baseline,
                    "config": model.get_config(),
                }, output_dir / "best_model.pt")
                
                status = "✅ MEJOR" if avg_val < baseline_mse else "⬆️ mejor (pero aún > baseline)"
                print(f"  └─ {status}")
            else:
                no_improve += config["val_freq"]
                if no_improve >= config["early_stopping"]:
                    print(f"\n⏸️  Early stopping en época {epoch}")
                    break
    
    # Guardar historia
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "history.csv", index=False)
    
    # Guardar diagnóstico de escalas
    if scale_history:
        scale_df = pd.DataFrame(scale_history)
        scale_df.to_csv(output_dir / "scale_diagnostics.csv", index=False)
    
    # Metadata
    total_time = time.time() - start_time
    best_improvement = (baseline_mse - best_val_mse) / baseline_mse * 100
    
    meta = {
        "model": "pinode_v3_optimized",
        "best_val_mse": best_val_mse,
        "baseline_mse": baseline_mse,
        "improvement_vs_baseline": best_improvement,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "total_time_minutes": total_time / 60.0,
        "train_config": config,
        "model_config": model.get_config(),
    }
    
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return best_val_mse, baseline_mse


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PINODE v3 con lambdas configurables y diagnóstico de escalas"
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gate", type=float, default=0.5,
                        help="Gate fijo (0=solo física, 1=solo neural)")
    
    # === NUEVO: Argumentos de lambda ===
    parser.add_argument("--lambda-cons", type=float, default=0.0,
                        help="Lambda fijo para conservation loss (0=desactivado)")
    parser.add_argument("--lambda-cons-auto", action="store_true",
                        help="Auto-escalar lambda_cons para que sea ~10%% del MSE")
    parser.add_argument("--lambda-cons-scale", type=float, default=0.1,
                        help="Fracción del MSE cuando lambda-cons-auto=True")
    
    parser.add_argument("--diag-epochs", type=int, default=5,
                        help="Número de épocas para imprimir diagnóstico de escalas")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quick-test", action="store_true")
    
    args = parser.parse_args()
    
    config = {
        "epochs": 10 if args.quick_test else args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_min": 1e-7,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "scheduler_patience": 20,
        "early_stopping": 50,
        "val_freq": 5,
        "log_every": 10,
        "diag_epochs": args.diag_epochs,
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    
    # Output dir con info de lambda
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.lambda_cons_auto:
            lambda_str = f"auto{args.lambda_cons_scale}"
        elif args.lambda_cons > 0:
            lambda_str = f"lc{args.lambda_cons}"
        else:
            lambda_str = "lc0"
        output_dir = Path(f"results/pinode_v3_g{args.gate}_{lambda_str}_seed{args.seed}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Info de lambda
    if args.lambda_cons_auto:
        lambda_info = f"AUTO (scale={args.lambda_cons_scale})"
    elif args.lambda_cons > 0:
        lambda_info = f"FIJO ({args.lambda_cons})"
    else:
        lambda_info = "DESACTIVADO (0.0)"
    
    print("\n" + "=" * 70)
    print(" 🧪 PINODE v3 - CON DIAGNÓSTICO DE ESCALAS")
    print("=" * 70)
    print(f" 📍 Device: {device}")
    print(f" 🎲 Seed: {args.seed}")
    print(f" 🎛️ Gate: {args.gate} (fijo)")
    print(f" ⚖️ Lambda conservation: {lambda_info}")
    print(f" 📁 Output: {output_dir}")
    print(f" 🚀 Quick test: {args.quick_test}")
    print(f" 📊 Diagnóstico escalas: primeras {args.diag_epochs} épocas")
    print("=" * 70)
    
    # Datos
    print("\n📂 Cargando datos...")
    data_dir = Path(args.data_dir)
    max_traj = 100 if args.quick_test else None
    
    train_dataset = TrajectoryDataset(data_dir, "train", max_traj)
    val_dataset = TrajectoryDataset(data_dir, "val", 
                                    max_traj // 5 if max_traj else None)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"],
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"],
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Modelo
    print("\n🧠 Creando modelo...")
    model_config = PINODEv3Config(
        hidden_dim=256,
        num_layers=4,
        activation="silu",
        use_layer_norm=True,
        gate_value=args.gate,
        lambda_cons=args.lambda_cons,
        lambda_cons_auto=args.lambda_cons_auto,
        lambda_cons_scale=args.lambda_cons_scale,
    )
    
    model = PINODEv3(model_config).to(device)
    print(f"   Parámetros: {model.count_parameters():,}")
    print(f"   Gate: {model.gate}")
    print(f"   Lambda config: {lambda_info}")
    
    # Entrenar
    best_mse, baseline_mse = train(
        model, train_loader, val_loader, config, device, output_dir
    )
    
    # Resumen final
    improvement = (baseline_mse - best_mse) / baseline_mse * 100
    
    print("\n" + "=" * 70)
    print(" ✅ COMPLETADO")
    print("=" * 70)
    print(f" 📊 Baseline (física incompleta): {baseline_mse:.6e}")
    print(f" 📊 PINODE v3 (mejor):            {best_mse:.6e}")
    print(f" 📈 Mejora vs baseline:           {improvement:+.1f}%")
    print(f" ⚖️ Lambda conservation:          {lambda_info}")
    
    if best_mse < baseline_mse:
        print(f" 🎉 ¡PINODE SUPERA a la física incompleta!")
    else:
        print(f" ⚠️  PINODE no supera a la física incompleta")
    
    print(f" 📁 Output: {output_dir}")
    print(f" 📊 Ver scale_diagnostics.csv para análisis de escalas")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()