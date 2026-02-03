#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PINODE v3 Optimizado - Con diagnóstico de escalas y lambdas configurables
==========================================================================


- lambda_cons configurable
- Opción lambda_cons_auto para escalar automáticamente
- Diagnóstico de escalas para debugging

Autor: Alberto Vidal Fernández
Fecha: 2025
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PINODEv3Config:
    """Configuración optimizada con lambdas configurables"""
    
    # Arquitectura
    hidden_dim: int = 256
    num_layers: int = 4
    activation: str = "silu"
    use_layer_norm: bool = True
    dropout: float = 0.0
    
    # Física
    physics_mode: str = "incomplete"  # SIEMPRE incompleto para que tenga sentido
    
    # Gate FIJO - clave para estabilidad
    gate_value: float = 0.5  # Contribución 50% física, 50% neural
    
    # === Lambdas para regularización ===
    lambda_cons: float = 0.0         # Lambda fijo (0 = desactivado)
    lambda_cons_auto: bool = False   # Si True, escala automáticamente
    lambda_cons_scale: float = 0.1   # Fracción del MSE cuando auto=True
    lambda_cons_min: float = 1e-4    # Límite inferior para auto-scaling
    lambda_cons_max: float = 10.0    # Límite superior para auto-scaling
    
    # Dimensiones
    param_dim: int = 3  # [Omega, Delta, gamma]
    state_dim: int = 3  # [u, v, w]


def get_activation(name: str):
    activations = {
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
    }
    return activations.get(name.lower(), nn.SiLU())


class CorrectionNetwork(nn.Module):
    """
    Red que predice las correcciones a la física incompleta.
    
    Input: [u, v, w, Ω, Δ, γ] (6 dims)
    Output: [Δu_corr, Δv_corr, Δw_corr] (3 dims)
    
    La red debería aprender aproximadamente:
        Δu_corr ≈ +Δ*v  (término faltante en du/dt)
        Δv_corr ≈ -Δ*u  (término faltante en dv/dt)
        Δw_corr ≈ 0     (no falta nada en dw/dt)
    """
    
    def __init__(self, config: PINODEv3Config):
        super().__init__()
        self.config = config
        
        input_dim = config.state_dim + config.param_dim  # 6
        
        layers = []
        in_dim = input_dim
        
        for i in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        
        layers.append(nn.Linear(config.hidden_dim, config.state_dim))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización para correcciones pequeñas inicialmente"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Capa de salida con ganancia muy pequeña
        last_linear = None
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.xavier_uniform_(last_linear.weight, gain=0.1)
            nn.init.zeros_(last_linear.bias)
    
    def forward(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, 3) = [u, v, w]
            params: (batch, 3) = [Ω, Δ, γ]
        
        Returns:
            corrections: (batch, 3) = [du_corr, dv_corr, dw_corr]
        """
        x = torch.cat([state, params], dim=-1)
        return self.net(x)


def bloch_rhs_incomplete(y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Ecuaciones de Bloch INCOMPLETAS (sin términos de detuning).
    
    Faltan:
        +Δ*v en du/dt
        -Δ*u en dv/dt
    
    Estos términos los aprenderá la red.
    """
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    
    Omega = params[:, 0:1]
    # Delta = params[:, 1:2]  # NO USADO - la red lo aprende
    gamma = params[:, 2:3]
    
    du = -gamma * u                        # Falta: + Delta * v
    dv = -gamma * v - Omega * w            # Falta: - Delta * u
    dw = -2.0 * gamma * (w + 1.0) + Omega * v
    
    return torch.cat([du, dv, dw], dim=-1)


def bloch_rhs_complete(y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Ecuaciones de Bloch COMPLETAS (para referencia/comparación)"""
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    
    Omega = params[:, 0:1]
    Delta = params[:, 1:2]
    gamma = params[:, 2:3]
    
    du = -gamma * u + Delta * v
    dv = -gamma * v - Delta * u - Omega * w
    dw = -2.0 * gamma * (w + 1.0) + Omega * v
    
    return torch.cat([du, dv, dw], dim=-1)


def rk4_step(y: torch.Tensor, params: torch.Tensor, 
             f_total, dt: float) -> torch.Tensor:
    """Un paso de RK4"""
    k1 = f_total(y, params)
    k2 = f_total(y + 0.5 * dt * k1, params)
    k3 = f_total(y + 0.5 * dt * k2, params)
    k4 = f_total(y + dt * k3, params)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class PINODEv3(nn.Module):
    """
    PINODE v3 Optimizado con lambdas configurables
    
    Combina física incompleta + correcciones neurales:
        dy/dt = f_physics(y, params) + gate * f_neural(y, params)
    
    Integra con RK4 usando los dt reales del t_span.
    """
    
    def __init__(self, config: PINODEv3Config = None):
        super().__init__()
        self.config = config or PINODEv3Config()
        
        # Red de correcciones
        self.correction_net = CorrectionNetwork(self.config)
        
        # Gate fijo (no es parámetro entrenable)
        self.register_buffer('_gate', torch.tensor(self.config.gate_value))
        
        # Para tracking del lambda efectivo
        self._last_lambda_eff = 0.0
    
    @property
    def gate(self) -> float:
        return float(self._gate.item())
    
    @gate.setter
    def gate(self, value: float):
        self._gate.fill_(value)
    
    def f_total(self, y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Campo vectorial total: física incompleta + corrección neural
        
        Args:
            y: (batch, 3) estado actual
            params: (batch, 3) parámetros físicos
        
        Returns:
            dy/dt: (batch, 3)
        """
        # Física incompleta
        f_phys = bloch_rhs_incomplete(y, params)
        
        # Corrección neural
        f_corr = self.correction_net(y, params)
        
        # Combinar con gate
        return f_phys + self._gate * f_corr
    
    def forward(self, y0: torch.Tensor, t_span: torch.Tensor, 
                params: torch.Tensor) -> torch.Tensor:
        """
        Integra la ODE desde y0 a través de t_span.
        
        Args:
            y0: (batch, 3) estado inicial
            t_span: (T,) tiempos de evaluación
            params: (batch, 3) parámetros físicos
        
        Returns:
            trajectory: (T, batch, 3)
        """
        y = y0.clone()
        outputs = [y]
        
        for i in range(len(t_span) - 1):
            dt = (t_span[i+1] - t_span[i]).item()
            y = rk4_step(y, params, self.f_total, dt)
            outputs.append(y)
        
        return torch.stack(outputs, dim=0)  # (T, batch, 3)
    
    def predict_trajectory(self, y0: torch.Tensor, t_span: torch.Tensor,
                          params: torch.Tensor) -> torch.Tensor:
        """Alias que devuelve (batch, T, 3)"""
        traj = self.forward(y0, t_span, params)
        return traj.permute(1, 0, 2)
    
    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                     params: Optional[torch.Tensor] = None,
                     return_diagnostics: bool = False) -> Dict[str, torch.Tensor]:
        """
        Pérdida con conservación opcional y diagnóstico de escalas.
        
        Args:
            y_pred: Predicciones (T, batch, 3) o (batch, T, 3)
            y_true: Ground truth (batch, T, 3)
            params: Parámetros físicos (opcional, no usado actualmente)
            return_diagnostics: Si True, incluye info de escalas para debugging
        
        Returns:
            Dict con 'total', 'mse', 'conservation', 'lambda_eff'
            Si return_diagnostics=True, incluye 'diagnostics' con escalas
        """
        # Asegurar dimensiones correctas
        if y_pred.dim() == 3 and y_pred.shape[0] != y_true.shape[0]:
            # (T, batch, 3) -> (batch, T, 3)
            y_pred = y_pred.permute(1, 0, 2)
        
        device = y_pred.device
        
        # MSE (pérdida principal)
        mse = (y_pred - y_true).pow(2).mean()
        
        # Conservation loss: penaliza si ||B|| > 1
        norm = torch.linalg.norm(y_pred, dim=-1)  # (batch, T)
        cons_raw = torch.relu(norm - 1.0).pow(2).mean()
        
        # Determinar lambda efectivo
        if self.config.lambda_cons_auto and cons_raw.item() > 1e-10:
            # Auto-escalar: que cons sea ~lambda_cons_scale del MSE
            lambda_eff = self.config.lambda_cons_scale * mse.detach() / cons_raw.detach()
            lambda_eff = lambda_eff.clamp(
                self.config.lambda_cons_min, 
                self.config.lambda_cons_max
            )
            lambda_eff_val = lambda_eff.item()
        else:
            lambda_eff = self.config.lambda_cons
            lambda_eff_val = lambda_eff
        
        # Guardar para referencia
        self._last_lambda_eff = lambda_eff_val
        
        # Pérdida total
        if lambda_eff_val > 0:
            if isinstance(lambda_eff, torch.Tensor):
                cons_weighted = lambda_eff * cons_raw
            else:
                cons_weighted = lambda_eff_val * cons_raw
            total = mse + cons_weighted
        else:
            cons_weighted = torch.tensor(0.0, device=device)
            total = mse
        
        result = {
            'total': total,
            'mse': mse,
            'conservation': cons_raw,
            'cons_weighted': cons_weighted,
            'lambda_eff': torch.tensor(lambda_eff_val, device=device),
        }
        
        # Diagnóstico de escalas (útil para debugging)
        if return_diagnostics:
            with torch.no_grad():
                mse_val = mse.item()
                cons_val = cons_raw.item()
                ratio = mse_val / (cons_val + 1e-10)
                
                result['diagnostics'] = {
                    'mse_scale': mse_val,
                    'cons_scale': cons_val,
                    'ratio_mse_cons': ratio,
                    'lambda_suggested_10pct': 0.1 * ratio,
                    'lambda_suggested_1pct': 0.01 * ratio,
                    'cons_contribution': lambda_eff_val * cons_val,
                    'cons_pct_of_total': 100 * lambda_eff_val * cons_val / (mse_val + lambda_eff_val * cons_val + 1e-10),
                }
        
        return result
    
    def get_corrections(self, y: torch.Tensor, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Devuelve las correcciones aprendidas y las esperadas (para análisis).
        
        Returns:
            dict con 'learned' y 'expected' corrections
        """
        u = y[:, 0:1]
        v = y[:, 1:2]
        
        Delta = params[:, 1:2]
        
        # Correcciones esperadas (lo que debería aprender la red)
        expected_du = Delta * v       # Término faltante en du
        expected_dv = -Delta * u      # Término faltante en dv
        expected_dw = torch.zeros_like(u)  # No falta nada en dw
        
        expected = torch.cat([expected_du, expected_dv, expected_dw], dim=-1)
        
        # Correcciones aprendidas
        learned = self.correction_net(y, params)
        
        return {
            'learned': learned,
            'expected': expected,
            'difference': learned - expected,
        }
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'activation': self.config.activation,
            'use_layer_norm': self.config.use_layer_norm,
            'dropout': self.config.dropout,
            'physics_mode': self.config.physics_mode,
            'gate_value': self.config.gate_value,
            'lambda_cons': self.config.lambda_cons,
            'lambda_cons_auto': self.config.lambda_cons_auto,
            'lambda_cons_scale': self.config.lambda_cons_scale,
            'n_parameters': self.count_parameters(),
        }


# =============================================================================
# BASELINE: Solo física incompleta (sin red)
# =============================================================================

class PhysicsOnlyBaseline(nn.Module):
    """
    Baseline que solo usa física incompleta.
    Sirve para medir cuánto mejora la red.
    """
    
    def forward(self, y0: torch.Tensor, t_span: torch.Tensor,
                params: torch.Tensor) -> torch.Tensor:
        """Integra solo con física incompleta"""
        y = y0.clone()
        outputs = [y]
        
        for i in range(len(t_span) - 1):
            dt = (t_span[i+1] - t_span[i]).item()
            # Solo física, sin corrección
            y = rk4_step(y, params, bloch_rhs_incomplete, dt)
            outputs.append(y)
        
        return torch.stack(outputs, dim=0)


# =============================================================================
# TEST
# =============================================================================

def _test():
    """Test del modelo con diferentes configuraciones de lambda"""
    print("=" * 70)
    print("TEST: PINODEv3 con lambdas configurables")
    print("=" * 70)
    
    # Test 1: Lambda = 0 (comportamiento original)
    print("\n--- Test 1: lambda_cons = 0 (original) ---")
    config1 = PINODEv3Config(
        hidden_dim=256,
        num_layers=4,
        gate_value=0.5,
        lambda_cons=0.0,
    )
    model1 = PINODEv3(config1)
    print(f"✓ Modelo creado: {model1.count_parameters():,} parámetros")
    
    # Test 2: Lambda fijo
    print("\n--- Test 2: lambda_cons = 0.01 (fijo) ---")
    config2 = PINODEv3Config(
        hidden_dim=256,
        num_layers=4,
        gate_value=0.5,
        lambda_cons=0.01,
    )
    model2 = PINODEv3(config2)
    print(f"✓ lambda_cons = {config2.lambda_cons}")
    
    # Test 3: Lambda auto
    print("\n--- Test 3: lambda_cons_auto = True ---")
    config3 = PINODEv3Config(
        hidden_dim=256,
        num_layers=4,
        gate_value=0.5,
        lambda_cons_auto=True,
        lambda_cons_scale=0.1,
    )
    model3 = PINODEv3(config3)
    print(f"✓ lambda_cons_auto = True, scale = {config3.lambda_cons_scale}")
    
    # Datos de prueba
    batch = 8
    T = 100
    
    y0 = torch.zeros(batch, 3)
    y0[:, 2] = -1.0  # w0 = -1
    
    t_span = torch.linspace(0, 30, T)
    
    params = torch.zeros(batch, 3)
    params[:, 0] = 2.0   # Omega
    params[:, 1] = 0.5   # Delta
    params[:, 2] = 0.02  # gamma
    
    print(f"\n--- Test forward y loss ---")
    
    for i, (name, model) in enumerate([
        ("lambda=0", model1),
        ("lambda=0.01", model2),
        ("lambda_auto", model3)
    ], 1):
        with torch.no_grad():
            traj = model(y0, t_span, params)
        
        # Simular targets con algo de error
        targets = traj.permute(1, 0, 2) + 0.05 * torch.randn_like(traj.permute(1, 0, 2))
        
        losses = model.compute_loss(traj, targets, params, return_diagnostics=True)
        
        print(f"\n  [{name}]")
        print(f"    MSE:          {losses['mse'].item():.4e}")
        print(f"    Conservation: {losses['conservation'].item():.4e}")
        print(f"    Lambda eff:   {losses['lambda_eff'].item():.4e}")
        print(f"    Total:        {losses['total'].item():.4e}")
        
        if 'diagnostics' in losses:
            d = losses['diagnostics']
            print(f"    --- Diagnóstico ---")
            print(f"    Ratio MSE/Cons:    {d['ratio_mse_cons']:.2f}")
            print(f"    λ sugerido (10%):  {d['lambda_suggested_10pct']:.4e}")
            print(f"    Cons contribution: {d['cons_contribution']:.4e}")
            print(f"    Cons % of total:   {d['cons_pct_of_total']:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ Todos los tests pasaron!")
    print("=" * 70)


if __name__ == "__main__":
    _test()