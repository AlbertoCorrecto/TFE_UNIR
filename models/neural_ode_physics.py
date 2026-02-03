#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural ODE con Loss Física (Neural ODE-P)
==========================================

Arquitectura Neural ODE estándar pero entrenada con supervisión física
en lugar de (o además de) supervisión por datos.


Variantes de Loss:
- 'physics': Solo residuo físico ||f_neural - f_Bloch||²
- 'data': Solo MSE de trayectorias (Neural ODE estándar)
- 'hybrid': Combinación ponderada de ambas

Autor: Alberto Vidal Fernández
Fecha: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("⚠️  torchdiffeq no instalado. Usando integrador RK4 interno.")


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class NeuralODEPhysicsConfig:
    """Configuración para Neural ODE con Loss Física"""
    
    # Arquitectura de red
    hidden_dim: int = 256
    num_layers: int = 4
    activation: str = "silu"
    use_layer_norm: bool = True
    dropout: float = 0.0
    
    # ODE Solver
    solver: str = "rk4"
    step_size: float = 0.05
    rtol: float = 1e-5
    atol: float = 1e-6
    use_adjoint: bool = False
    
    # Dimensiones (fijas para Bloch)
    state_dim: int = 3      # [u, v, w]
    param_dim: int = 3      # [Ω, Δ, γ]
    
    # Modo de Loss
    loss_mode: str = "physics"  # 'physics', 'data', 'hybrid'
    lambda_physics: float = 1.0
    lambda_data: float = 1.0
    
    # Física objetivo (qué ecuaciones debe aprender)
    physics_target: str = "complete"  # 'complete', 'incomplete', 'minimal'
    
    # Puntos de colocación para loss física
    n_collocation: int = 1000
    collocation_strategy: str = "uniform"  # 'uniform', 'trajectory'
    collocation_source: str = "uniform"    # Alias para compatibilidad
    
    # Regularización
    lambda_norm: float = 0.0  # Penalización si ||y|| > 1
    
    @property
    def input_dim(self) -> int:
        return self.state_dim + self.param_dim  # 6
    
    @property
    def output_dim(self) -> int:
        return self.state_dim  # 3


# =============================================================================
# ECUACIONES DE BLOCH (TARGETS FÍSICOS)
# =============================================================================

def bloch_rhs_complete(y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Ecuaciones de Bloch COMPLETAS.
    
    du/dt = -γu + Δv
    dv/dt = -γv - Δu - Ωw
    dw/dt = -2γ(w+1) + Ωv
    """
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Omega, Delta, gamma = params[:, 0:1], params[:, 1:2], params[:, 2:3]
    
    du = -gamma * u + Delta * v
    dv = -gamma * v - Delta * u - Omega * w
    dw = -2.0 * gamma * (w + 1.0) + Omega * v
    
    return torch.cat([du, dv, dw], dim=-1)


def bloch_rhs_incomplete(y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Ecuaciones de Bloch INCOMPLETAS (sin detuning Δ).
    
    du/dt = -γu           (falta: +Δv)
    dv/dt = -γv - Ωw      (falta: -Δu)
    dw/dt = -2γ(w+1) + Ωv
    """
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    Omega, gamma = params[:, 0:1], params[:, 2:3]
    
    du = -gamma * u
    dv = -gamma * v - Omega * w
    dw = -2.0 * gamma * (w + 1.0) + Omega * v
    
    return torch.cat([du, dv, dw], dim=-1)


def bloch_rhs_minimal(y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Ecuaciones de Bloch MÍNIMAS (solo disipación γ).
    
    du/dt = -γu
    dv/dt = -γv
    dw/dt = -2γ(w+1)
    """
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    gamma = params[:, 2:3]
    
    du = -gamma * u
    dv = -gamma * v
    dw = -2.0 * gamma * (w + 1.0)
    
    return torch.cat([du, dv, dw], dim=-1)


def get_physics_function(name: str):
    """Obtiene la función de física por nombre"""
    functions = {
        'complete': bloch_rhs_complete,
        'incomplete': bloch_rhs_incomplete,
        'minimal': bloch_rhs_minimal,
    }
    if name not in functions:
        raise ValueError(f"Física desconocida: {name}. Opciones: {list(functions.keys())}")
    return functions[name]


# =============================================================================
# COMPONENTES DE RED
# =============================================================================

def get_activation(name: str) -> nn.Module:
    """Obtiene función de activación por nombre"""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(0.1),
    }
    if name not in activations:
        raise ValueError(f"Activación desconocida: {name}")
    return activations[name]


class MLPBlock(nn.Module):
    """Bloque MLP con normalización y dropout opcional"""
    
    def __init__(self, in_dim: int, out_dim: int, activation: str = "silu",
                 use_layer_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        
        layers = [nn.Linear(in_dim, out_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================================
# FUNCIÓN ODE NEURONAL
# =============================================================================

class ODEFunc(nn.Module):
    """
    Red neuronal que aproxima dy/dt = f(y, params).
    
    Esta es la "caja negra" que debe aprender a ser las ecuaciones de Bloch.
    """
    
    def __init__(self, config: NeuralODEPhysicsConfig):
        super().__init__()
        self.config = config
        
        # Construir backbone
        layers = []
        layers.append(MLPBlock(
            config.input_dim, config.hidden_dim,
            config.activation, config.use_layer_norm, config.dropout
        ))
        
        for _ in range(config.num_layers - 1):
            layers.append(MLPBlock(
                config.hidden_dim, config.hidden_dim,
                config.activation, config.use_layer_norm, config.dropout
            ))
        
        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(config.hidden_dim, config.output_dim)
        
        self._init_weights()
        self._params = None
    
    def _init_weights(self):
        """Inicialización de pesos"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Salida con ganancia pequeña
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
    
    def set_params(self, params: torch.Tensor):
        """Establece parámetros físicos para integración"""
        self._params = params
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Calcula dy/dt.
        
        Args:
            t: Tiempo (escalar, no usado pero requerido por torchdiffeq)
            state: Estado (batch, 3)
        
        Returns:
            Derivada (batch, 3)
        """
        if self._params is None:
            raise RuntimeError("Llamar set_params() primero")
        
        x = torch.cat([state, self._params], dim=-1)
        h = self.backbone(x)
        return self.output_layer(h)
    
    def forward_with_params(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Calcula dy/dt con parámetros explícitos (para loss física).
        
        Args:
            state: Estado (batch, 3)
            params: Parámetros (batch, 3)
        
        Returns:
            Derivada (batch, 3)
        """
        x = torch.cat([state, params], dim=-1)
        h = self.backbone(x)
        return self.output_layer(h)


# =============================================================================
# INTEGRADOR RK4 INTERNO
# =============================================================================

def rk4_step(y: torch.Tensor, params: torch.Tensor, 
             f_func, dt: float) -> torch.Tensor:
    """Un paso de Runge-Kutta 4"""
    k1 = f_func(y, params)
    k2 = f_func(y + 0.5 * dt * k1, params)
    k3 = f_func(y + 0.5 * dt * k2, params)
    k4 = f_func(y + dt * k3, params)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_rk4(y0: torch.Tensor, t_span: torch.Tensor, 
                  params: torch.Tensor, f_func) -> torch.Tensor:
    """Integra usando RK4 interno"""
    y = y0.clone()
    outputs = [y]
    
    for i in range(len(t_span) - 1):
        dt = (t_span[i+1] - t_span[i]).item()
        y = rk4_step(y, params, f_func, dt)
        outputs.append(y)
    
    return torch.stack(outputs, dim=0)  # (T, batch, 3)


# =============================================================================
# MUESTREO DE PUNTOS DE COLOCACIÓN
# =============================================================================

class CollocationSampler:
    """
    Genera puntos de colocación para el loss física.
    
    Los puntos pueden ser:
    - 'uniform': Uniformes en el espacio de estados válido
    - 'trajectory': A lo largo de trayectorias integradas
    - 'importance': Concentrados donde el error es mayor
    """
    
    def __init__(self, config: NeuralODEPhysicsConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # Rangos para muestreo uniforme (esfera de Bloch: ||y|| ≤ 1)
        self.state_ranges = {
            'u': (-1.0, 1.0),
            'v': (-1.0, 1.0),
            'w': (-1.0, 1.0),
        }
    
    def sample_uniform(self, n_points: int, param_ranges: Dict[str, Tuple[float, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Muestrea puntos uniformemente en el espacio de estados.
        
        Returns:
            states: (n_points, 3)
            params: (n_points, 3)
        """
        # Muestrear estados en la esfera de Bloch
        # Método: rechazar puntos con norma > 1
        states = []
        while len(states) < n_points:
            # Muestrear uniformemente en cubo [-1, 1]³
            candidates = torch.rand(n_points * 2, 3, device=self.device) * 2 - 1
            
            # Filtrar por norma ≤ 1
            norms = torch.norm(candidates, dim=-1)
            valid = candidates[norms <= 1.0]
            states.append(valid)
        
        states = torch.cat(states, dim=0)[:n_points]
        
        # Muestrear parámetros
        Omega = torch.rand(n_points, 1, device=self.device) * (param_ranges['Omega'][1] - param_ranges['Omega'][0]) + param_ranges['Omega'][0]
        Delta = torch.rand(n_points, 1, device=self.device) * (param_ranges['Delta'][1] - param_ranges['Delta'][0]) + param_ranges['Delta'][0]
        gamma = torch.rand(n_points, 1, device=self.device) * (param_ranges['gamma'][1] - param_ranges['gamma'][0]) + param_ranges['gamma'][0]
        
        params = torch.cat([Omega, Delta, gamma], dim=-1)
        
        return states, params
    
    def sample_from_trajectories(self, trajectories: torch.Tensor, 
                                 params: torch.Tensor, 
                                 n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Muestrea puntos a lo largo de trayectorias dadas.
        
        Args:
            trajectories: (batch, T, 3)
            params: (batch, 3)
            n_points: Número de puntos a muestrear
        
        Returns:
            states: (n_points, 3)
            params_out: (n_points, 3)
        """
        batch, T, _ = trajectories.shape
        
        # Aplanar trayectorias
        all_states = trajectories.reshape(-1, 3)  # (batch*T, 3)
        
        # Expandir parámetros
        params_expanded = params.unsqueeze(1).expand(-1, T, -1).reshape(-1, 3)  # (batch*T, 3)
        
        # Muestrear índices aleatorios
        n_total = all_states.shape[0]
        indices = torch.randperm(n_total, device=self.device)[:n_points]
        
        return all_states[indices], params_expanded[indices]


# =============================================================================
# MODELO PRINCIPAL
# =============================================================================

class NeuralODEPhysics(nn.Module):
    """
    Neural ODE con Loss Física.
    
    Arquitectura: dy/dt = f_neural(y, θ) (igual que Neural ODE estándar)
    Loss: ||f_neural(y, θ) - f_Bloch(y, θ)||² (supervisión por física)
    
    La red aprende a SER las ecuaciones de Bloch.
    """
    
    def __init__(self, config: NeuralODEPhysicsConfig = None):
        super().__init__()
        
        self.config = config or NeuralODEPhysicsConfig()
        self.ode_func = ODEFunc(self.config)
        
        # Función física objetivo
        self.physics_fn = get_physics_function(self.config.physics_target)
        
        # Sampler para puntos de colocación
        self.collocation_sampler = None  # Se inicializa en set_device
        
        # Seleccionar integrador
        if HAS_TORCHDIFFEQ and self.config.solver != "rk4_internal":
            if self.config.use_adjoint:
                self._odeint = odeint_adjoint
            else:
                self._odeint = odeint
            self._use_torchdiffeq = True
        else:
            self._use_torchdiffeq = False
    
    def set_device(self, device: str):
        """Configura el dispositivo y el sampler"""
        self.collocation_sampler = CollocationSampler(self.config, device)
    
    def _f_for_rk4(self, y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Wrapper para usar con RK4 interno"""
        return self.ode_func.forward_with_params(y, params)
    
    def forward(self, y0: torch.Tensor, t_span: torch.Tensor,
                params: torch.Tensor) -> torch.Tensor:
        """
        Integra la ODE para obtener trayectoria.
        
        Args:
            y0: Estado inicial (batch, 3)
            t_span: Tiempos (T,)
            params: Parámetros (batch, 3)
        
        Returns:
            Trayectoria (T, batch, 3)
        """
        if self._use_torchdiffeq:
            self.ode_func.set_params(params)
            trajectory = self._odeint(
                self.ode_func, y0, t_span,
                method=self.config.solver,
                rtol=self.config.rtol,
                atol=self.config.atol
            )
        else:
            trajectory = integrate_rk4(y0, t_span, params, self._f_for_rk4)
        
        return trajectory
    
    def predict_trajectory(self, y0: torch.Tensor, t_span: torch.Tensor,
                          params: torch.Tensor) -> torch.Tensor:
        """Devuelve (batch, T, 3)"""
        traj = self.forward(y0, t_span, params)
        return traj.permute(1, 0, 2)
    
    def compute_physics_loss(self, states: torch.Tensor, 
                             params: torch.Tensor) -> torch.Tensor:
        """
        Calcula el loss de residuo físico.
        
        Loss = ||f_neural(y, θ) - f_physics(y, θ)||²
        
        Args:
            states: Puntos de colocación (N, 3)
            params: Parámetros físicos (N, 3)
        
        Returns:
            Loss escalar
        """
        # Predicción de la red
        f_pred = self.ode_func.forward_with_params(states, params)
        
        # Target físico
        f_true = self.physics_fn(states, params)
        
        # MSE
        return ((f_pred - f_true) ** 2).mean()
    
    def compute_data_loss(self, y_pred: torch.Tensor, 
                          y_true: torch.Tensor) -> torch.Tensor:
        """
        Calcula el loss de datos (MSE de trayectorias).
        
        Args:
            y_pred: Trayectorias predichas (batch, T, 3) o (T, batch, 3)
            y_true: Trayectorias verdaderas (batch, T, 3)
        
        Returns:
            Loss escalar
        """
        if y_pred.dim() == 3 and y_pred.shape[0] != y_true.shape[0]:
            y_pred = y_pred.permute(1, 0, 2)
        
        return ((y_pred - y_true) ** 2).mean()
    
    def compute_norm_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Penalización si ||y|| > 1 (fuera de esfera de Bloch).
        """
        norms = torch.norm(states, dim=-1)
        return torch.relu(norms - 1.0).pow(2).mean()
    
    def compute_loss(self, 
                     y_pred: Optional[torch.Tensor] = None,
                     y_true: Optional[torch.Tensor] = None,
                     collocation_states: Optional[torch.Tensor] = None,
                     collocation_params: Optional[torch.Tensor] = None,
                     return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Calcula el loss total según el modo configurado.
        
        Args:
            y_pred: Trayectorias predichas (para loss de datos)
            y_true: Trayectorias verdaderas (para loss de datos)
            collocation_states: Puntos de colocación (para loss física)
            collocation_params: Parámetros en puntos de colocación
            return_components: Si devolver componentes individuales
        
        Returns:
            Dict con 'total' y opcionalmente componentes
        """
        device = next(self.parameters()).device
        losses = {}
        total = torch.tensor(0.0, device=device)
        
        # Loss física
        if self.config.loss_mode in ['physics', 'hybrid']:
            if collocation_states is None or collocation_params is None:
                raise ValueError("Se requieren puntos de colocación para loss física")
            
            physics_loss = self.compute_physics_loss(collocation_states, collocation_params)
            losses['physics'] = physics_loss
            total = total + self.config.lambda_physics * physics_loss
        
        # Loss de datos
        if self.config.loss_mode in ['data', 'hybrid']:
            if y_pred is None or y_true is None:
                raise ValueError("Se requieren trayectorias para loss de datos")
            
            data_loss = self.compute_data_loss(y_pred, y_true)
            losses['data'] = data_loss
            total = total + self.config.lambda_data * data_loss
        
        # Loss de norma (opcional)
        if self.config.lambda_norm > 0 and collocation_states is not None:
            norm_loss = self.compute_norm_loss(collocation_states)
            losses['norm'] = norm_loss
            total = total + self.config.lambda_norm * norm_loss
        
        losses['total'] = total
        
        return losses
    
    def get_derivative_comparison(self, states: torch.Tensor, 
                                  params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compara derivadas predichas vs físicas (para análisis).
        
        Returns:
            Dict con 'predicted', 'target', 'error', 'correlation'
        """
        with torch.no_grad():
            f_pred = self.ode_func.forward_with_params(states, params)
            f_true = self.physics_fn(states, params)
            
            error = f_pred - f_true
            
            # Correlación por componente
            correlations = []
            for i in range(3):
                pred_i = f_pred[:, i]
                true_i = f_true[:, i]
                
                if pred_i.std() > 1e-8 and true_i.std() > 1e-8:
                    corr = torch.corrcoef(torch.stack([pred_i, true_i]))[0, 1]
                else:
                    corr = torch.tensor(0.0)
                correlations.append(corr)
            
            return {
                'predicted': f_pred,
                'target': f_true,
                'error': error,
                'mse': (error ** 2).mean(),
                'mae': error.abs().mean(),
                'correlations': torch.stack(correlations),
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
            'solver': self.config.solver,
            'loss_mode': self.config.loss_mode,
            'physics_target': self.config.physics_target,
            'lambda_physics': self.config.lambda_physics,
            'lambda_data': self.config.lambda_data,
            'n_collocation': self.config.n_collocation,
            'n_parameters': self.count_parameters(),
        }


# =============================================================================
# TEST
# =============================================================================

def _test():
    """Test del modelo"""
    print("=" * 70)
    print("TEST: Neural ODE con Loss Física")
    print("=" * 70)
    
    # Configuración
    config = NeuralODEPhysicsConfig(
        hidden_dim=256,
        num_layers=4,
        loss_mode='physics',
        physics_target='complete',
        n_collocation=500,
    )
    
    model = NeuralODEPhysics(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.set_device(device)
    
    n_params = model.count_parameters()
    print(f"✓ Modelo creado: {n_params:,} parámetros")
    print(f"  Loss mode: {config.loss_mode}")
    print(f"  Physics target: {config.physics_target}")
    
    # Test forward (integración)
    batch = 8
    T = 100
    
    y0 = torch.zeros(batch, 3, device=device)
    y0[:, 2] = -1.0
    
    t_span = torch.linspace(0, 30, T, device=device)
    
    params = torch.zeros(batch, 3, device=device)
    params[:, 0] = 2.0   # Omega
    params[:, 1] = 0.5   # Delta
    params[:, 2] = 0.02  # gamma
    
    print(f"\n✓ Test forward:")
    with torch.no_grad():
        traj = model(y0, t_span, params)
    print(f"  Output shape: {traj.shape}")
    
    # Test loss física
    print(f"\n✓ Test loss física:")
    
    # Generar puntos de colocación
    param_ranges = {
        'Omega': (0.5, 5.0),
        'Delta': (-2.0, 2.0),
        'gamma': (0.01, 0.1),
    }
    
    coll_states, coll_params = model.collocation_sampler.sample_uniform(
        config.n_collocation, param_ranges
    )
    
    losses = model.compute_loss(
        collocation_states=coll_states,
        collocation_params=coll_params,
        return_components=True
    )
    
    print(f"  Physics loss: {losses['physics'].item():.6f}")
    print(f"  Total loss: {losses['total'].item():.6f}")
    
    # Test comparación de derivadas
    print(f"\n✓ Test comparación derivadas (antes de entrenar):")
    comparison = model.get_derivative_comparison(coll_states[:100], coll_params[:100])
    print(f"  MSE: {comparison['mse'].item():.6f}")
    print(f"  Correlaciones: {comparison['correlations'].tolist()}")
    
    # Test modo híbrido
    print(f"\n✓ Test modo híbrido:")
    config_hybrid = NeuralODEPhysicsConfig(
        loss_mode='hybrid',
        lambda_physics=1.0,
        lambda_data=1.0,
    )
    model_hybrid = NeuralODEPhysics(config_hybrid)
    model_hybrid.to(device)
    model_hybrid.set_device(device)
    
    # Generar trayectorias "verdaderas" (simuladas)
    with torch.no_grad():
        y_true = model_hybrid.predict_trajectory(y0, t_span, params)
        y_pred = model_hybrid.predict_trajectory(y0, t_span, params)
    
    losses_hybrid = model_hybrid.compute_loss(
        y_pred=y_pred,
        y_true=y_true,
        collocation_states=coll_states,
        collocation_params=coll_params,
    )
    
    print(f"  Physics loss: {losses_hybrid['physics'].item():.6f}")
    print(f"  Data loss: {losses_hybrid['data'].item():.6f}")
    print(f"  Total: {losses_hybrid['total'].item():.6f}")
    
    print("\n" + "=" * 70)
    print("✅ Todos los tests pasaron!")
    print("=" * 70)


if __name__ == "__main__":
    _test()