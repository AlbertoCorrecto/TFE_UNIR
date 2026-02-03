"""
Neural ODE para el Oscilador de Rabi
=====================================

Arquitectura que aprende la dinámica del sistema sin conocimiento físico previo.
La red aprende directamente la función f tal que dy/dt = f(y, params).

Entrada: [u, v, w, Ω, Δ, γ] (6 dims)
Salida:  [du/dt, dv/dt, dw/dt] (3 dims)

Autor: Alberto Vidal Fernández
Fecha: 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    from torchdiffeq import odeint, odeint_adjoint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("⚠️  torchdiffeq no instalado. Instalar con: pip install torchdiffeq")


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class NeuralODEConfig:
    """Configuración de la arquitectura Neural ODE"""
    # Arquitectura
    hidden_dim: int = 256
    num_layers: int = 4
    activation: str = "silu"
    use_layer_norm: bool = True
    dropout: float = 0.0
    
    # ODE Solver
    solver: str = "rk4" # "dopri5" mas lento, mejor*
    step_size=0.05
    rtol: float = 1e-5
    atol: float = 1e-6
    use_adjoint: bool = False # Controla memoria, puede duplicar velocidad
    
    # Dimensiones (fijas para este problema)
    state_dim: int = 3      # [u, v, w]
    param_dim: int = 3      # [Ω, Δ, γ]
    
    @property
    def input_dim(self) -> int:
        return self.state_dim + self.param_dim  # 6
    
    @property
    def output_dim(self) -> int:
        return self.state_dim  # 3


# Configuración por defecto (~350k parámetros)
DEFAULT_CONFIG = NeuralODEConfig()


# =============================================================================
# COMPONENTES
# =============================================================================

def get_activation(name: str) -> nn.Module:
    """Obtiene la función de activación por nombre"""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),  # Swish
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(0.1),
    }
    if name not in activations:
        raise ValueError(f"Activación desconocida: {name}. Opciones: {list(activations.keys())}")
    return activations[name]


class MLPBlock(nn.Module):
    """Bloque MLP con normalización y dropout opcional"""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "silu",
        use_layer_norm: bool = True,
        dropout: float = 0.0
    ):
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
# FUNCIÓN ODE (dy/dt = f(y, params))
# =============================================================================

class ODEFunc(nn.Module):
    """
    Red neuronal que aproxima la derivada temporal del estado.
    
    Aprende: f(u, v, w, Ω, Δ, γ) → (du/dt, dv/dt, dw/dt)
    
    Esta es una caja negra que no conoce las ecuaciones de Bloch.
    """
    
    def __init__(self, config: NeuralODEConfig = None):
        super().__init__()
        
        if config is None:
            config = DEFAULT_CONFIG
        
        self.config = config
        
        # Construir red
        layers = []
        
        # Capa de entrada
        layers.append(MLPBlock(
            config.input_dim,
            config.hidden_dim,
            config.activation,
            config.use_layer_norm,
            config.dropout
        ))
        
        # Capas ocultas
        for _ in range(config.num_layers - 1):
            layers.append(MLPBlock(
                config.hidden_dim,
                config.hidden_dim,
                config.activation,
                config.use_layer_norm,
                config.dropout
            ))
        
        self.backbone = nn.Sequential(*layers)
        
        # Capa de salida (sin activación)
        self.output_layer = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Inicialización
        self._init_weights()
        
        # Parámetros del sistema (se setean antes de integrar)
        self._params = None
    
    def _init_weights(self):
        """Inicialización de pesos"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Capa de salida con ganancia pequeña (predicciones iniciales cercanas a 0)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
    
    def set_params(self, params: torch.Tensor):
        """
        Establece los parámetros físicos para la integración.
        
        Args:
            params: Tensor (batch, 3) con [Ω, Δ, γ] para cada trayectoria
        """
        self._params = params
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Calcula dy/dt dado el estado actual.
        
        Args:
            t: Tiempo actual (escalar, no usado pero requerido por torchdiffeq)
            state: Estado actual (batch, 3) = [u, v, w]
        
        Returns:
            Derivada (batch, 3) = [du/dt, dv/dt, dw/dt]
        """
        if self._params is None:
            raise RuntimeError("Parámetros no establecidos. Llamar set_params() primero.")
        
        # Concatenar estado y parámetros
        x = torch.cat([state, self._params], dim=-1)  # (batch, 6)
        
        # Forward pass
        h = self.backbone(x)
        dydt = self.output_layer(h)
        
        return dydt


# =============================================================================
# MODELO COMPLETO
# =============================================================================

class NeuralODE(nn.Module):
    """
    Modelo Neural ODE completo para el oscilador de Rabi.
    
    Integra la ODE aprendida para predecir trayectorias completas.
    
    Uso:
        model = NeuralODE()
        trajectory = model(initial_state, t_span, params)
        # trajectory: (T, batch, 3)
    """
    
    def __init__(self, config: NeuralODEConfig = None):
        super().__init__()
        
        if config is None:
            config = DEFAULT_CONFIG
        
        self.config = config
        self.ode_func = ODEFunc(config)
        
        # Seleccionar integrador
        if config.use_adjoint:
            self._odeint = odeint_adjoint
        else:
            self._odeint = odeint
    
    def forward(
        self,
        initial_state: torch.Tensor,
        t_span: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Integra la ODE para obtener la trayectoria completa.
        
        Args:
            initial_state: Estado inicial (batch, 3) = [u₀, v₀, w₀]
            t_span: Tiempos de evaluación (T,)
            params: Parámetros físicos (batch, 3) = [Ω, Δ, γ]
        
        Returns:
            Trayectoria (T, batch, 3)
        """
        # Establecer parámetros en la función ODE
        self.ode_func.set_params(params)
        
        # Integrar
        trajectory = self._odeint(
            self.ode_func,
            initial_state,
            t_span,
            method=self.config.solver,
            rtol=self.config.rtol,
            atol=self.config.atol
        )
        
        return trajectory
    
    def predict_trajectory(
        self,
        initial_state: torch.Tensor,
        t_span: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Alias de forward() con nombre más descriptivo.
        Devuelve (batch, T, 3) en lugar de (T, batch, 3).
        """
        trajectory = self.forward(initial_state, t_span, params)
        return trajectory.permute(1, 0, 2)  # (batch, T, 3)
    
    def count_parameters(self) -> int:
        """Cuenta el número total de parámetros entrenables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        """Devuelve la configuración como diccionario"""
        return {
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'activation': self.config.activation,
            'use_layer_norm': self.config.use_layer_norm,
            'dropout': self.config.dropout,
            'solver': self.config.solver,
            'rtol': self.config.rtol,
            'atol': self.config.atol,
            'use_adjoint': self.config.use_adjoint,
            'n_parameters': self.count_parameters(),
        }


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def create_neural_ode(
    hidden_dim: int = 256,
    num_layers: int = 4,
    activation: str = "silu",
    **kwargs
) -> NeuralODE:
    """
    Crea un modelo Neural ODE con la configuración especificada.
    
    Args:
        hidden_dim: Dimensión de capas ocultas
        num_layers: Número de capas
        activation: Función de activación
        **kwargs: Otros argumentos para NeuralODEConfig
    
    Returns:
        Modelo NeuralODE
    """
    config = NeuralODEConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        **kwargs
    )
    return NeuralODE(config)


def load_neural_ode(path: str, device: str = "cpu") -> NeuralODE:
    """
    Carga un modelo Neural ODE desde un archivo.
    
    Args:
        path: Ruta al archivo .pt
        device: Dispositivo donde cargar el modelo
    
    Returns:
        Modelo cargado
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Reconstruir configuración
    if 'config' in checkpoint:
        config = NeuralODEConfig(**checkpoint['config'])
    else:
        config = DEFAULT_CONFIG
    
    model = NeuralODE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model


# =============================================================================
# TEST
# =============================================================================

def _test():
    """Test básico del modelo"""
    print("="*60)
    print("TEST: Neural ODE")
    print("="*60)
    
    # Crear modelo
    model = NeuralODE()
    n_params = model.count_parameters()
    print(f"✓ Modelo creado: {n_params:,} parámetros")
    
    # Test forward pass
    batch_size = 8
    T = 100
    
    initial_state = torch.zeros(batch_size, 3)
    initial_state[:, 2] = -1.0  # w₀ = -1 (estado fundamental)
    
    t_span = torch.linspace(0, 10, T)
    
    params = torch.zeros(batch_size, 3)
    params[:, 0] = 2.0   # Ω
    params[:, 1] = 0.5   # Δ
    params[:, 2] = 0.01  # γ
    
    print(f"✓ Input: initial_state={initial_state.shape}, t_span={t_span.shape}, params={params.shape}")
    
    # Forward
    with torch.no_grad():
        trajectory = model(initial_state, t_span, params)
    
    print(f"✓ Output: trajectory={trajectory.shape}")
    print(f"  Esperado: ({T}, {batch_size}, 3)")
    
    # Verificar forma
    assert trajectory.shape == (T, batch_size, 3), f"Shape incorrecto: {trajectory.shape}"
    
    # Verificar que no hay NaN
    assert not torch.isnan(trajectory).any(), "Hay NaN en la salida"
    
    # Verificar norma (debería estar cerca de 1 para γ pequeño)
    norms = torch.norm(trajectory, dim=-1)
    print(f"  Norma: [{norms.min():.4f}, {norms.max():.4f}]")
    
    # Test predict_trajectory (formato alternativo)
    traj_alt = model.predict_trajectory(initial_state, t_span, params)
    assert traj_alt.shape == (batch_size, T, 3), f"predict_trajectory shape incorrecto: {traj_alt.shape}"
    print(f"✓ predict_trajectory: {traj_alt.shape}")
    
    # Mostrar configuración
    print(f"\n📋 Configuración:")
    for k, v in model.get_config().items():
        print(f"   {k}: {v}")
    
    print("\n✅ Todos los tests pasaron!")
    print("="*60)


if __name__ == "__main__":
    _test()