#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_unified_data.py

Generación de datasets sintéticos para el TFM:
comparativa Neural ODE, PINODE y Neural ODE-P sobre ecuaciones ópticas de Bloch.

El script genera conjuntos de entrenamiento, validación y test con muestreo Sobol,
garantizando particiones independientes mediante semillas distintas.

Uso:
    python generate_unified_data.py --jobs 8 --output-dir datasets --datasets all

Autor: Alberto José Vidal Fernández
Año: 2025
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import qmc


# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    n_trajectories: int
    omega_range: Tuple[float, float]
    delta_range: Tuple[float, float]
    gamma_values: List[float]
    t_span: Tuple[float, float]
    n_time_points: int
    sampling: str = "sobol"  # "sobol", "random", "grid"
    seed: int = 42
    description: str = ""
    omega_log_scale: bool = True


DATASETS: Dict[str, DatasetConfig] = {
    "train": DatasetConfig(
        name="train",
        n_trajectories=5000,
        omega_range=(0.5, 5.0),
        delta_range=(-3.0, 3.0),
        gamma_values=[0.0, 0.02, 0.05],
        t_span=(0.0, 30.0),
        n_time_points=400,
        sampling="sobol",
        seed=42,
        description="Entrenamiento",
    ),
    "val": DatasetConfig(
        name="val",
        n_trajectories=1000,
        omega_range=(0.5, 5.0),
        delta_range=(-3.0, 3.0),
        gamma_values=[0.0, 0.02, 0.05],
        t_span=(0.0, 30.0),
        n_time_points=400,
        sampling="sobol",
        seed=123,
        description="Validación",
    ),
    "test_id": DatasetConfig(
        name="test_id",
        n_trajectories=1000,
        omega_range=(0.5, 5.0),
        delta_range=(-3.0, 3.0),
        gamma_values=[0.0, 0.02, 0.05],
        t_span=(0.0, 30.0),
        n_time_points=400,
        sampling="sobol",
        seed=456,
        description="Test in-distribution",
    ),
    "test_resonance": DatasetConfig(
        name="test_resonance",
        n_trajectories=500,
        omega_range=(1.0, 4.0),
        delta_range=(-0.5, 0.5),
        gamma_values=[0.0, 0.02],
        t_span=(0.0, 30.0),
        n_time_points=400,
        sampling="sobol",
        seed=789,
        description="Test cerca de resonancia (Δ ≈ 0)",
    ),
    "test_extrap_time": DatasetConfig(
        name="test_extrap_time",
        n_trajectories=500,
        omega_range=(0.5, 5.0),
        delta_range=(-3.0, 3.0),
        gamma_values=[0.0, 0.02, 0.05],
        t_span=(0.0, 60.0),
        n_time_points=800,
        sampling="sobol",
        seed=1011,
        description="Test extrapolación temporal (t hasta 60)",
    ),
    "test_extrap_params": DatasetConfig(
        name="test_extrap_params",
        n_trajectories=500,
        omega_range=(5.0, 8.0),
        delta_range=(3.0, 5.0),
        gamma_values=[0.0, 0.02, 0.05],
        t_span=(0.0, 30.0),
        n_time_points=400,
        sampling="sobol",
        seed=2022,
        description="Test extrapolación paramétrica (Ω, Δ fuera de rango)",
    ),
}


# ---------------------------------------------------------------------
# Dinámica: ecuaciones de Bloch
# ---------------------------------------------------------------------

def bloch_ode(
    t: float,
    y: np.ndarray,
    omega: float,
    delta: float,
    gamma: float,
) -> np.ndarray:
    """Ecuaciones ópticas de Bloch para y = [u, v, w]."""
    u, v, w = y
    du = -gamma * u + delta * v
    dv = -gamma * v - delta * u - omega * w
    dw = -2.0 * gamma * (w + 1.0) + omega * v
    return np.array([du, dv, dw], dtype=np.float64)


def solve_bloch(
    omega: float,
    delta: float,
    gamma: float,
    t_span: Tuple[float, float],
    n_points: int,
    y0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integra las ecuaciones de Bloch con un solver de alta precisión."""
    if y0 is None:
        y0 = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    t_eval = np.linspace(t_span[0], t_span[1], n_points, dtype=np.float64)

    sol = solve_ivp(
        fun=bloch_ode,
        t_span=t_span,
        y0=y0,
        args=(omega, delta, gamma),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success or sol.y is None:
        raise RuntimeError(f"Integración fallida: {sol.message}")

    # sol.y tiene forma (dim, n_points)
    y = sol.y.T
    if y.shape != (n_points, 3):
        raise RuntimeError(f"Forma inesperada en la solución: {y.shape}")

    return sol.t, y


# ---------------------------------------------------------------------
# Muestreo de parámetros
# ---------------------------------------------------------------------

def _scale_omega(samples_01: np.ndarray, omega_range: Tuple[float, float], log_scale: bool) -> np.ndarray:
    if log_scale:
        lo, hi = np.log10(omega_range[0]), np.log10(omega_range[1])
        return 10 ** (lo + samples_01 * (hi - lo))
    return omega_range[0] + samples_01 * (omega_range[1] - omega_range[0])


def sample_parameters_sobol(
    n_samples: int,
    omega_range: Tuple[float, float],
    delta_range: Tuple[float, float],
    gamma_values: List[float],
    omega_log_scale: bool,
    seed: int,
) -> np.ndarray:
    """Muestreo quasi-Monte Carlo (Sobol) estratificado por valores discretos de γ."""
    if n_samples < len(gamma_values):
        raise ValueError("n_samples debe ser >= número de valores de gamma")

    n_gamma = len(gamma_values)
    base = n_samples // n_gamma
    remainder = n_samples % n_gamma

    blocks: List[np.ndarray] = []
    for i, gamma in enumerate(gamma_values):
        n_this = base + (1 if i < remainder else 0)

        sampler = qmc.Sobol(d=2, scramble=True, seed=seed + i)
        s01 = sampler.random(n_this)

        omega = _scale_omega(s01[:, 0], omega_range, omega_log_scale)
        delta = delta_range[0] + s01[:, 1] * (delta_range[1] - delta_range[0])
        gamma_arr = np.full(n_this, gamma, dtype=np.float64)

        blocks.append(np.column_stack([omega, delta, gamma_arr]))

    return np.vstack(blocks)


def sample_parameters_random(
    n_samples: int,
    omega_range: Tuple[float, float],
    delta_range: Tuple[float, float],
    gamma_values: List[float],
    omega_log_scale: bool,
    seed: int,
) -> np.ndarray:
    """Muestreo uniforme aleatorio (útil como baseline)."""
    rng = np.random.default_rng(seed)

    if omega_log_scale:
        log_omega = rng.uniform(np.log10(omega_range[0]), np.log10(omega_range[1]), n_samples)
        omega = 10 ** log_omega
    else:
        omega = rng.uniform(omega_range[0], omega_range[1], n_samples)

    delta = rng.uniform(delta_range[0], delta_range[1], n_samples)
    gamma = rng.choice(np.array(gamma_values), n_samples)

    return np.column_stack([omega, delta, gamma]).astype(np.float64)


# ---------------------------------------------------------------------
# Generación de trayectorias y datasets
# ---------------------------------------------------------------------

def _trajectory_worker(args: Tuple[int, float, float, float, Tuple[float, float], int]) -> dict:
    traj_id, omega, delta, gamma, t_span, n_points = args
    try:
        t, y = solve_bloch(omega, delta, gamma, t_span, n_points)
        return {
            "trajectory_id": traj_id,
            "omega": float(omega),
            "delta": float(delta),
            "gamma": float(gamma),
            "t": t,
            "u": y[:, 0],
            "v": y[:, 1],
            "w": y[:, 2],
            "success": True,
        }
    except Exception as exc:
        return {
            "trajectory_id": traj_id,
            "omega": float(omega),
            "delta": float(delta),
            "gamma": float(gamma),
            "success": False,
            "error": str(exc),
        }


def generate_dataset(config: DatasetConfig, n_jobs: int) -> pd.DataFrame:
    """Genera un dataset en formato largo (una fila por instante)."""
    print("\n" + "-" * 70)
    print(f"Dataset: {config.name} | {config.description}")
    print(f"Trajectorias: {config.n_trajectories}")
    print(f"Ω: {config.omega_range} ({'log' if config.omega_log_scale else 'lin'}) | Δ: {config.delta_range} | γ: {config.gamma_values}")
    print(f"t: {config.t_span} con {config.n_time_points} puntos | sampling={config.sampling} | seed={config.seed}")
    print("-" * 70)

    if config.sampling == "sobol":
        params = sample_parameters_sobol(
            n_samples=config.n_trajectories,
            omega_range=config.omega_range,
            delta_range=config.delta_range,
            gamma_values=config.gamma_values,
            omega_log_scale=config.omega_log_scale,
            seed=config.seed,
        )
    elif config.sampling == "random":
        params = sample_parameters_random(
            n_samples=config.n_trajectories,
            omega_range=config.omega_range,
            delta_range=config.delta_range,
            gamma_values=config.gamma_values,
            omega_log_scale=config.omega_log_scale,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Método de muestreo no soportado: {config.sampling}")

    args_list = [
        (i, params[i, 0], params[i, 1], params[i, 2], config.t_span, config.n_time_points)
        for i in range(len(params))
    ]

    start = time.time()
    if n_jobs > 1:
        with mp.Pool(processes=n_jobs) as pool:
            results = list(pool.imap(_trajectory_worker, args_list, chunksize=50))
    else:
        results = [_trajectory_worker(a) for a in args_list]
    elapsed = time.time() - start

    ok = [r for r in results if r["success"]]
    failed = len(results) - len(ok)
    print(f"Generadas {len(ok)} trayectorias en {elapsed:.1f}s. Fallidas: {failed}")

    rows = []
    for traj in ok:
        tid = traj["trajectory_id"]
        omega = traj["omega"]
        delta = traj["delta"]
        gamma = traj["gamma"]
        t_arr = traj["t"]
        u_arr = traj["u"]
        v_arr = traj["v"]
        w_arr = traj["w"]

        for j in range(len(t_arr)):
            rows.append(
                {
                    "trajectory_id": tid,
                    "t": float(t_arr[j]),
                    "u": float(u_arr[j]),
                    "v": float(v_arr[j]),
                    "w": float(w_arr[j]),
                    "Omega_R": omega,
                    "detuning": delta,
                    "gamma": gamma,
                }
            )

    df = pd.DataFrame(rows)

    norms = np.sqrt(df["u"] ** 2 + df["v"] ** 2 + df["w"] ** 2)
    print(f"Puntos: {len(df):,} | Ω in [{df['Omega_R'].min():.3f}, {df['Omega_R'].max():.3f}]"
          f" | Δ in [{df['detuning'].min():.3f}, {df['detuning'].max():.3f}]"
          f" | γ={sorted(df['gamma'].unique())}")
    print(f"Norma Bloch: min={norms.min():.6f}, max={norms.max():.6f}")

    return df


def save_dataset(df: pd.DataFrame, config: DatasetConfig, output_dir: Path) -> Tuple[Path, Path]:
    """Guarda dataset en CSV y un JSON con metadatos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{config.name}.csv"
    df.to_csv(csv_path, index=False)

    size_mb = csv_path.stat().st_size / (1024 * 1024)

    meta = {
        **asdict(config),
        "n_trajectories_actual": int(df["trajectory_id"].nunique()),
        "n_points_total": int(len(df)),
        "file_size_mb": round(size_mb, 2),
        "columns": list(df.columns),
        "stats": {
            "omega": {"min": float(df["Omega_R"].min()), "max": float(df["Omega_R"].max())},
            "delta": {"min": float(df["detuning"].min()), "max": float(df["detuning"].max())},
            "gamma": sorted([float(g) for g in df["gamma"].unique()]),
        },
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_path = output_dir / f"{config.name}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Guardado: {csv_path.name} ({size_mb:.1f} MB) + {meta_path.name}")
    return csv_path, meta_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera datasets sintéticos para el TFM (ecuaciones de Bloch).")
    parser.add_argument("--jobs", "-j", type=int, default=8, help="Procesos en paralelo (default: 8)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("datasets"), help="Directorio de salida")
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
        help="Datasets a generar (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Muestra configuración y termina")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if "all" in args.datasets:
        to_generate = list(DATASETS.keys())
    else:
        to_generate = args.datasets

    print("=" * 70)
    print("Generación de datasets (Bloch / Rabi)")
    print(f"Python {sys.version.split()[0]} | NumPy {np.__version__} | SciPy {__import__('scipy').__version__}")
    print(f"jobs={args.jobs} | output_dir={args.output_dir}")
    print(f"datasets={to_generate}")
    print("=" * 70)

    if args.dry_run:
        for name in to_generate:
            cfg = DATASETS[name]
            print(f"\n{name}")
            for k, v in asdict(cfg).items():
                print(f"  {k}: {v}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.time()
    generated = []

    for name in to_generate:
        cfg = DATASETS[name]

        # organización básica por partición
        if name.startswith("train"):
            out = args.output_dir / "train"
        elif name.startswith("val"):
            out = args.output_dir / "val"
        else:
            out = args.output_dir / "test"

        df = generate_dataset(cfg, n_jobs=args.jobs)
        csv_path, _ = save_dataset(df, cfg, out)
        generated.append((name, csv_path, len(df)))

    elapsed_total = time.time() - start_total
    total_points = sum(n for _, _, n in generated)
    total_size = sum(p.stat().st_size for _, p, _ in generated) / (1024 * 1024)

    print("\nResumen")
    print(f"Tiempo total: {elapsed_total/60:.1f} min")
    for name, path, n_points in generated:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {name}: {n_points:,} puntos | {size_mb:.1f} MB | {path}")
    print(f"Total: {total_points:,} puntos | {total_size:.1f} MB")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_minutes": round(elapsed_total / 60, 2),
        "datasets": {
            name: {"path": str(path), "n_points": int(n_points), "size_mb": round(path.stat().st_size / (1024 * 1024), 2)}
            for name, path, n_points in generated
        },
    }
    summary_path = args.output_dir / "generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Resumen guardado en: {summary_path}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
