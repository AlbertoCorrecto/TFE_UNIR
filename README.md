# Neural ODE-P vs PINODE (Bloch / Rabi) — Código reproducible del TFM

Este repositorio contiene el **código mínimo** para reproducir los experimentos del TFM: generación de datos a partir de las **ecuaciones ópticas de Bloch** y entrenamiento/evaluación de los paradigmas **Neural ODE**, **PINODE** y **Neural ODE-P** (Neural ODE con regularización/supervisión física).

> Nota: el objetivo del repositorio es **reproducibilidad científica**, no servir como framework general.

---

## Contenido

- Generación de datos mediante muestreo **Sobol / QMC**
- Modelos:
  - Neural ODE (baseline data-driven)
  - PINODE (modelo híbrido con física incompleta)
  - Neural ODE-P (física como término de pérdida)
- Entrenamiento, evaluación y scripts de visualización

---

## Estructura del proyecto

```
data/
 └ generate_unified_data.py
models/
 ├ neural_ode.py
 ├ neural_ode_physics.py
 ├ pinode_v3_optimized.py
 ├ pinn.py
 └ pinn_fourier.py
trainers/
 ├ train.py
 ├ train_neural_ode_physics.py
 ├ train_pinode_v3.py
 └ train_pinn_fourier.py
evaluate/
 └ evaluate_model.py
experiments/
 ├ visualize_comparison.py
 ├ visualize_all.py
 └ long_horizon_eval_plots.py
```


---

## Generación de datos

```bash
python data/generate_unified_data.py --jobs 8 --output-dir datasets --datasets all
```

Modo inspección:

```bash
python data/generate_unified_data.py --dry-run
```

---

## Entrenamiento

### Neural ODE-P (loss física)

```bash
python trainers/train_neural_ode_physics.py   --data-dir datasets   --loss-mode physics   --physics-target complete
```

Modo híbrido:

```bash
python trainers/train_neural_ode_physics.py   --loss-mode hybrid   --lambda-physics 1.0   --lambda-data 1.0
```

---

## Evaluación

```bash
python evaluate/evaluate_model_v2.py   --data-dir datasets   --split test_id
```

---

## Reproducibilidad

- Semillas fijas por split
- Configuración guardada en cada ejecución
- Scripts mínimos incluidos para replicar resultados

---

## Autor

Alberto José Vidal Fernández  
TFM (2025)
