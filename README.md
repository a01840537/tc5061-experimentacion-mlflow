# TC5061 — Experimentación base con MLflow
Predicción de inasistencia a citas médicas (no-shows) con regresión logística,
instrumentada con MLflow Tracking.

**Autor:** Rolando Calderón Carballo — A01840537

## Dataset
[Medical Appointment No Shows (Kaggle)](https://www.kaggle.com/datasets/joniarroba/noshowappointments) — 110,527 citas médicas de Brasil.
Colocar `KaggleV2-May-2016.csv` en la carpeta `data/` (no se versiona en el repo).

## Instalación
python -m venv .venv
.venv\Scripts\Activate.ps1 # Windows PowerShell
pip install -r requirements.txt


## Ejecución
Con valores por defecto (C=1.0, class_weight=none, max_iter=1000, seed=42):
python train.py

Variando hiperparámetros:
python train.py --C 0.01 --class-weight balanced


Argumentos disponibles (`python train.py --help`):
| Argumento | Default | Descripción |
|---|---|---|
| `--C` | 1.0 | Inversa de la fuerza de regularización |
| `--class-weight` | none | Peso de clases: `none` o `balanced` |
| `--max-iter` | 1000 | Iteraciones máximas del optimizador |
| `--test-size` | 0.2 | Proporción del set de prueba |
| `--seed` | 42 | Semilla aleatoria (reproducibilidad) |
| `--data` | data/KaggleV2-May-2016.csv | Ruta al CSV |

## Ver los experimentos
mlflow ui

Abrir http://localhost:5000 — experimento: `experimentacion-noshow-citas`.

## Estructura
- `train.py` — script de entrenamiento parametrizado
- `TC5061_Notebook_Baseline_CalderonRolando.ipynb` — exploración de datos (EDA)
- `requirements.txt` — dependencias con versiones fijadas

