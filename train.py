"""
TC5061 - Actividad 3: Experimentación base con MLflow
Script de entrenamiento parametrizado para el modelo baseline de no-shows.
Autor: Rolando Calderón Carballo - A01840537
"""

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: solo guarda archivos
import matplotlib.pyplot as plt


def cargar_datos(ruta_csv):
    """Carga el CSV y devuelve el DataFrame crudo."""
    df = pd.read_csv(
        ruta_csv,
        parse_dates=["ScheduledDay", "AppointmentDay"],
    )
    return df


def limpiar_datos(df):
    """Aplica la limpieza definida en el notebook y devuelve el DataFrame listo."""
    df = df.copy()

    # Variable derivada: días de espera entre agendamiento y cita
    df["DiasEspera"] = (
        df["AppointmentDay"].dt.normalize() - df["ScheduledDay"].dt.normalize()
    ).dt.days

    # Variable derivada: día de la semana de la cita
    df["DiaSemanaCita"] = df["AppointmentDay"].dt.day_name()

    # Eliminar registros inválidos (edad negativa, espera negativa)
    df = df[df["Age"] >= 0]
    df = df[df["DiasEspera"] >= 0]

    # Eliminar identificadores sin valor predictivo
    df = df.drop(columns=["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"])

    # Handcap como categórica
    df["Handcap"] = df["Handcap"].astype("category")

    # Variable objetivo binaria: 1 = no asistió
    df["NoShow"] = (df["No-show"] == "Yes").astype(int)
    df = df.drop(columns=["No-show"])

    return df


def construir_pipeline(C, class_weight, max_iter):
    """Construye el preprocesador + LogisticRegression con los hiperparámetros dados."""
    columnas_num = ["Age", "DiasEspera"]
    columnas_cat = ["Gender", "Neighbourhood", "Handcap", "DiaSemanaCita"]
    columnas_bin = ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "SMS_received"]

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), columnas_num),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), columnas_cat),
            ("bin", "passthrough", columnas_bin),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocesamiento", preproc),
            ("clasificador", LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=max_iter,
            )),
        ]
    )
    return modelo


def evaluar_modelo(modelo, X_test, y_test):
    """Calcula métricas y guarda la matriz de confusión; devuelve (metricas, ruta_png)."""
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="black", fontsize=12)
            
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Asiste", "No asiste"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Asiste", "No asiste"])
    ax.set_xlabel("Predicción"); ax.set_ylabel("Realidad")
    ax.set_title("Matriz de confusión (test)")
    fig.tight_layout()

    ruta_png = "confusion_matrix.png"
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)

    return metricas, ruta_png


def main():
    # ---------- 1. Argumentos de línea de comandos ----------
    parser = argparse.ArgumentParser(
        description="Entrena el modelo baseline de no-shows y registra la corrida en MLflow."
    )
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inversa de la fuerza de regularización (default: 1.0)")
    parser.add_argument("--class-weight", type=str, default="none",
                        choices=["none", "balanced"],
                        help="Peso de clases: 'none' o 'balanced' (default: none)")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="Iteraciones máximas del optimizador (default: 1000)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proporción del set de prueba (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad (default: 42)")
    parser.add_argument("--data", type=str, default="data/KaggleV2-May-2016.csv",
                        help="Ruta al CSV de citas (default: data/KaggleV2-May-2016.csv)")
    args = parser.parse_args()

    # argparse no acepta None directamente; lo traducimos aquí
    class_weight = None if args.class_weight == "none" else "balanced"

    # ---------- 2. Datos ----------
    df = limpiar_datos(cargar_datos(args.data))
    X = df.drop(columns=["NoShow"])
    y = df["NoShow"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # ---------- 3. Entrenamiento ----------
    modelo = construir_pipeline(args.C, class_weight, args.max_iter)
    modelo.fit(X_train, y_train)

    # ---------- 4. Evaluación ----------
    metricas, ruta_png = evaluar_modelo(modelo, X_test, y_test)

    # ---------- 5. Registro en MLflow ----------
    mlflow.set_experiment("experimentacion-noshow-citas")
    nombre_run = f"logreg_C{args.C}_cw-{args.class_weight}"
    with mlflow.start_run(run_name=nombre_run):
        
        mlflow.log_param("modelo", "LogisticRegression")
        mlflow.log_param("C", args.C)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("registros_entrenamiento", len(X_train))

        mlflow.log_metrics(metricas)
        mlflow.log_artifact(ruta_png)
        mlflow.sklearn.log_model(modelo, name="modelo")

    # ---------- 6. Resumen en consola ----------
    print(f"Run: {nombre_run}")
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor:.4f}")


if __name__ == "__main__":
    main()
    
