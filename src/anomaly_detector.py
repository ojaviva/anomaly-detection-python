"""
Anomaly Detection Pipeline
==========================
Detección de anomalías usando Isolation Forest y métodos estadísticos.
Aplicable a: detección de fraude, sensores IoT, análisis de logs.

Autor: [Tu Nombre]
LinkedIn: [Tu LinkedIn]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


# =============================================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# =============================================================================

def generate_synthetic_data(
    n_normal: int = 1000,
    n_anomalies: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Genera un dataset sintético que simula transacciones financieras.

    Parámetros
    ----------
    n_normal : int
        Cantidad de transacciones normales.
    n_anomalies : int
        Cantidad de transacciones anómalas (fraude).
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    pd.DataFrame con columnas: amount, frequency, hour, is_anomaly
    """
    rng = np.random.RandomState(random_state)

    # Transacciones normales
    normal = pd.DataFrame({
        "amount": rng.normal(loc=500, scale=150, size=n_normal),
        "frequency": rng.poisson(lam=5, size=n_normal),
        "hour": rng.choice(range(8, 22), size=n_normal),
        "is_anomaly": 0,
    })

    # Transacciones anómalas (montos altos, horarios inusuales, alta frecuencia)
    anomalies = pd.DataFrame({
        "amount": rng.normal(loc=3000, scale=800, size=n_anomalies),
        "frequency": rng.poisson(lam=25, size=n_anomalies),
        "hour": rng.choice([0, 1, 2, 3, 4, 5, 23], size=n_anomalies),
        "is_anomaly": 1,
    })

    df = pd.concat([normal, anomalies], ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return df


# =============================================================================
# 2. PREPROCESAMIENTO
# =============================================================================

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Escala las features numéricas con StandardScaler.

    Retorna
    -------
    (df_scaled, scaler)
    """
    features = ["amount", "frequency", "hour"]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled, scaler


# =============================================================================
# 3. ENTRENAMIENTO DEL MODELO
# =============================================================================

def train_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    """
    Entrena un Isolation Forest para detectar anomalías.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset con features escaladas.
    contamination : float
        Proporción esperada de anomalías (0.0 a 0.5).
    random_state : int
        Semilla para reproducibilidad.
    """
    features = ["amount", "frequency", "hour"]
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(df[features])
    return model


# =============================================================================
# 4. DETECCIÓN Y EVALUACIÓN
# =============================================================================

def detect_anomalies(
    model: IsolationForest,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ejecuta predicciones y calcula anomaly scores.

    Retorna
    -------
    DataFrame con columnas adicionales: prediction, anomaly_score
    """
    features = ["amount", "frequency", "hour"]
    df = df.copy()
    df["prediction"] = model.predict(df[features])
    df["prediction"] = df["prediction"].map({1: 0, -1: 1})  # 1 = anomalía
    df["anomaly_score"] = model.decision_function(df[features])
    return df


def evaluate(df: pd.DataFrame) -> dict:
    """
    Evalúa el rendimiento del modelo comparando prediction vs is_anomaly.
    """
    report = classification_report(
        df["is_anomaly"], df["prediction"], output_dict=True
    )
    cm = confusion_matrix(df["is_anomaly"], df["prediction"])

    print("\n" + "=" * 50)
    print("REPORTE DE CLASIFICACIÓN")
    print("=" * 50)
    print(classification_report(df["is_anomaly"], df["prediction"]))
    print("Matriz de Confusión:")
    print(cm)
    print("=" * 50)

    return {"report": report, "confusion_matrix": cm}


# =============================================================================
# 5. VISUALIZACIONES
# =============================================================================

def plot_scatter(df: pd.DataFrame, save_path: str = "outputs/scatter_anomalies.png"):
    """Scatter plot: monto vs frecuencia, coloreado por anomalía detectada."""
    fig, ax = plt.subplots(figsize=(10, 6))
    normal = df[df["prediction"] == 0]
    anomaly = df[df["prediction"] == 1]

    ax.scatter(
        normal["amount"], normal["frequency"],
        c="#1D9E75", alpha=0.5, s=30, label="Normal", edgecolors="none",
    )
    ax.scatter(
        anomaly["amount"], anomaly["frequency"],
        c="#E24B4A", alpha=0.8, s=80, label="Anomalía", edgecolors="#991D1D",
        linewidths=1, marker="X",
    )
    ax.set_xlabel("Monto de transacción ($)", fontsize=12)
    ax.set_ylabel("Frecuencia de transacciones", fontsize=12)
    ax.set_title("Detección de anomalías — Monto vs Frecuencia", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Scatter guardado en {save_path}")


def plot_anomaly_scores(df: pd.DataFrame, save_path: str = "outputs/anomaly_scores.png"):
    """Distribución de anomaly scores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    normal = df[df["is_anomaly"] == 0]["anomaly_score"]
    anomaly = df[df["is_anomaly"] == 1]["anomaly_score"]

    ax.hist(normal, bins=40, alpha=0.6, color="#1D9E75", label="Normal", edgecolor="white")
    ax.hist(anomaly, bins=15, alpha=0.8, color="#E24B4A", label="Anomalía", edgecolor="white")
    ax.axvline(x=0, color="#BA7517", linestyle="--", linewidth=1.5, label="Threshold (0)")
    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.set_title("Distribución de Anomaly Scores", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Scores guardado en {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str = "outputs/confusion_matrix.png"):
    """Heatmap de la matriz de confusión."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdYlGn_r",
        xticklabels=["Normal", "Anomalía"],
        yticklabels=["Normal", "Anomalía"],
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title("Matriz de Confusión", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Confusion matrix guardada en {save_path}")


def plot_time_series(df: pd.DataFrame, save_path: str = "outputs/time_series.png"):
    """Simula una serie temporal con anomalías marcadas."""
    fig, ax = plt.subplots(figsize=(12, 5))
    df_sorted = df.reset_index(drop=True)
    normal_idx = df_sorted[df_sorted["prediction"] == 0].index
    anomaly_idx = df_sorted[df_sorted["prediction"] == 1].index

    ax.plot(df_sorted.index, df_sorted["amount"], color="#888780", alpha=0.4, linewidth=0.8)
    ax.scatter(normal_idx, df_sorted.loc[normal_idx, "amount"],
               c="#1D9E75", s=10, alpha=0.5, label="Normal")
    ax.scatter(anomaly_idx, df_sorted.loc[anomaly_idx, "amount"],
               c="#E24B4A", s=60, alpha=0.9, zorder=5, marker="X", label="Anomalía")
    ax.set_xlabel("Índice de transacción", fontsize=12)
    ax.set_ylabel("Monto ($)", fontsize=12)
    ax.set_title("Serie temporal de transacciones con anomalías detectadas", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Time series guardado en {save_path}")


# =============================================================================
# 6. PIPELINE PRINCIPAL
# =============================================================================

def main():
    print("🔍 Anomaly Detection Pipeline")
    print("=" * 50)

    # Paso 1: Generar datos
    print("\n[1/5] Generando datos sintéticos...")
    df = generate_synthetic_data(n_normal=1000, n_anomalies=50)
    print(f"      → {len(df)} registros ({df['is_anomaly'].sum()} anomalías reales)")

    # Paso 2: Preprocesar
    print("[2/5] Preprocesando features...")
    df_scaled, scaler = preprocess(df)

    # Paso 3: Entrenar
    print("[3/5] Entrenando Isolation Forest...")
    model = train_isolation_forest(df_scaled, contamination=0.05)

    # Paso 4: Detectar
    print("[4/5] Detectando anomalías...")
    df_results = detect_anomalies(model, df_scaled)

    # Copiar predicciones al df original (sin escalar) para visualización
    df["prediction"] = df_results["prediction"]
    df["anomaly_score"] = df_results["anomaly_score"]

    # Paso 5: Evaluar
    print("[5/5] Evaluando resultados...")
    metrics = evaluate(df)

    # Generar visualizaciones
    import os
    os.makedirs("outputs", exist_ok=True)

    print("\n📊 Generando visualizaciones...")
    plot_scatter(df)
    plot_anomaly_scores(df)
    plot_confusion_matrix(metrics["confusion_matrix"])
    plot_time_series(df)

    print("\n✅ Pipeline completado. Revisa la carpeta outputs/")
    return df, model, metrics


if __name__ == "__main__":
    df, model, metrics = main()
