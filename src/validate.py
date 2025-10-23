# src/validate.py
import os
import sys
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error

# Parámetro de umbral de calidad del modelo (MSE)
# El MSE para los costos de seguro puede ser alto, ajustamos el umbral a un valor razonable.
# Un buen modelo en este dataset tiene un MSE ~36,000,000. Ponemos un umbral flexible.
THRESHOLD = 40000000.0

# --- Configuración de MLflow para leer el experimento ---
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

# --- Cargar datos de validación ---
try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("--- ERROR: No se encontró 'train.csv'. ---")
    sys.exit(1)

# --- Preparar Datos (debe ser idéntico al preprocesamiento en train.py) ---
target_col = "charges" 
y = df[target_col]
X = df.drop(columns=[target_col])
X = pd.get_dummies(X, drop_first=True)

# --- Cargar el Modelo desde el Run más reciente de MLflow ---
try:
    experiment_name = "CI-CD-Lab2"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception(f"Experimento '{experiment_name}' no encontrado.")

    # Buscar el último run exitoso
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    if len(runs) == 0:
        raise Exception("No se encontraron runs en el experimento.")

    latest_run_id = runs.iloc[0]["run_id"]
    print(f"--- Debug: Cargando modelo del Run ID: {latest_run_id} ---")

    # Cargar el modelo usando su URI de MLflow
    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

except Exception as e:
    print(f"--- ERROR: No se pudo cargar el modelo desde MLflow: {e} ---")
    sys.exit(1)

# --- Alinear Columnas por si hay diferencias entre datasets ---
model_feature_names = model.feature_names_in_
X = X.reindex(columns=model_feature_names, fill_value=0)

# --- Predicción y Validación ---
print("--- Debug: Realizando predicciones... ---")
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
print(f"🔍 MSE del modelo: {mse:.4f} (Umbral aceptable: {THRESHOLD})")

# Validación final
if mse <= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)  # Éxito
else:
    print("❌ El modelo NO CUMPLE el umbral de calidad. Pipeline detenido.")
    sys.exit(1)  # Error