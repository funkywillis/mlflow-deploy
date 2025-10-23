# src/train.py
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlflow.models import infer_signature
import sys
import traceback

# --- Configurar MLflow ---
# Usar rutas relativas para que funcione bien en GitHub Actions
mlflow.set_tracking_uri("file://" + os.path.join(os.getcwd(), "mlruns"))
experiment_name = "CI-CD-Lab2"
mlflow.set_experiment(experiment_name)

# --- Cargar Datos y Entrenar Modelo ---
try:
    df = pd.read_csv("train.csv")
    print("--- Debug: train.csv cargado correctamente. ---")
except FileNotFoundError:
    print("--- ERROR: No se encontró 'train.csv'. Asegúrate de que esté en la raíz del proyecto. ---")
    sys.exit(1)

# Determinar la columna objetivo (target)
# Para el dataset de seguros, el objetivo es 'charges'
possible_targets = ["charges", "target", "y", "label"]
target_col = next((col for col in possible_targets if col in df.columns), df.columns[-1])
print(f"--- Debug: Columna objetivo detectada: '{target_col}' ---")

# Preparar X e y
y = df[target_col]
X = df.drop(columns=[target_col])

# ¡IMPORTANTE! Convertir columnas categóricas a numéricas
X = pd.get_dummies(X, drop_first=True)

# Split de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# --- Iniciar Run de MLflow ---
with mlflow.start_run() as run:
    print(f"--- Debug: Run ID: {run.info.run_id} ---")

    # Log de métricas
    mlflow.log_metric("mse", mse)
    print(f"MSE: {mse:.4f}")

    # Inferir firma y registrar el modelo
    input_example = X_train.head(5)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    print("✅ Modelo entrenado y registrado en MLflow exitosamente.")