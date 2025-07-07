import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === Configuración de rutas ===
archivo_entrenamiento = "datos_entrenamiento.csv"
ruta_salida_modelo = "modelos/modelo_fallos.pkl"

# === Cargar datos ===
if not os.path.exists(archivo_entrenamiento):
    raise FileNotFoundError(f"❌ Archivo no encontrado: {archivo_entrenamiento}")

df_raw = pd.read_csv(archivo_entrenamiento, sep=';')

# === Validaciones básicas ===
if 'VarName' not in df_raw.columns or 'VarValue' not in df_raw.columns:
    raise ValueError("⚠️ El CSV debe contener columnas 'VarName' y 'VarValue'")

if 'Fallo' not in df_raw.columns:
    raise ValueError("⚠️ El CSV de entrenamiento debe tener una columna 'Fallo' con 0 o 1")

# === Asegurar que VarValue es numérico ===
df_raw["VarValue"] = df_raw["VarValue"].astype(str).str.replace(",", ".").astype(float)

# === Pivotear datos (Time_ms como índice, columnas = VarName, valores = VarValue) ===
df_pivot = df_raw.pivot_table(
    index="Time_ms",
    columns="VarName",
    values="VarValue",
    aggfunc="mean"
).reset_index()

# Agregar la columna 'Fallo' al DataFrame pivoteado
df_fallos = df_raw[["Time_ms", "Fallo"]].drop_duplicates(subset="Time_ms")
df_final = pd.merge(df_pivot, df_fallos, on="Time_ms", how="inner")

# Separar features y etiquetas
X = df_final.drop(columns=["Time_ms", "Fallo"])
y = df_final["Fallo"]

# === División entrenamiento/prueba ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Entrenar modelo ===
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# === Evaluar ===
y_pred = modelo.predict(X_test)
print("=== Resultados del modelo ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# === Guardar modelo ===
os.makedirs(os.path.dirname(ruta_salida_modelo), exist_ok=True)
joblib.dump(modelo, ruta_salida_modelo)
print(f"✅ Modelo guardado en: {ruta_salida_modelo}")

# === Mostrar importancia de variables ===
importancias = pd.Series(modelo.feature_importances_, index=X.columns)
importancias = importancias.sort_values(ascending=False)
print("\n🔍 Variables más importantes:")
print(importancias.head(10))
