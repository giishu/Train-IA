import pandas as pd
import joblib
import os


# === Configuración ===
ruta_modelo = "modelos/modelo_fallos.pkl"
archivo_datos_nuevos = "datos_nuevos.csv"
archivo_salida = "predicciones_fallos.csv"


# === Cargar el modelo ===
if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"❌ Modelo no encontrado en: {ruta_modelo}")


model = joblib.load(ruta_modelo)


# === Cargar los nuevos datos ===
if not os.path.exists(archivo_datos_nuevos):
    raise FileNotFoundError(f"❌ Archivo de datos nuevos no encontrado: {archivo_datos_nuevos}")


df_raw = pd.read_csv(archivo_datos_nuevos, sep=';')


# === Preprocesamiento ===
# Asegurarse de que VarValue es numérico
df_raw["VarValue"] = df_raw["VarValue"].astype(str).str.replace(",", ".").astype(float)


# Convertir a tabla pivote estilo: Time_ms como índice, columnas = VarName, valores = VarValue
df_pivot = df_raw.pivot_table(
    index="Time_ms",
    columns="VarName",
    values="VarValue",
    aggfunc="mean"
).reset_index()


# Eliminar columna de índice si no se necesita en el modelo
if "Time_ms" in df_pivot.columns:
    df_features = df_pivot.drop(columns=["Time_ms"])
else:
    df_features = df_pivot


# Verificar columnas esperadas por el modelo
modelo_columnas = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None


if modelo_columnas is not None:
    faltantes = set(modelo_columnas) - set(df_features.columns)
    if faltantes:
        raise ValueError(f"❌ Faltan columnas en los datos: {faltantes}")
    # Reordenar las columnas para que coincidan
    df_features = df_features[modelo_columnas]


# === Realizar predicciones ===
predicciones = model.predict(df_features)


# Agregar predicciones a la tabla original pivoteada
df_pivot["Prediccion_Fallo"] = predicciones


# Guardar a archivo
df_pivot.to_csv(archivo_salida, index=False)
print(f"✅ Predicciones guardadas en: {archivo_salida}")


# Mostrar resultados
for idx, fila in df_pivot.iterrows():
    status = "⚠️ Fallo" if fila["Prediccion_Fallo"] == 1 else "✅ Sin fallo"
    print(f"[{idx}] {status}")
