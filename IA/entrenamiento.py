import pandas as pd
import os
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib


# Rutas
ruta_modelo = "modelos/modelo_fallos.pkl"
archivo_base = "datos_locomotoras.csv"
archivo_nuevos = "nuevos_registros.csv"
ruta_db = "data/memoria.db"


def cargar_y_actualizar():
    if os.path.exists(archivo_base):
        df_base = pd.read_csv(archivo_base)
    else:
        df_base = pd.DataFrame()


    if os.path.exists(archivo_nuevos):
        df_nuevos = pd.read_csv(archivo_nuevos)


        if not df_base.empty:
            df_total = pd.concat([df_base, df_nuevos]).drop_duplicates()
        else:
            df_total = df_nuevos


        df_total.to_csv(archivo_base, index=False)
        os.remove(archivo_nuevos)
        print("✅ Nuevos datos incorporados y 'nuevos_registros.csv' eliminado.")
    else:
        df_total = df_base


    if df_total.empty:
        raise Exception("⚠️ No hay datos disponibles para entrenar.")
   
    return df_total


def entrenar_y_guardar_modelo(df):
    if "fails" not in df.columns:
        raise Exception("Falta la columna 'fails' en el dataset.")
   
    X = df.drop(columns=["fails"])
    y = df["fails"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")


    print("📊 Evaluación del modelo:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


    os.makedirs("modelos", exist_ok=True)
    joblib.dump(model, ruta_modelo)
    print(f"💾 Modelo guardado en: {ruta_modelo}")


    # Guardar estadísticas de entrenamiento
    with sqlite3.connect(ruta_db) as conn:
        conn.execute("""
            INSERT INTO historial_entrenamientos (registros_entrenados, accuracy, f1_macro)
            VALUES (?, ?, ?)
        """, (len(df), acc, f1))
        conn.commit()


if __name__ == "__main__":
    df = cargar_y_actualizar()
    entrenar_y_guardar_modelo(df)