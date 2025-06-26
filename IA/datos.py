import pandas as pd
import sqlite3
import os
import chardet
import hashlib
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from typing import Optional


DB_PATH = "data/memoria.db"
PROCESADOS_TABLA = "archivos_procesados"


# --- UTILIDADES ---
def conectar_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def calcular_hash(ruta: str) -> Optional[str]:
    try:
        hasher = hashlib.sha256()
        with open(ruta, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"❌ Error al calcular hash de {ruta}: {e}")
        return None


# --- BASE DE DATOS ---
def crear_tablas():
    """Crear todas las tablas necesarias si no existen."""
    with conectar_db() as conn:
        conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS estadisticas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tabla TEXT NOT NULL,
            columna TEXT NOT NULL,
            promedio REAL, minimo REAL, maximo REAL, desviacion REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS valores_unicos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tabla TEXT NOT NULL,
            columna TEXT NOT NULL,
            valor TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS historial_consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo_consulta TEXT NOT NULL,
            parametros TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS historial_entrenamientos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            registros_entrenados INTEGER,
            accuracy REAL,
            f1_macro REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS {PROCESADOS_TABLA} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_archivo TEXT NOT NULL,
            hash_sha256 TEXT NOT NULL UNIQUE,
            registros INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_hash ON {PROCESADOS_TABLA}(hash_sha256);
        CREATE INDEX IF NOT EXISTS idx_nombre ON {PROCESADOS_TABLA}(nombre_archivo);
        """)
        conn.commit()


# --- ARCHIVOS ---
def seleccionar_archivo() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            title="Seleccione el archivo CSV",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        ) or None
    finally:
        root.destroy()


def registrar_archivo(ruta: str, df: pd.DataFrame) -> bool:
    try:
        hash_archivo = calcular_hash(ruta)
        if not hash_archivo:
            return False
        with conectar_db() as conn:
            conn.execute(
                f"INSERT INTO {PROCESADOS_TABLA} (nombre_archivo, hash_sha256, registros) VALUES (?, ?, ?)",
                (os.path.basename(ruta), hash_archivo, len(df))
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        print(f"⏭️ Archivo ya registrado: {os.path.basename(ruta)}")
        return False
    except Exception as e:
        print(f"❌ Error al registrar archivo: {e}")
        return False


# --- LECTURA CSV ---
def detectar_codificacion(ruta: str) -> str:
    with open(ruta, 'rb') as f:
        datos = f.read(10000)
        resultado = chardet.detect(datos)
        return resultado.get("encoding") or "utf-8"


def leer_csv(ruta: str) -> pd.DataFrame:
    try:
        encoding = detectar_codificacion(ruta)
        df = pd.read_csv(
            ruta,
            sep=';',
            quotechar='"',
            encoding=encoding,
            parse_dates=True,
            dayfirst=True,
            on_bad_lines='skip',
            dtype={'VarValue': str}
        )
        if 'VarValue' in df.columns:
            df['VarValue'] = df['VarValue'].str.replace(',', '.').astype(float, errors='ignore')
        return df.dropna(subset=['VarName', 'VarValue'])
    except Exception as e:
        print(f"❌ Error al leer CSV {ruta}: {e}")
        return pd.DataFrame()


def cargar_csv(ruta1: str, ruta2: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(ruta1):
        print(f"❌ No existe: {ruta1}")
        return pd.DataFrame()


    df1 = leer_csv(ruta1)
    df2 = pd.DataFrame()


    if ruta2 and os.path.exists(ruta2):
        df2 = leer_csv(ruta2)


    df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=['VarName', 'TimeString', 'VarValue'])


    if not df.empty:
        registrar_archivo(ruta1, df1)
        if not df2.empty:
            registrar_archivo(ruta2, df2)
    else:
        print("🔴 Nada nuevo para cargar.")


    return df




# --- GUARDAR DATOS EN DB ---
def guardar_en_db(df: pd.DataFrame, tabla: str) -> bool:
    try:
        with conectar_db() as conn:
            tablas = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tabla not in tablas:
                df.to_sql(tabla, conn, index=False)
            else:
                df.to_sql(f"temp_{tabla}", conn, index=False, if_exists="replace")
                conn.execute(f"""
                    INSERT OR IGNORE INTO {tabla}
                    SELECT * FROM temp_{tabla}
                """)
            conn.commit()
            return True
    except Exception as e:
        print(f"❌ Error al guardar datos en {tabla}: {e}")
        return False


# --- REGISTRAR CONSULTAS ---
def registrar_consulta(tipo: str, parametros: dict) -> bool:
    try:
        with conectar_db() as conn:
            conn.execute(
                "INSERT INTO historial_consultas (tipo_consulta, parametros) VALUES (?, ?)",
                (tipo, str(parametros)[:1500])
            )
            conn.commit()
        return True
    except Exception as e:
        print(f"❌ Error al registrar consulta: {e}")
        return False
