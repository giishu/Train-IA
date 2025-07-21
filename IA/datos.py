import pandas as pd
import sqlite3
import os
import chardet
import hashlib
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from typing import Optional, Dict

DB_PATH = "data/memoria.db"
PROCESADOS_TABLA = "archivos_procesados"
CLASIFICACION_TABLA = "clasificaciones"

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

def crear_tablas():
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
        CREATE TABLE IF NOT EXISTS {CLASIFICACION_TABLA} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            locomotora TEXT NOT NULL,
            var_name TEXT NOT NULL,
            var_value REAL,
            time_string TEXT,
            time_ms REAL,
            estado TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_hash ON {PROCESADOS_TABLA}(hash_sha256);
        CREATE INDEX IF NOT EXISTS idx_nombre ON {PROCESADOS_TABLA}(nombre_archivo);
        CREATE INDEX IF NOT EXISTS idx_clasificacion ON {CLASIFICACION_TABLA}(locomotora, var_name, time_ms);
        """)
        conn.commit()

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

def detectar_codificacion(ruta: str) -> str:
    with open(ruta, 'rb') as f:
        datos = f.read(10000)
        resultado = chardet.detect(datos)
        return resultado.get("encoding") or "utf-8"

def leer_csv(ruta: str) -> pd.DataFrame:
    try:
        encoding = detectar_codificacion(ruta)
        # Intentar leer como CSV estándar
        try:
            df = pd.read_csv(
                ruta,
                sep=';',
                quotechar='"',
                encoding=encoding,
                parse_dates=['TimeString'],
                dayfirst=True,
                on_bad_lines='skip',
                dtype={'VarValue': str}
            )
        except:
            # Intentar leer como una sola línea
            with open(ruta, 'r', encoding=encoding) as f:
                lineas = f.readlines()
            if len(lineas) == 1:
                datos = lineas[0].strip().split(';')
                columnas = ['VarName', 'TimeString', 'VarValue', 'Validity', 'Time_ms']
                datos_parseados = [datos[i:i+len(columnas)] for i in range(0, len(datos), len(columnas))]
                df = pd.DataFrame(datos_parseados, columns=columnas[:len(datos_parseados[0])])
            else:
                return pd.DataFrame()
        
        if 'VarValue' in df.columns:
            df['VarValue'] = pd.to_numeric(df['VarValue'].str.replace(',', '.'), errors='ignore')
        if 'Time_ms' in df.columns:
            df['Time_ms'] = pd.to_numeric(df['Time_ms'].str.replace(',', '.'), errors='ignore')
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

def guardar_clasificaciones(df: pd.DataFrame, locomotora: str) -> bool:
    try:
        with conectar_db() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    f"INSERT INTO {CLASIFICACION_TABLA} (locomotora, var_name, var_value, time_string, time_ms, estado) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        locomotora,
                        row['VarName'] if 'VarName' in row else 'Unknown',
                        row['VarValue'] if 'VarValue' in row else None,
                        row['TimeString'] if 'TimeString' in row else None,
                        row['Time_ms'] if 'Time_ms' in row else None,
                        row['estado'] if 'estado' in row else 'SIN_CLASIFICAR'
                    )
                )
            conn.commit()
        return True
    except Exception as e:
        print(f"❌ Error al guardar clasificaciones: {e}")
        return False

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

def cargar_limites(tipo_locomotora: str = "ALCO") -> Dict[str, Dict[str, float]]:
    ruta_archivo = os.path.join(os.path.dirname(__file__), "..", "data", "Clasificación Variables LOGs IA (1) - Hoja1.csv")
    if not os.path.exists(ruta_archivo):
        print(f"❌ Archivo de límites no encontrado: {ruta_archivo}")
        return {}
    df = pd.read_csv(ruta_archivo, sep=",", skiprows=1)
    columnas = {
        "ALCO": ("Mínimo", "Máximo", "Alerta"),
        "GAIA": ("Mínimo.1", "Máximo.1", "Alerta.1"),
        "GR12": ("Mínimo.2", "Máximo.2", "Alerta.2"),
        "GT22": ("Mínimo.3", "Máximo.3", "Alerta.3"),
    }
    if tipo_locomotora not in columnas:
        print(f"⚠️ Tipo de locomotora no válido: {tipo_locomotora}. Usando ALCO por defecto.")
        tipo_locomotora = "ALCO"
    min_col, max_col, alert_col = columnas[tipo_locomotora]
    limites = {}
    for _, row in df.iterrows():
        var_name = str(row.get("Variable", "")).strip().upper()
        try:
            minimo = float(str(row.get(min_col, "")).replace(",", ".")) if pd.notna(row.get(min_col)) and str(row.get(min_col, "")).strip() else None
            maximo = float(str(row.get(max_col, "")).replace(",", ".")) if pd.notna(row.get(max_col)) and str(row.get(max_col, "")).strip() else None
            alerta = float(str(row.get(alert_col, "")).replace(",", ".")) if pd.notna(row.get(alert_col)) and str(row.get(alert_col, "")).strip() else None
            if minimo is not None and maximo is not None:
                limites[var_name] = {"min": minimo, "max": maximo, "alerta": alerta}
        except (ValueError, TypeError):
            continue
    return limites

def comparar_periodos(locomotora: str, variable: str, periodo1: tuple, periodo2: tuple) -> dict:
    try:
        with conectar_db() as conn:
            inicio1, fin1 = pd.to_datetime(periodo1, format="%d.%m.%Y %H:%M:%S")
            inicio2, fin2 = pd.to_datetime(periodo2, format="%d.%m.%Y %H:%M:%S")
            query = f"""
                SELECT VarName, VarValue, TimeString, Time_ms
                FROM {CLASIFICACION_TABLA}
                WHERE locomotora = ? AND VarName = ?
                AND TimeString BETWEEN ? AND ?
            """
            df1 = pd.read_sql_query(query, conn, params=(locomotora, variable, inicio1, fin1))
            df2 = pd.read_sql_query(query, conn, params=(locomotora, variable, inicio2, fin2))
            stats = {
                'periodo1': {
                    'promedio': df1['VarValue'].mean() if not df1.empty else None,
                    'maximo': df1['VarValue'].max() if not df1.empty else None,
                    'minimo': df1['VarValue'].min() if not df1.empty else None,
                    'registros': len(df1)
                },
                'periodo2': {
                    'promedio': df2['VarValue'].mean() if not df2.empty else None,
                    'maximo': df2['VarValue'].max() if not df2.empty else None,
                    'minimo': df2['VarValue'].min() if not df2.empty else None,
                    'registros': len(df2)
                }
            }
            return stats
    except Exception as e:
        print(f"❌ Error al comparar períodos: {str(e)}")
        return {}
    
def cargar_csv_streamlit(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=';', quotechar='"', dayfirst=True, on_bad_lines='skip')
    if 'VarValue' in df.columns:
        df['VarValue'] = pd.to_numeric(df['VarValue'], errors='coerce')
    df['TimeString'] = pd.to_datetime(df['TimeString'], errors='coerce')
    return df.dropna(subset=['VarName', 'VarValue'])
