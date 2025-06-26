from IA.datos import cargar_csv, seleccionar_archivo, registrar_consulta  
from IA.analisis import detectar_cambios_porcentuales
from IA.ia import consultar_bot
import pandas as pd
import sqlite3
import tkinter as tk
from tkinter import filedialog
import time
from IA.ia import bot
import os


def seleccionar_locomotora():
    opciones = ["LOCOMOTORA 1", "LOCOMOTORA 2", "LOCOMOTORA 3", "LOCOMOTORA 4"]
    print("Seleccione el tipo de locomotora:")
    for i, opcion in enumerate(opciones, 1):
        print(f"{i}. {opcion}")
    while True:
        try:
            seleccion = int(input("Ingrese el número de la locomotora (1-4): "))
            if 1 <= seleccion <= 4:
                locomotora = opciones[seleccion - 1]
                print(f"Has seleccionado: {locomotora}")
                return locomotora
            else:
                print("Por favor, ingrese un número válido entre 1 y 4.")
        except ValueError:
            print("Entrada no válida. Ingrese un número del 1 al 4.")


# Elección de locomotora al inicio
locomotora_seleccionada = seleccionar_locomotora()


def seleccionar_archivo_manual():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    initial_dir = "data" if os.path.exists("data") else None
    try:
        archivos = filedialog.askopenfilenames(
            title="Seleccione 1 o 2 archivos CSV (mantenga CTRL para seleccionar dos)",
            filetypes=[("Archivos CSV", "*.csv")],
            initialdir=initial_dir
        )
        if not archivos:
            print("\n⚠️ No se seleccionaron archivos. Usando archivo por defecto.")
            default = "data/LOG ENTRADAS Y SALIDAS FISICAS0.csv" if initial_dir else "LOG ENTRADAS Y SALIDAS FISICAS0.csv"
            return default, None
        return (archivos[0], archivos[1] if len(archivos) > 1 else None)
    except Exception as e:
        print(f"Error al seleccionar archivos: {str(e)}")
        return None, None
    finally:
        root.destroy()




def menu_carga() -> pd.DataFrame:
    print("\n--- CARGAR DATOS DE LOCOMOTORAS ---")
    print("Opciones de carga:")
    print("1. Cargar archivo(s) manualmente")
    print("2. Usar archivo por defecto")
    print("3. Salir")


    opcion = input("Seleccione (1-3): ").strip()
   
    if opcion == "1":
        ruta1, ruta2 = seleccionar_archivo_manual()
        if not ruta1:
            print("⚠️ No se seleccionó archivo. Usando archivo por defecto.")
            ruta1 = "data/LOG ENTRADAS Y SALIDAS FISICAS0.csv"
        df = cargar_csv(ruta1, ruta2)
    elif opcion == "2":
        ruta1 = "data/LOG ENTRADAS Y SALIDAS FISICAS0.csv"
        df = cargar_csv(ruta1)
    elif opcion == "3":
        print("👋 Saliendo del programa.")
        exit()
    else:
        print("⚠️ Opción inválida. Usando archivo por defecto.")
        ruta1 = "data/LOG ENTRADAS Y SALIDAS FISICAS0.csv"
        df = cargar_csv(ruta1)


    if df.empty:
        print("🔴 No se cargaron datos nuevos (¿archivo duplicado?).")
    else:
        print(f"✅ Datos cargados: {len(df)} registros nuevos.")


    return df


def mostrar_menu():
    print("\n--- MENÚ DE CONSULTAS ---")
    print("1. Consultar IA")
    print("2. Ver primeros N datos")
    print("3. Filtrar por intervalo de tiempo")
    print("4. Detectar cambios bruscos")
    print("5. Ver historial")
    print("6. Salir")


def ver_primeros_n(df):
    try:
        n = int(input("¿Cuántas filas querés ver?: "))
        print(df.head(n))
        registrar_consulta("mostrar_primeros", {"n": n})
    except Exception as e:
        print(f"Error: {str(e)}")


def filtrar_por_intervalo(df):
    if df.empty:
        print("No hay datos para filtrar")
        return
    time_col = next((col for col in df.columns if 'timestring' in col.lower()), None)
    if not time_col:
        print("No se encontró columna de tiempo")
        return
    try:
        print("\nPrimeras filas de tiempo para referencia:")
        print(df[time_col].head().tolist())
        inicio = input("\nIngresá fecha de inicio (ej: 04.12.2024 08:56:26): ").strip()
        fin = input("Ingresá fecha de fin (ej: 04.12.2024 08:56:27): ").strip()
        df[time_col] = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M:%S", errors='coerce')
        df = df.dropna(subset=[time_col])
        inicio_dt = pd.to_datetime(inicio, format="%d.%m.%Y %H:%M:%S")
        fin_dt = pd.to_datetime(fin, format="%d.%m.%Y %H:%M:%S")
        mask = (df[time_col] >= inicio_dt) & (df[time_col] <= fin_dt)
        resultado = df.loc[mask].copy()
        if resultado.empty:
            print("\nNo se encontraron registros en ese intervalo.")
            print(f"Rango disponible: {df[time_col].min()} a {df[time_col].max()}")
        else:
            print(f"\nRegistros encontrados: {len(resultado)}")
            print(resultado.to_string(index=False))
        registrar_consulta("intervalo_tiempo", {"inicio": inicio, "fin": fin})
    except Exception as e:
        print(f"\nError al filtrar: {str(e)}")


def parsear_dataframe(df):
    # Si ya está bien formado, retornar original
    if len(df.columns) > 1:
        return df
   
    # Extraer y parsear la única columna
    col_unica = df.columns[0]
    datos_parseados = df[col_unica].str.split(';', expand=True)
   
    # Si la primera fila parece tener nombres, úsala para columnas
    primera_fila = datos_parseados.iloc[0].astype(str)
    if 'VarName' in primera_fila.values:
        datos_parseados.columns = primera_fila.str.replace('"', '', regex=False)
        datos_parseados = datos_parseados.drop(index=0).reset_index(drop=True)
    else:
        # Si no, asignar nombres genéricos
        columnas = [f"col_{i}" for i in range(datos_parseados.shape[1])]
        datos_parseados.columns = columnas
   
    return datos_parseados


def detectar_cambios(df):
    print("Columnas originales del DataFrame:", df.columns.tolist())
    print(df.head(3))
    if df.empty:
        print("El DataFrame esta vacio, no se puede procesar")
        return
    try:
        df = parsear_dataframe(df)
        varname_col = 'VarName'
        varvalue_col = 'VarValue'
        variables_unicas = df[varname_col].dropna().unique()
        if len(variables_unicas) == 0:
            print("\nNo se encontraron variables para analizar")
            return
        print("\nVariables disponibles para análisis:")
        for i, var in enumerate(sorted(variables_unicas), 1):
            print(f"{i}. {var}")
        seleccion = input("\nIngresá el número o nombre exacto de la variable: ").strip()
        try:
            idx = int(seleccion) - 1
            variable = sorted(variables_unicas)[idx]
        except ValueError:
            variable = seleccion
        except IndexError:
            print("\nNúmero fuera de rango")
            return
        if variable not in variables_unicas:
            print(f"\nError: La variable '{variable}' no existe")
            return
        try:
            umbral = float(input("\nUmbral de cambio porcentual (ej: 30 para 30%): "))
        except ValueError:
            print("\nDebe ingresar un número válido")
            return
        cambios = detectar_cambios_porcentuales(df, variable, umbral)
        if cambios.empty:
            print("\nNo se detectaron cambios significativos.")
        else:
            print(f"\nSe detectaron {len(cambios)} cambios bruscos (> {umbral}%):")
            print(cambios.to_string(index=False))
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")


def ver_historial():
    try:
        conn = sqlite3.connect("data/memoria.db")
        cursor = conn.cursor()
        cursor.execute("SELECT tipo_consulta, parametros, timestamp FROM historial_consultas ORDER BY timestamp DESC LIMIT 10")
        print("\n--- ÚLTIMAS 10 CONSULTAS ---")
        for consulta in cursor.fetchall():
            print(f"\n[{consulta[2]}] {consulta[0]}:")
            print(consulta[1][:100] + ("..." if len(consulta[1]) > 100 else ""))
    except Exception as e:
        print(f"Error al leer historial: {str(e)}")
    finally:
        conn.close()


def consultar_ia(df, bot):
    pregunta = input("\n📤 Ingresá tu pregunta para la IA (ej. '¿Cuál es el promedio de temperatura?'):\n> ")
    usar_codigo = input("¿Querés que la IA genere y ejecute código? (s/n): ").lower().strip() == "s"
    if usar_codigo:
         respuesta = bot.analisis_con_codigo_sin_ver_df(pregunta, df)
    else:
          respuesta = consultar_bot(pregunta, df)
    print("\n🧠 Respuesta IA:\n", respuesta)


def main():
    df = menu_carga()
   
    print("DataFrame cargado para análisis:")
    print(df.head())
    print(df.columns)


    if df.empty:
        print("¡Advertencia! No se cargaron datos correctamente")
        return
    while True:
        try:
            mostrar_menu()
            opcion = input("Opción (1-6): ").strip()
            if opcion == "1":
                consultar_ia(df, bot)
            elif opcion == "2":
                ver_primeros_n(df)
            elif opcion == "3":
                filtrar_por_intervalo(df)
            elif opcion == "4":
                detectar_cambios(df)
            elif opcion == "5":
                ver_historial()
            elif opcion == "6":
                print("Saliendo...")
                break
            else:
                print("Opción inválida")
        except Exception as e:
            print(f"\nError inesperado: {str(e)}\n")


if __name__ == "__main__":
    main()
