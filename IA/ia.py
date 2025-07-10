# llave = 'AIzaSyA2PipvauvVPmrGQz-Hn7nhu_VcWHypeEo'
import google.generativeai as genai
import pandas as pd
import io
import contextlib
from IA.datos import cargar_csv, seleccionar_archivo, registrar_consulta
from typing import Optional
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import lru_cache
import json
import html
import numpy as np



# Configuración (usa variable de entorno en producción!)
genai.configure(api_key='AIzaSyA2PipvauvVPmrGQz-Hn7nhu_VcWHypeEo')

@dataclass
class VariableInfo:
    nombre_natural: str
    tipo: str
    ciclo: str
    descripcion: str
    minimo: Optional[float] = None
    maximo: Optional[float] = None
    alerta: Optional[float] = None

class LocomotoraBot:
    def __init__(self, model, catalogo_path: str = "data/Clasificación Variables LOGs IA (1) - Hoja1.csv"):
        self.model = model
        self.catalogo_path = catalogo_path
        self._catalogo_cache = None
        
        # Mapeo de columnas por tipo de locomotora
        self.columnas_por_tipo = {
            "ALCO": {"min": "Mínimo", "max": "Máximo", "alerta": "Alerta"},
            "GAIA": {"min": "Mínimo.1", "max": "Máximo.1", "alerta": "Alerta.1"},
            "GR12": {"min": "Mínimo.2", "max": "Máximo.2", "alerta": "Alerta.2"},
            "GT22": {"min": "Mínimo.3", "max": "Máximo.3", "alerta": "Alerta.3"}
        }
    lru_cache(maxsize=1)

    def generar_respuesta(self, pregunta: str, df: Optional[pd.DataFrame] = None) -> str:
        """Evalúa el tipo de pregunta y responde acorde"""
        pregunta = pregunta.lower().strip()
       
        # Modo conversacional
        if any(palabra in pregunta for palabra in ["hola", "hi", "qué tal", "cómo estás"]):
            return random.choice(self.saludos)
           
        if any(palabra in pregunta for palabra in ["adiós", "chao", "hasta luego"]):
            return random.choice(self.despedidas)
       
        # Modo técnico
        if df is not None and not df.empty:
            return self._analisis_tecnico(pregunta, df)
        else:
            return "🔍 Por favor carga datos primero para análisis técnico"


    def _analisis_tecnico(self, pregunta: str, df: pd.DataFrame) -> str:
        """Análisis especializado con datos"""
        try:
            contexto = "Eres un ingeniero senior de locomotoras diésel. " + \
                     "Combina conocimiento técnico con explicaciones claras.\n\n"
           
            if "corriente" in pregunta:
                contexto += "Foco en análisis eléctrico (umbral seguro: 15-25A)"
            elif "temperatura" in pregunta:
                contexto += "Foco en termodinámica (rango óptimo: 65-90°C)"
           
            datos_relevantes = df.tail(50).to_string()
           
            prompt = f"""
            Eres un ingeniero especializado en locomotoras diésel. Analiza estos datos:


            **Variables Clave**:
            - Presión aceite: Rango normal (10000-12000)
            - RPM: Rango normal (8000-9000)
            - Temperaturas (IMT): Rango normal (-30 a 50)


            **Datos Recientes**:
            {df.tail(20).to_string()}


            **Pregunta**: "{pregunta}"


            **Formato de Respuesta**:
            1. 📌 Hallazgo principal
            2. 🔍 Variable crítica (si aplica)
            3. 🚨 Nivel de riesgo (1-5)
            4. 🛠️ Acción recomendada
            """
           
            response = self.model.generate_content(prompt)
            return self._formatear_respuesta(response.text)
           
        except Exception as e:
            return f"{random.choice(self.errores)}. Detalle: {str(e)}"


    def _formatear_respuesta(self, respuesta: str) -> str:
        """Da formato humano a la respuesta técnica"""
        lineas = respuesta.split('\n')
        if len(lineas) > 2:  # Se ajustó el chequeo porque ya no hay tip
            return "\n".join([
                f"🔧 **Análisis Técnico** 🔧",
                f"{lineas[0]}",
                "",
                "🚨 **Riesgo/Causas**:",
                f"{lineas[1]}",
                "",
                "🛠 **Acciones Recomendadas**:",
                f"{lineas[2]}"
            ])
        return respuesta
   
    def analisis_con_codigo_sin_ver_df(self, pregunta: str, df: pd.DataFrame, locomotora_seleccionada: str = "ALCO") -> str:
        """La IA genera código basándose en la pregunta y el catálogo de variables."""
        import numpy as np
        import os
        
        try:
            # 1. Preprocesamiento crítico del DataFrame
            df['VarValue'] = pd.to_numeric(df['VarValue'], errors='coerce')
            df['TimeString'] = pd.to_datetime(df['TimeString'], errors='coerce')
            
            # 2. Cargar y procesar el catálogo con el formato correcto
            try:
                ruta_catalogo = "data/Clasificación Variables LOGs IA (1) - Hoja1.csv"
                # Usar skiprows=1 para saltar la primera fila con los headers de locomotoras
                df_catalogo = pd.read_csv(ruta_catalogo, skiprows=1)
                print(f"🔍 DEBUG - Columnas del catálogo: {list(df_catalogo.columns)}")
                print(f"🔍 DEBUG - Primeras filas del catálogo:\n{df_catalogo.head()}")
            except Exception as e:
                return f"❌ Error cargando catálogo: {e}"
            
            # 3. Mapear columnas según el tipo de locomotora
            columnas_por_tipo = {
                "ALCO": {"min": "Mínimo", "max": "Máximo", "alerta": "Alerta"},
                "GAIA": {"min": "Mínimo.1", "max": "Máximo.1", "alerta": "Alerta.1"},
                "GR12": {"min": "Mínimo.2", "max": "Máximo.2", "alerta": "Alerta.2"},
                "GT22": {"min": "Mínimo.3", "max": "Máximo.3", "alerta": "Alerta.3"}
            }
            
            if locomotora_seleccionada not in columnas_por_tipo:
                return f"❌ Error: Tipo de locomotora '{locomotora_seleccionada}' no válido. Opciones: {list(columnas_por_tipo.keys())}"
            
            cols = columnas_por_tipo[locomotora_seleccionada]
            
            # 4. Crear diccionario completo de variables con límites
            catalogo_vars = {}
            limites_vars = {}
            
            for _, row in df_catalogo.iterrows():
                if pd.notna(row.get('Variable')) and str(row['Variable']).strip():
                    var_name = str(row['Variable']).strip()
                    var_name_upper = var_name.upper()
                    
                    # Información básica
                    nombre_natural = str(row.get('Nombre', '')).strip() if pd.notna(row.get('Nombre', '')) else ''
                    tipo = str(row.get('Tipo', '')).strip() if pd.notna(row.get('Tipo', '')) else ''
                    ciclo = str(row.get('Ciclo reporte', '')).strip() if pd.notna(row.get('Ciclo reporte', '')) else ''
                    descripcion = str(row.get('Detalle', '')).strip() if pd.notna(row.get('Detalle', '')) else ''
                    
                    # Límites para el tipo de locomotora específico
                    try:
                        minimo = float(str(row[cols['min']]).replace(",", ".")) if pd.notna(row.get(cols['min'])) else None
                        maximo = float(str(row[cols['max']]).replace(",", ".")) if pd.notna(row.get(cols['max'])) else None
                        alerta = float(str(row[cols['alerta']]).replace(",", ".")) if pd.notna(row.get(cols['alerta'])) else None
                    except (ValueError, TypeError):
                        minimo = maximo = alerta = None
                    
                    catalogo_vars[var_name] = {
                        'nombre_natural': nombre_natural,
                        'tipo': tipo,
                        'ciclo': ciclo,
                        'descripcion': descripcion,
                        'minimo': minimo,
                        'maximo': maximo,
                        'alerta': alerta
                    }
                    
                    # También guardamos con nombre en mayúsculas para búsqueda
                    if minimo is not None and maximo is not None:
                        limites_vars[var_name_upper] = {
                            'min': minimo,
                            'max': maximo,
                            'alerta': alerta
                        }
            
            print(f"🔍 DEBUG - Variables encontradas en catálogo: {len(catalogo_vars)}")
            print(f"🔍 DEBUG - Variables con límites para {locomotora_seleccionada}: {len(limites_vars)}")
            
            # 5. Función de validación de límites integrada
            def validar_valor(variable, valor, limites):
                """Valida si un valor está dentro de los límites esperados"""
                var_upper = variable.upper()
                if var_upper not in limites:
                    return "SIN_LIMITES"
                
                lim = limites[var_upper]
                if valor < lim['min']:
                    return "DEBAJO_MINIMO"
                elif valor > lim['max']:
                    return "ENCIMA_MAXIMO"
                elif lim.get('alerta') is not None and valor >= lim['alerta']:
                    return "ZONA_ALERTA"
                else:
                    return "NORMAL"
            
            # 6. Construir información del catálogo para la IA
            catalogo_info = ""
            for var_name, info in list(catalogo_vars.items())[:40]:  # Limitar para no sobrecargar
                if info['minimo'] is not None and info['maximo'] is not None:
                    rango = f"[{info['minimo']}-{info['maximo']}]"
                    if info['alerta'] is not None:
                        rango += f" ⚠️>{info['alerta']}"
                else:
                    rango = "[Sin límites]"
                
                catalogo_info += f"• {var_name}: {info['tipo']} {rango}\n"
                catalogo_info += f"  Descripción: {info['descripcion']}\n"
                if info['nombre_natural']:
                    catalogo_info += f"  Nombre natural: {info['nombre_natural']}\n"
                catalogo_info += "\n"
            
            prompt = f"""
            Eres un analista experto de datos de locomotoras {locomotora_seleccionada}. Trabajarás con un DataFrame `df` que contiene:
            - VarName: Nombre de la variable (ej: 'BAJA SETPOINT EGRESO FS1')
            - VarValue: Valor numérico (ya convertido a float)
            - TimeString: Marca temporal (ya convertido a datetime)

            CATÁLOGO DE VARIABLES PARA LOCOMOTORA {locomotora_seleccionada}:
            {catalogo_info}

            FUNCIONES Y DATOS DISPONIBLES:
            - `validar_valor(variable, valor, limites_vars)` retorna:
            - "NORMAL": Valor dentro del rango normal
            - "ZONA_ALERTA": Valor en zona de alerta
            - "DEBAJO_MINIMO": Valor por debajo del mínimo
            - "ENCIMA_MAXIMO": Valor por encima del máximo
            - "SIN_LIMITES": Variable sin límites definidos

            - `limites_vars`: Diccionario con los límites válidos de TODAS las variables, extraídas del catálogo. 
            ❗NO DEFINAS OTRO DICCIONARIO DE LÍMITES.
            ✅ Usá exclusivamente `limites_vars`.

            INSTRUCCIONES CRÍTICAS:
            1. Las variables binarias usan 0/1 (ya convertidos a numéricos)
            2. Para encontrar variables relevantes, usá `.str.contains()`, por ejemplo:
            ```python
            df[df['VarName'].str.contains("RPM", case=False, na=False)]

            NO compares con ==, porque los nombres no siempre coinciden exactamente.
            3. Si hay múltiples coincidencias (ej: muchas variables con “RPM”), analizá todas.
            4. Para series temporales:
            df_filtrado = df_filtrado[['TimeString', 'VarValue']].set_index('TimeString')
            resultado = df_filtrado.resample('h').mean()

            5. SIEMPRE validá los valores con validar_valor().

            6. Explicá qué significa cada variable según su descripción en el catálogo.

            7. Para variables binarias, interpretá 0 = apagado, 1 = activado.

            8. NO crees ni simules el DataFrame. Ya está cargado como df.

            FORMATO DE RESPUESTA ESPERADO:

            - Descripción de la variable(s) encontrada(s)

            - Valor(es) máximo(s), promedio(s), etc.

            - Validación con validar_valor() (normal / alerta / fuera de rango)

            - Análisis temporal si corresponde

            - Conclusiones claras y útiles

            ⚠️ MUY IMPORTANTE:
            - ❌ NO DEFINAS ni simules el DataFrame `df` ni los diccionarios `limites_vars` o la función `validar_valor`.
            - ✅ Todos estos objetos ya están cargados y disponibles.
            - Cualquier intento de definir `df = pd.DataFrame(...)` es un ERROR.


            📌 Pregunta a responder:
            "{pregunta}"

            🔧 Generá código Python que:

            - Filtre variables relevantes con .str.contains()

            - Analice sus valores

            - Valide con validar_valor()

            - Imprima resultados interpretados
            """
            
            # 8. Generar y ejecutar el código
            response = self.model.generate_content(prompt)
            codigo_raw = response.text.strip()
            
            # Limpiar el código de markdown
            if "```python" in codigo_raw:
                codigo = codigo_raw.split("```python")[1].split("```")[0].strip()
            elif "```" in codigo_raw:
                codigo = codigo_raw.split("```")[1].strip()
            else:
                codigo = codigo_raw
            
            if not codigo:
                return "❌ Error: No se generó código válido"

            print(f"🔍 DEBUG - Código generado:\n{codigo}")

            # 9. Ejecutar con validación y funciones auxiliares
            local_vars = {
                "df": df.copy(), 
                "pd": pd, 
                "np": np,
                "limites_vars": limites_vars,
                "validar_valor": validar_valor,
                "catalogo_vars": catalogo_vars
            }
            
            buffer = io.StringIO()
            
            try:
                with contextlib.redirect_stdout(buffer):
                    exec(codigo, local_vars)
            except Exception as e:
                error_msg = f"❌ Error en ejecución: {type(e).__name__}: {str(e)}\n"
                error_msg += f"🔍 Código que falló:\n{codigo}\n"
                
                if "agg function failed" in str(e):
                    error_msg += "💡 SOLUCIÓN: Revisa que todas las columnas usadas en operaciones numéricas sean de tipo float/int"
                elif "KeyError" in str(e):
                    error_msg += "💡 SOLUCIÓN: Verifica que los nombres de variables sean exactos (case-sensitive)"
                elif "empty" in str(e).lower():
                    error_msg += "💡 SOLUCIÓN: Es posible que el filtro no encuentre datos. Verifica nombres de variables."
                
                # Mostrar variables disponibles para debugging
                variables_disponibles = df['VarName'].unique()[:10]
                error_msg += f"\n🔍 Variables disponibles en el DataFrame: {variables_disponibles}"
                    
                return error_msg

            resultado = buffer.getvalue().strip()
            
            # 10. Agregar contexto adicional si la respuesta es muy corta
            if resultado and len(resultado) < 100:
                resultado += f"\n\n📊 Análisis para locomotora {locomotora_seleccionada}"
                resultado += f"\n🔍 Variables disponibles en catálogo: {len(catalogo_vars)}"
                resultado += f"\n⚙️ Variables con límites definidos: {len(limites_vars)}"
            
            return resultado if resultado else "✅ Análisis completado (sin output visible)"

        except Exception as e:
            return f"❌ Error general: {type(e).__name__}: {str(e)}"


# Función independiente para compatibilidad (si la necesitas fuera de la clase)
def crear_locomotora_bot(model, catalogo_path: str = "data/Clasificación Variables LOGs IA (1) - Hoja1.csv"):
    """Factory function para crear una instancia de LocomotoraBot"""
    return LocomotoraBot(model, catalogo_path)

def mostrar_grafico_si_aplica(df: pd.DataFrame):
    try:
        if "TimeString" in df.columns and "VarValue" in df.columns:
            df = df.copy()
            df["TimeString"] = pd.to_datetime(df["TimeString"], errors='coerce')
            df = df.dropna(subset=["TimeString", "VarValue"])
            df = df.sort_values("TimeString")
           
            # Colores para las variables
            colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
           
            # Matplotlib: Un solo gráfico con todas las variables
            if "VarName" in df.columns:
                plt.figure(figsize=(12, 6))
                for idx, (nombre, subdf) in enumerate(df.groupby("VarName")):
                    if len(subdf) > 10:  # Solo graficar si hay suficientes datos
                        plt.plot(
                            subdf["TimeString"],
                            subdf["VarValue"],
                            label=nombre,
                            color=colors[idx % len(colors)]
                        )
                plt.title("Evolución de Variables")
                plt.xlabel("Tiempo")
                plt.ylabel("Valor")
                plt.legend(title="Variables")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
           
            # Chart.js: Un solo gráfico interactivo con todas las variables
            if "VarName" in df.columns:
                datasets = []
                for idx, (nombre, subdf) in enumerate(df.groupby("VarName")):
                    if len(subdf) > 10:
                        datasets.append({
                            "label": nombre,
                            "data": subdf["VarValue"].tolist(),
                            "borderColor": colors[idx % len(colors)],
                            "backgroundColor": colors[idx % len(colors)] + "80",  # Añade opacidad
                            "fill": False
                        })
               
                if datasets:  # Solo mostrar si hay datos
                    print(f"```chartjs\n"
                          f"chart: {{\n"
                          f"  type: 'line',\n"
                          f"  data: {{\n"
                          f"    labels: {df['TimeString'].dt.strftime('%Y-%m-%d %H:%M:%S').drop_duplicates().tolist()},\n"
                          f"    datasets: {datasets}\n"
                          f"  }},\n"
                          f"  options: {{\n"
                          f"    scales: {{\n"
                          f"      x: {{ title: {{ display: true, text: 'Tiempo' }} }},\n"
                          f"      y: {{ title: {{ display: true, text: 'Valor' }} }}\n"
                          f"    }},\n"
                          f"    plugins: {{ legend: {{ display: true }} }}\n"
                          f"  }}\n"
                          f"}}\n"
                          f"```")
           
            # Caso sin VarName (una sola variable)
            elif len(df) > 10:
                plt.figure(figsize=(12, 6))
                plt.plot(df["TimeString"], df["VarValue"], color=colors[0])
                plt.title("Evolución de VarValue")
                plt.xlabel("Tiempo")
                plt.ylabel("Valor")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
               
                print(f"```chartjs\n"
                      f"chart: {{\n"
                      f"  type: 'line',\n"
                      f"  data: {{\n"
                      f"    labels: {df['TimeString'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()},\n"
                      f"    datasets: [{{\n"
                      f"      label: 'VarValue',\n"
                      f"      data: {df['VarValue'].tolist()},\n"
                      f"      borderColor: '{colors[0]}',\n"
                      f"      backgroundColor: '{colors[0]}80',\n"
                      f"      fill: False\n"
                      f"    }}]\n"
                      f"  }},\n"
                      f"  options: {{\n"
                      f"    scales: {{\n"
                      f"      x: {{ title: {{ display: true, text: 'Tiempo' }} }},\n"
                      f"      y: {{ title: {{ display: true, text: 'Valor' }} }}\n"
                      f"    }}\n"
                      f"  }}\n"
                      f"}}\n"
                      f"```")
   
    except Exception as e:
        print(f"⚠️ No se pudo graficar: {e}")


bot=LocomotoraBot()


# Interfaz mejoradarun
def consultar_bot(pregunta: str, df: Optional[pd.DataFrame] = None, ruta_csv: Optional[str] = None) -> str:
    if df is None and ruta_csv:
        df = cargar_csv(ruta_csv)
   
    return bot.generar_respuesta(pregunta, df)


