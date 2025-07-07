# llave = 'AIzaSyA2PipvauvVPmrGQz-Hn7nhu_VcWHypeEo'
import google.generativeai as genai
import pandas as pd
import io
import contextlib
from IA.datos import cargar_csv, seleccionar_archivo, registrar_consulta
from typing import Optional
import random
import matplotlib.pyplot as plt
import json
import html
import numpy as np



# Configuración (usa variable de entorno en producción!)
genai.configure(api_key='AIzaSyA2PipvauvVPmrGQz-Hn7nhu_VcWHypeEo')


class LocomotoraBot:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.saludos = [
            "¡Hola! 👋 Soy tu asistente de locomotoras. ¿En qué puedo ayudarte hoy?",
            "¡Buenas! 🚂 Aquí analizando datos ferroviarios. ¿Qué necesitas?",
            "¡Hola humano! 🤖💬 Listo para diagnosticar esas máquinas."
        ]
        self.despedidas = [
            "¡Hasta luego! Que tus rieles siempre estén alineados 🛤️",
            "Nos vemos. ¡Recuerda hacer mantenimiento preventivo! 🔧",
            "Bot desconectado. ¡Chuuu-chuuu! 🚆"
        ]
        self.errores = [
            "Ups, tengo un cortocircuito... 💥 Intenta reformular tu pregunta",
            "Parece que mi motor analítico falló 🛠️ ¿Podrías repetirlo?",
            "Error 404: No encontré esa respuesta en mi banco de datos"
        ]


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
   
    def analisis_con_codigo_sin_ver_df(self, pregunta: str, df: pd.DataFrame) -> str:
        """La IA genera código basándose en la pregunta y el catálogo de variables."""
        import numpy as np
        try:
            # 1. Preprocesamiento crítico del DataFrame
            df['VarValue'] = pd.to_numeric(df['VarValue'], errors='coerce')
            df['TimeString'] = pd.to_datetime(df['TimeString'], errors='coerce')
            
            # 2. Cargar y diagnosticar el catálogo de variables
            try:
                df_catalogo = pd.read_excel("data/Clasificación Variables LOGs IA.xlsx", engine='openpyxl')
                print(f"🔍 DEBUG - Columnas del catálogo: {list(df_catalogo.columns)}")
                print(f"🔍 DEBUG - Primeras filas del catálogo:\n{df_catalogo.head()}")
            except Exception as e:
                return f"❌ Error cargando catálogo: {e}"
            
            # 3. Identificar la columna correcta de variables (flexible)
            # Basado en tu CSV, la columna se llama 'Variable'
            posibles_columnas_variables = [
                'Variable',  # Esta es la correcta según tu CSV
                'LOG VARIABLES LOCOMOTORA',
                'VarName', 
                'variable',
                'nombre',
                'Nombre'
            ]
            
            columna_variables = None
            for col in posibles_columnas_variables:
                if col in df_catalogo.columns:
                    columna_variables = col
                    break
            
            # Si no encuentra ninguna, usar la primera columna
            if columna_variables is None:
                columna_variables = df_catalogo.columns[0]
                print(f"⚠️ WARNING: Usando primera columna como variables: '{columna_variables}'")
            
            # 4. Crear diccionario de variables para fácil acceso
            catalogo_vars = {}
            for _, row in df_catalogo.iterrows():
                if pd.notna(row[columna_variables]) and str(row[columna_variables]).strip():
                    variable_name = str(row[columna_variables]).strip()
                    
                    # Extraer información usando los nombres reales de las columnas
                    tipo = str(row.get('Tipo', '')).strip() if pd.notna(row.get('Tipo', '')) else ''
                    descripcion = str(row.get('Detalle', '')).strip() if pd.notna(row.get('Detalle', '')) else ''
                    minimo = str(row.get('Mínimo', '')).strip() if pd.notna(row.get('Mínimo', '')) else ''
                    maximo = str(row.get('Máximo', '')).strip() if pd.notna(row.get('Máximo', '')) else ''
                    
                    catalogo_vars[variable_name] = {
                        'tipo': tipo,
                        'descripcion': descripcion,
                        'minimo': minimo,
                        'maximo': maximo
                    }
            
            print(f"🔍 DEBUG - Variables encontradas en catálogo: {len(catalogo_vars)}")
            
            # 5. Construir el prompt con información contextual
            catalogo_info = "\n".join(
                f"{k}: {v['tipo']} ({v['minimo']}-{v['maximo']}) - {v['descripcion']}"
                for k, v in list(catalogo_vars.items())[:50]  # Limitar a 50 para no sobrecargar
                if k  # Solo incluir variables con nombre no vacío
            )
            
            prompt = f"""
            Eres un analista experto de datos de locomotoras. Trabajarás con un DataFrame `df` que contiene:
            - VarName: Nombre de la variable (ej: 'BAJA SETPOINT EGRESO FS1')
            - VarValue: Valor numérico (ya convertido a float)
            - TimeString: Marca temporal (ya convertido a datetime)

            Catálogo de variables disponibles (formato Nombre: Tipo - Descripción):
            {catalogo_info}

            Instrucciones CRÍTICAS:
            1. Las variables binarias usan 0/1 (ya convertidos a numéricos)
            2. Siempre filtra primero por VarName relevante usando df[df['VarName'] == 'NOMBRE_VARIABLE']
            3. Para series temporales con resample, USA SOLO COLUMNAS NUMÉRICAS:
            - df_filtrado = df[df['VarName'] == 'variable'].copy()
            - df_filtrado = df_filtrado[['TimeString', 'VarValue']].set_index('TimeString')
            - resultado = df_filtrado.resample('h').mean()  # Usa 'h' no 'H'
            4. Si hay valores faltantes, usa .dropna() antes de operaciones
            5. Verifica que las variables existan antes de usarlas
            6. Para análisis temporal, ordena por TimeString primero
            7. NUNCA hagas resample() sobre DataFrames que contengan columnas de texto
            8. Siempre selecciona solo las columnas numéricas antes del resample

            🚫 IMPORTANTE: El DataFrame `df` ya está cargado y contiene los datos reales. NO lo crees ni lo reemplaces. No simules datos. Usa directamente el `df` que ya existe.

            Pregunta a responder:
            "{pregunta}"

            Genera SOLO código Python válido que:
            1. Comience con 'import pandas as pd' e 'import numpy as np'
            2. Contenga manejo de tipos de datos seguro
            3. Incluya filtrado por variables relevantes
            4. Verifique que las variables existan
            5. Finalice con print(resultado)
            """
            
            # 6. Generar y ejecutar el código
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

            # 7. Ejecutar con validación
            import numpy as np
            local_vars = {"df": df.copy(), "pd": pd, "np": np}
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
                    error_msg += "💡 SOLUCIÓN: Verifica que los nombres de variables sean exactos"
                elif "empty" in str(e).lower():
                    error_msg += "💡 SOLUCIÓN: Es posible que el filtro no encuentre datos"
                    
                return error_msg

            resultado = buffer.getvalue().strip()
            return resultado if resultado else "✅ Análisis completado (sin output)"

        except Exception as e:
            return f"❌ Error general: {type(e).__name__}: {str(e)}"


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


