# llave = 'AIzaSyA2PipvauvVPmrGQz-Hn7nhu_VcWHypeEo'
import google.generativeai as genai
import pandas as pd
import io
import contextlib
from IA.datos import cargar_csv, seleccionar_archivo, registrar_consulta
from typing import Optional
import random
import matplotlib.pyplot as plt




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
        """La IA genera código basándose solo en la pregunta. Luego lo ejecuta localmente sobre el df."""
        try:
            columnas = df.columns.tolist()


            # Paso 1: Generar el código
            prompt_codigo = f"""
            Eres un experto en análisis de datos con pandas.


            El DataFrame se llama `df` y tiene las siguientes columnas: {columnas}.


            - `VarName` contiene el nombre de la variable (por ejemplo, 'RPM - 7KF00', 'PRESION ACEITE COMPRESOR - 7KF00', etc.)
            - `VarValue` contiene el valor medido (número).
            - `TimeString` contiene la fecha y hora de la medición.
            - Cada fila representa una única medición de una variable en un instante de tiempo.


            Usá pandas para responder la siguiente pregunta. Si es necesario, filtrá las filas que coincidan con palabras clave dentro de la columna 'VarName'.


            Pregunta del usuario:
            \"{pregunta}\"


            Genera solamente el código Python necesario para responder a esa pregunta. Sin explicaciones. Sin comentarios. Asegurate de que el resultado se imprima con `print(...)` al final.
            """


            response = self.model.generate_content(prompt_codigo)
            codigo = response.text.strip().strip("```python").strip("```")


            local_vars = {"df": df.copy(), "pd": pd}


            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(codigo, local_vars)


            resultado = buffer.getvalue().strip()


            # Buscar el último DataFrame creado que no sea 'df'
            dfs_generados = [v for k, v in local_vars.items() if isinstance(v, pd.DataFrame) and k != 'df']
            df_resultado = dfs_generados[-1] if dfs_generados else None


            if df_resultado is not None:
                mostrar_grafico_si_aplica(df_resultado)




            # Paso 3: Explicar el resultado
            prompt_explicacion = f"""
            Este fue el resultado de ejecutar código Python sobre un DataFrame en pandas:


            Código:
            {codigo}


            Resultado:
            {resultado}


            Explica al usuario qué significa este resultado, como si no supiera programar.
            """


            explicacion = self.model.generate_content(prompt_explicacion).text.strip()
           
            resumen = {
                "resultado": resultado,
                "respuesta": explicacion
            }


            exito = registrar_consulta("analisis_resultado", resumen)
            return f"📊 Código generado:\n```python\n{codigo}\n```\n\n📈 Resultado:\n{resultado}\n\n🧠 Explicación:\n{explicacion}"


        except Exception as e:
            return f"❌ Error ejecutando el análisis: {str(e)}"


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
