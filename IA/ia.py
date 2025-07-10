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
from typing import Dict, Any, Optional, Tuple



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
    
    @lru_cache(maxsize=1)
    def _cargar_catalogo(self) -> pd.DataFrame:
        """Carga el catálogo de variables (con cache)"""
        try:
            return pd.read_csv(self.catalogo_path, skiprows=1)
        except Exception as e:
            raise FileNotFoundError(f"Error cargando catálogo: {e}")
    
    def _procesar_limites_locomotora(self, locomotora: str) -> Tuple[Dict[str, VariableInfo], Dict[str, Dict[str, float]]]:
        """Procesa los límites y información para un tipo de locomotora específico"""
        if locomotora not in self.columnas_por_tipo:
            raise ValueError(f"Locomotora '{locomotora}' no válida. Opciones: {list(self.columnas_por_tipo.keys())}")
        
        df_catalogo = self._cargar_catalogo()
        cols = self.columnas_por_tipo[locomotora]
        
        catalogo_vars = {}
        limites_vars = {}
        
        for _, row in df_catalogo.iterrows():
            if pd.notna(row.get('Variable')) and str(row['Variable']).strip():
                var_name = str(row['Variable']).strip()
                
                # Información básica
                info = VariableInfo(
                    nombre_natural=str(row.get('Nombre', '')).strip() if pd.notna(row.get('Nombre', '')) else '',
                    tipo=str(row.get('Tipo', '')).strip() if pd.notna(row.get('Tipo', '')) else '',
                    ciclo=str(row.get('Ciclo reporte', '')).strip() if pd.notna(row.get('Ciclo reporte', '')) else '',
                    descripcion=str(row.get('Detalle', '')).strip() if pd.notna(row.get('Detalle', '')) else ''
                )
                
                # Procesar límites
                try:
                    info.minimo = float(str(row[cols['min']]).replace(",", ".")) if pd.notna(row.get(cols['min'])) else None
                    info.maximo = float(str(row[cols['max']]).replace(",", ".")) if pd.notna(row.get(cols['max'])) else None
                    info.alerta = float(str(row[cols['alerta']]).replace(",", ".")) if pd.notna(row.get(cols['alerta'])) else None
                except (ValueError, TypeError):
                    pass
                
                catalogo_vars[var_name] = info
                
                # Guardar límites para validación rápida
                if info.minimo is not None and info.maximo is not None:
                    limites_vars[var_name.upper()] = {
                        'min': info.minimo,
                        'max': info.maximo,
                        'alerta': info.alerta
                    }
        
        return catalogo_vars, limites_vars
    
    def _validar_valor(self, variable: str, valor: float, limites: Dict[str, Dict[str, float]]) -> str:
        """Valida si un valor está dentro de los límites esperados"""
        # Buscar por nombre exacto primero
        if variable in limites:
            lim = limites[variable]
        # Luego buscar por nombre en mayúsculas (compatibilidad)
        elif variable.upper() in limites:
            lim = limites[variable.upper()]
        else:
            return "SIN_LIMITES"
        
        if valor < lim['min']:
            return "DEBAJO_MINIMO"
        elif valor > lim['max']:
            return "ENCIMA_MAXIMO"
        elif lim.get('alerta') is not None and valor >= lim['alerta']:
            return "ZONA_ALERTA"
        else:
            return "NORMAL"
    
    def _generar_prompt_optimizado(self, pregunta: str, catalogo_vars: Dict[str, VariableInfo], 
                                 locomotora: str, limites_count: int, df_info: str) -> str:
        """Genera un prompt optimizado con acceso completo a variables"""
        
        # Crear resumen de tipos de variables disponibles
        tipos_variables = {}
        variables_con_limites = []
        
        for var_name, info in catalogo_vars.items():
            tipo = info.tipo if info.tipo else "General"
            if tipo not in tipos_variables:
                tipos_variables[tipo] = []
            tipos_variables[tipo].append(var_name)
            
            if info.minimo is not None and info.maximo is not None:
                variables_con_limites.append(f"{var_name} [{info.minimo}-{info.maximo}]")
        
        # Crear lista completa de variables (nombres únicamente para economizar espacio)
        todas_variables = list(catalogo_vars.keys())
        
        # Mostrar ejemplos de variables por tipo
        ejemplos_por_tipo = ""
        for tipo, vars_list in tipos_variables.items():
            ejemplos = vars_list[:3]  # 3 ejemplos por tipo
            ejemplos_por_tipo += f"• {tipo}: {', '.join(ejemplos)}"
            if len(vars_list) > 3:
                ejemplos_por_tipo += f" (y {len(vars_list)-3} más)"
            ejemplos_por_tipo += "\n"
        
        return f"""
Eres un analista de datos de locomotoras {locomotora}. 

⚠️ IMPORTANTE: YA TIENES ACCESO A TODAS LAS VARIABLES NECESARIAS. NO CREES DATOS SIMULADOS.

DATOS YA CARGADOS:
{df_info}

VARIABLES DISPONIBLES:
- DataFrame `df` YA CARGADO con columnas: VarName, VarValue (float), TimeString (datetime)
- Función `validar_valor(variable, valor, limites_vars)` YA DEFINIDA
- Diccionario `catalogo_vars` YA CARGADO con información completa de TODAS las {len(catalogo_vars)} variables
- Diccionario `limites_vars` YA CARGADO con límites para {limites_count} variables

TIPOS DE VARIABLES DISPONIBLES:
{ejemplos_por_tipo}

VARIABLES CON LÍMITES DEFINIDOS (ejemplos):
{chr(10).join(variables_con_limites[:10])}
{"..." if len(variables_con_limites) > 10 else ""}

TODAS LAS VARIABLES DISPONIBLES:
{', '.join(todas_variables)}

🚫 PROHIBIDO TERMINANTEMENTE:
- NO crear DataFrames simulados (data = {{}}, df = pd.DataFrame())
- NO redefinir df, catalogo_vars, limites_vars, validar_valor
- NO importar librerías ya disponibles
- NO crear datos de ejemplo

✅ INSTRUCCIONES OBLIGATORIAS:
1. USA ÚNICAMENTE el DataFrame `df` que ya está cargado
2. Para buscar variables: `df[df['VarName'].str.contains("TÉRMINO", case=False, na=False)]`
3. Para obtener info: `catalogo_vars.get(nombre_variable)`
4. Para descripción: `catalogo_vars[nombre_variable].descripcion`
5. Genera respuestas CONCISAS con estadísticas resumidas
6. Solo muestra registros individuales si hay anomalías críticas

⚠️ IMPORTANTE SOBRE validar_valor():
- La función validar_valor() retorna STRING, no boolean
- Valores posibles: "NORMAL", "DEBAJO_MINIMO", "ENCIMA_MAXIMO", "ZONA_ALERTA", "SIN_LIMITES"
- Para detectar anomalías usa: variable['validacion'] != "NORMAL"
- Ejemplo correcto: anomalias = df_var[df_var['validacion'] != "NORMAL"]

FORMATO ESPERADO:
- Resumen estadístico (promedio, max, min, count)
- Estado de validación general  
- Anomalías si las hay
- Conclusiones

Pregunta: "{pregunta}"

Genera ÚNICAMENTE código Python que analice los datos YA CARGADOS:
"""
    
    def _ejecutar_codigo_con_contexto(self, codigo: str, df: pd.DataFrame, 
                                    limites_vars: Dict[str, Dict[str, float]], 
                                    catalogo_vars: Dict[str, VariableInfo]) -> str:
        """Ejecuta el código generado con el contexto necesario"""
        
        # Preparar variables locales
        local_vars = {
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "limites_vars": limites_vars,
            "validar_valor": self._validar_valor,
            "catalogo_vars": catalogo_vars
        }
        
        # Capturar output
        buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(buffer):
                exec(codigo, local_vars)
        except Exception as e:
            error_msg = f"❌ Error: {type(e).__name__}: {str(e)}\n"
            
            # Sugerencias específicas de error
            if "agg function failed" in str(e):
                error_msg += "💡 Revisa operaciones numéricas en columnas no numéricas"
            elif "KeyError" in str(e):
                error_msg += "💡 Verifica nombres de variables (case-sensitive)"
            elif "empty" in str(e).lower():
                error_msg += "💡 El filtro no encontró datos"
            
            # Mostrar variables disponibles para debugging
            variables_muestra = df['VarName'].unique()[:5]
            error_msg += f"\n🔍 Variables ejemplo: {list(variables_muestra)}"
            
            return error_msg
        
        resultado = buffer.getvalue().strip()
        return resultado if resultado else "✅ Análisis completado"
    
    def _limpiar_codigo(self, codigo_raw: str) -> str:
        """Limpia el código de markdown y formato"""
        # Eliminar comentarios sobre simulación de datos
        lineas = codigo_raw.split('\n')
        lineas_limpias = []
        
        for linea in lineas:
            linea_strip = linea.strip()
            # Filtrar líneas que crean datos simulados
            if any(palabra in linea_strip for palabra in [
                'Simulación de datos', 
                'Reemplazar con el DataFrame real',
                'data = {',
                'df = pd.DataFrame(data)',
                'Simulación de limites_vars',
                'limites_vars = {',
                'def validar_valor('
            ]):
                continue
            lineas_limpias.append(linea)
        
        codigo_limpio = '\n'.join(lineas_limpias)
        
        if "```python" in codigo_limpio:
            return codigo_limpio.split("```python")[1].split("```")[0].strip()
        elif "```" in codigo_limpio:
            return codigo_limpio.split("```")[1].strip()
        return codigo_limpio.strip()
    
    def _preprocesar_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesa el DataFrame con validaciones"""
        df_processed = df.copy()
        
        # Conversiones críticas
        df_processed['VarValue'] = pd.to_numeric(df_processed['VarValue'], errors='coerce')
        df_processed['TimeString'] = pd.to_datetime(df_processed['TimeString'], errors='coerce')
        
        # Eliminar filas con valores críticos nulos
        df_processed = df_processed.dropna(subset=['VarName', 'VarValue'])
        
        return df_processed
    
    def _generar_info_dataframe(self, df: pd.DataFrame) -> str:
        """Genera información del DataFrame para incluir en el prompt"""
        info = f"- DataFrame `df` con {len(df)} registros\n"
        info += f"- Variables únicas: {len(df['VarName'].unique())}\n"
        info += f"- Rango de fechas: {df['TimeString'].min()} a {df['TimeString'].max()}\n"
        info += f"- Ejemplos de variables: {list(df['VarName'].unique()[:5])}"
        return info
    
    def analisis_con_codigo_sin_ver_df(self, pregunta: str, df: pd.DataFrame, locomotora_seleccionada: str = "ALCO") -> str:
        """
        Función principal para analizar datos de locomotoras con IA (Compatible con código existente)
        
        Args:
            pregunta: Pregunta a responder
            df: DataFrame con datos de sensores
            locomotora_seleccionada: Tipo de locomotora (ALCO, GAIA, GR12, GT22)
            
        Returns:
            Resultado del análisis
        """
        try:
            # 1. Preprocesar datos
            df_processed = self._preprocesar_dataframe(df)
            
            # 2. Cargar catálogo y límites
            catalogo_vars, limites_vars = self._procesar_limites_locomotora(locomotora_seleccionada)
            
            # 3. Generar información del DataFrame
            df_info = self._generar_info_dataframe(df_processed)
            
            print(f"🔍 Variables en catálogo: {len(catalogo_vars)}")
            print(f"🔍 Variables con límites: {len(limites_vars)}")
            print(f"📊 Registros en DataFrame: {len(df_processed)}")
            
            # 4. Generar prompt optimizado
            prompt = self._generar_prompt_optimizado(
                pregunta, catalogo_vars, locomotora_seleccionada, len(limites_vars), df_info
            )
            
            # 5. Generar código con IA
            response = self.model.generate_content(prompt)
            codigo_limpio = self._limpiar_codigo(response.text)
            
            if not codigo_limpio:
                return "❌ Error: No se generó código válido"
            
            print(f"🔧 Código generado:\n{codigo_limpio}")
            
            # 6. Ejecutar código
            resultado = self._ejecutar_codigo_con_contexto(
                codigo_limpio, df_processed, limites_vars, catalogo_vars
            )
            
            # 7. Agregar contexto final
            if resultado and len(resultado) < 50:
                resultado += f"\n\n📊 Análisis para {locomotora_seleccionada}"
                resultado += f"\n🔍 Variables procesadas: {len(df_processed['VarName'].unique())}"
            
            return resultado
            
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

modelo = genai.GenerativeModel('gemini-1.5-flash')
bot=LocomotoraBot(modelo)


# Interfaz mejoradarun
def consultar_bot(pregunta: str, df: Optional[pd.DataFrame] = None, ruta_csv: Optional[str] = None) -> str:
    if df is None and ruta_csv:
        df = cargar_csv(ruta_csv)
   
    return bot.generar_respuesta(pregunta, df)


