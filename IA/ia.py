import google.generativeai as genai
import pandas as pd
import io
import contextlib
from IA.datos import cargar_csv
from typing import Optional
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

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
    
    def _convertir_valor_limite(self, valor_raw):
        """Convierte valores que pueden tener formato '904/990' o '100/200/300' a float (toma el máximo)"""
        if pd.isna(valor_raw) or not str(valor_raw).strip():
            return None
        
        valor_str = str(valor_raw).strip()
        
        # Si contiene '/', dividir y tomar el valor máximo
        if '/' in valor_str:
            try:
                valores = valor_str.split('/')
                valores_numericos = []
                for v in valores:
                    v_clean = v.strip().replace(",", ".")
                    if v_clean:  # Solo si no está vacío
                        valores_numericos.append(float(v_clean))
                
                if valores_numericos:
                    return max(valores_numericos)
                else:
                    return None
            except (ValueError, TypeError) as e:
                print(f"⚠️  Error procesando valor con barras '{valor_str}': {e}")
                return None
        
        # Si es un valor normal (sin barras)
        try:
            return float(valor_str.replace(",", "."))
        except (ValueError, TypeError) as e:
            print(f"⚠️  Error procesando valor simple '{valor_str}': {e}")
            return None

    def _procesar_limites_locomotora(self, locomotora: str) -> Tuple[Dict[str, VariableInfo], Dict[str, Dict[str, float]]]:
        """Procesa los límites y información para un tipo de locomotora específico"""
        if locomotora not in self.columnas_por_tipo:
            raise ValueError(f"Locomotora '{locomotora}' no válida. Opciones: {list(self.columnas_por_tipo.keys())}")
        
        df_catalogo = self._cargar_catalogo()
        cols = self.columnas_por_tipo[locomotora]
        
        catalogo_vars = {}
        limites_vars = {}
        
        variables_procesadas = 0
        variables_con_limites = 0
        
        for idx, row in df_catalogo.iterrows():
            if pd.notna(row.get('Variable')) and str(row['Variable']).strip():
                var_name = str(row['Variable']).strip()
                
                # Información básica
                info = VariableInfo(
                    nombre_natural=str(row.get('Nombre', '')).strip() if pd.notna(row.get('Nombre', '')) else '',
                    tipo=str(row.get('Tipo', '')).strip() if pd.notna(row.get('Tipo', '')) else '',
                    ciclo=str(row.get('Ciclo reporte', '')).strip() if pd.notna(row.get('Ciclo reporte', '')) else '',
                    descripcion=str(row.get('Detalle', '')).strip() if pd.notna(row.get('Detalle', '')) else ''
                )
                
                # Procesar límites usando la función existente
                try:
                    # Obtener valores de límites
                    min_raw = row.get(cols['min'])
                    max_raw = row.get(cols['max'])
                    alerta_raw = row.get(cols['alerta'])
                    
                    # Convertir usando la función existente
                    min_val = self._convertir_valor_limite(min_raw)
                    max_val = self._convertir_valor_limite(max_raw)
                    alerta_val = self._convertir_valor_limite(alerta_raw)
                    
                    # Asignar valores procesados
                    if min_val is not None:
                        info.minimo = min_val
                    if max_val is not None:
                        info.maximo = max_val
                    if alerta_val is not None:
                        info.alerta = alerta_val
                    
                    # Solo debug para variables específicas si es necesario
                    if "RPM" in var_name.upper() and False:  # Cambiar a True si necesitas debug
                        print(f"🔍 DEBUG RPM: {var_name}")
                        print(f"   Min: {min_val}, Max: {max_val}, Alerta: {alerta_val}")
                        
                except Exception as e:
                    print(f"⚠️  Error procesando límites para {var_name}: {e}")
                    continue
                
                catalogo_vars[var_name] = info
                variables_procesadas += 1
                
                # Guardar límites para validación rápida (solo si tiene min y max)
                if min_val is not None and max_val is not None:
                    limite_dict = {
                        'min': min_val,
                        'max': max_val,
                        'alerta': alerta_val
                    }
                    
                    # Guardar con nombre exacto y normalizado
                    limites_vars[var_name] = limite_dict
                    limites_vars[var_name.upper().strip()] = limite_dict
                    
                    variables_con_limites += 1
        
        return catalogo_vars, limites_vars
    
    def _validar_valor(self, variable: str, valor: float, limites: Dict[str, Dict[str, float]]) -> str:
        """Valida si un valor está dentro de los límites esperados"""
        if variable in limites:
            lim = limites[variable]
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
    
    def debug_limites_variable(self, variable_name: str, locomotora: str = "ALCO") -> str:
        """Método para debuggear límites de una variable específica"""
        try:
            catalogo_vars, limites_vars = self._procesar_limites_locomotora(locomotora)
            
            info = f"🔍 DEBUG para variable: '{variable_name}'\n"
            info += f"🔍 Locomotora: {locomotora}\n\n"
            
            # Buscar en catálogo
            if variable_name in catalogo_vars:
                var_info = catalogo_vars[variable_name]
                info += f"✅ Variable encontrada en catálogo:\n"
                info += f"   Nombre natural: {var_info.nombre_natural}\n"
                info += f"   Tipo: {var_info.tipo}\n"
                info += f"   Mínimo: {var_info.minimo}\n"
                info += f"   Máximo: {var_info.maximo}\n"
                info += f"   Alerta: {var_info.alerta}\n\n"
            else:
                info += f"❌ Variable NO encontrada en catálogo\n\n"
            
            # Buscar en límites
            found_in_limits = False
            for key, limits in limites_vars.items():
                if key.upper() == variable_name.upper() or key == variable_name:
                    info += f"✅ Límites encontrados con clave: '{key}'\n"
                    info += f"   Min: {limits['min']}\n"
                    info += f"   Max: {limits['max']}\n"
                    info += f"   Alerta: {limits.get('alerta', 'N/A')}\n\n"
                    found_in_limits = True
                    break
            
            if not found_in_limits:
                info += f"❌ Variable NO encontrada en limites_vars\n"
                info += f"🔍 Claves disponibles en limites_vars:\n"
                for key in list(limites_vars.keys())[:10]:  # Mostrar primeras 10
                    info += f"   - '{key}'\n"
                info += f"   ... (total: {len(limites_vars)} claves)\n\n"
            
            return info
            
        except Exception as e:
            return f"❌ Error en debug: {e}"
    
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

class LocomotoraMLBot:
    def __init__(self, locomotora_bot: LocomotoraBot):
        self.bot = locomotora_bot
        self.modelos_entrenados = {}
        self.scalers = {}
        self.encoders = {}
        
    def preparar_datos_ml(self, df: pd.DataFrame, ventana_temporal: int = 10) -> pd.DataFrame:
        """
        Prepara los datos para machine learning creando características temporales
        """
        df_ml = df.copy()
        df_ml['TimeString'] = pd.to_datetime(df_ml['TimeString'])
        df_ml = df_ml.sort_values(['VarName', 'TimeString'])
        
        # Crear características por variable
        features_list = []
        
        for var_name in df_ml['VarName'].unique():
            var_data = df_ml[df_ml['VarName'] == var_name].copy()
            
            if len(var_data) < ventana_temporal:
                continue
                
            # Características estadísticas en ventana móvil
            var_data['rolling_mean'] = var_data['VarValue'].rolling(window=ventana_temporal).mean()
            var_data['rolling_std'] = var_data['VarValue'].rolling(window=ventana_temporal).std()
            var_data['rolling_min'] = var_data['VarValue'].rolling(window=ventana_temporal).min()
            var_data['rolling_max'] = var_data['VarValue'].rolling(window=ventana_temporal).max()
            
            # Características de tendencia
            var_data['diff_1'] = var_data['VarValue'].diff()
            var_data['diff_2'] = var_data['VarValue'].diff().diff()
            var_data['pct_change'] = var_data['VarValue'].pct_change()
            
            # Características temporales
            var_data['hour'] = var_data['TimeString'].dt.hour
            var_data['day_of_week'] = var_data['TimeString'].dt.dayofweek
            var_data['month'] = var_data['TimeString'].dt.month
            
            # Identificar anomalías usando límites del catálogo
            _, limites_vars = self.bot._procesar_limites_locomotora("ALCO")
            
            if var_name in limites_vars:
                limite = limites_vars[var_name]
                var_data['anomalia'] = (
                    (var_data['VarValue'] < limite['min']) | 
                    (var_data['VarValue'] > limite['max'])
                ).astype(int)
            else:
                var_data['anomalia'] = 0
                
            features_list.append(var_data)
        
        if features_list:
            df_features = pd.concat(features_list, ignore_index=True)
            df_features = df_features.dropna()
            return df_features
        else:
            return pd.DataFrame()
    
    def detectar_anomalias_ml(self, df: pd.DataFrame, variable_objetivo: str = None) -> dict:
        """
        Detecta anomalías usando Isolation Forest
        """
        try:
            df_ml = self.preparar_datos_ml(df)
            
            if df_ml.empty:
                return {"error": "No hay suficientes datos para ML"}
            
            # Filtrar por variable si se especifica
            if variable_objetivo:
                df_ml = df_ml[df_ml['VarName'] == variable_objetivo]
                
            if len(df_ml) < 50:
                return {"error": "Datos insuficientes para detección de anomalías"}
            
            # Preparar características numéricas
            feature_cols = ['VarValue', 'rolling_mean', 'rolling_std', 'rolling_min', 
                          'rolling_max', 'diff_1', 'diff_2', 'pct_change', 'hour', 
                          'day_of_week', 'month']
            
            X = df_ml[feature_cols].fillna(0)
            
            # Escalar datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Entrenar Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalias_pred = iso_forest.fit_predict(X_scaled)
            
            # Agregar predicciones al DataFrame
            df_ml['anomalia_ml'] = (anomalias_pred == -1).astype(int)
            
            # Estadísticas
            total_anomalias = sum(anomalias_pred == -1)
            porcentaje_anomalias = (total_anomalias / len(df_ml)) * 100
            
            # Anomalías más críticas (con mayor desviación)
            scores = iso_forest.decision_function(X_scaled)
            df_ml['anomalia_score'] = scores
            
            anomalias_criticas = df_ml[df_ml['anomalia_ml'] == 1].nsmallest(10, 'anomalia_score')
            
            resultado = {
                "total_registros": len(df_ml),
                "total_anomalias": total_anomalias,
                "porcentaje_anomalias": round(porcentaje_anomalias, 2),
                "anomalias_criticas": anomalias_criticas[['VarName', 'VarValue', 'TimeString', 'anomalia_score']].to_dict('records'),
                "variables_afectadas": df_ml[df_ml['anomalia_ml'] == 1]['VarName'].value_counts().to_dict()
            }
            
            return resultado
            
        except Exception as e:
            return {"error": f"Error en detección de anomalías: {str(e)}"}
    
    def predecir_fallos(self, df: pd.DataFrame, variable_objetivo: str, 
                       horas_adelante: int = 24) -> dict:
        """
        Predice posibles fallos usando Random Forest y proporciona explicaciones detalladas
        """
        try:
            df_ml = self.preparar_datos_ml(df)
            
            if df_ml.empty:
                return {"error": "No hay suficientes datos para predicción"}
            
            # Filtrar por variable objetivo
            df_var = df_ml[df_ml['VarName'] == variable_objetivo].copy()
            
            if len(df_var) < 100:
                return {"error": "Datos insuficientes para predicción"}
            
            # Crear variable objetivo (fallo en las próximas horas)
            df_var = df_var.sort_values('TimeString')
            df_var['fallo_futuro'] = df_var['anomalia'].shift(-horas_adelante).fillna(0)
            
            # Características para el modelo
            feature_cols = ['VarValue', 'rolling_mean', 'rolling_std', 'rolling_min', 
                          'rolling_max', 'diff_1', 'diff_2', 'pct_change', 'hour', 
                          'day_of_week', 'month']
            
            X = df_var[feature_cols].fillna(0)
            y = df_var['fallo_futuro']
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Entrenar modelo
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = rf_model.predict(X_test)
            
            # Importancia de características
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Predicción para los datos más recientes
            datos_recientes = X.tail(10)
            prediccion_reciente = rf_model.predict_proba(datos_recientes)[:, 1]
            
            # Guardar modelo
            self.modelos_entrenados[f"fallo_{variable_objetivo}"] = rf_model
            
            # Generar explicaciones usando Gemini
            _, limites_vars = self.bot._procesar_limites_locomotora("ALCO")
            limites_info = limites_vars.get(variable_objetivo, {'min': 'N/A', 'max': 'N/A', 'alerta': 'N/A'})
            
            # Preparar contexto para Gemini
            contexto = f"""
Datos recientes de la variable '{variable_objetivo}':
- Últimos 10 valores: {datos_recientes['VarValue'].tolist()}
- Media móvil: {datos_recientes['rolling_mean'].tolist()}
- Desviación estándar móvil: {datos_recientes['rolling_std'].tolist()}
- Cambios porcentuales: {datos_recientes['pct_change'].tolist()}
- Límites catálogo: min={limites_info['min']}, max={limites_info['max']}, alerta={limites_info['alerta']}
Características más importantes del modelo:
{importances.head(3).to_string(index=False)}
Probabilidad de fallo predicha: {prediccion_reciente.max():.2%}
Riesgo asignado: {'ALTO' if prediccion_reciente.max() > 0.7 else 'MEDIO' if prediccion_reciente.max() > 0.3 else 'BAJO'}
Precisión del modelo: {rf_model.score(X_test, y_test):.2%}
Reporte de clasificación:
{classification_report(y_test, y_pred, output_dict=True)}
"""

            prompt_explicacion = f"""
Eres un experto en mantenimiento de locomotoras y análisis de datos. Explica detalladamente por qué se obtuvieron los siguientes resultados en la predicción de fallos para la variable '{variable_objetivo}':

Contexto:
{contexto}

Proporciona una explicación clara y técnica para cada métrica:
1. **Probabilidad de fallo ({prediccion_reciente.max():.2%})**: ¿Por qué el modelo asigna esta probabilidad? Considera las características más importantes y los valores recientes.
2. **Riesgo de fallo ({'ALTO' if prediccion_reciente.max() > 0.7 else 'MEDIO' if prediccion_reciente.max() > 0.3 else 'BAJO'})**: ¿Por qué se clasifica en este nivel de riesgo? Explica los umbrales y el contexto.
3. **Precisión del modelo ({rf_model.score(X_test, y_test):.2%})**: ¿Qué significa esta precisión? ¿Es confiable? Considera el reporte de clasificación y el tamaño del conjunto de datos ({len(X_test)} registros en prueba).

Usa un tono técnico pero accesible, como si explicaras a un ingeniero de mantenimiento. Sé conciso y enfócate en los factores clave que influyen en cada métrica.
"""

            # Generar explicaciones con Gemini
            try:
                response = self.bot.model.generate_content(prompt_explicacion)
                explicaciones = response.text
            except Exception as e:
                explicaciones = f"⚠️ Error al generar explicaciones: {str(e)}"

            resultado = {
                "variable": variable_objetivo,
                "precision_test": rf_model.score(X_test, y_test),
                "importancia_features": importances.head(5).to_dict('records'),
                "prediccion_reciente": {
                    "probabilidad_fallo": float(prediccion_reciente.max()),
                    "riesgo": "ALTO" if prediccion_reciente.max() > 0.7 else "MEDIO" if prediccion_reciente.max() > 0.3 else "BAJO"
                },
                "reporte_clasificacion": classification_report(y_test, y_pred, output_dict=True),
                "explicaciones": {
                    "probabilidad_fallo": explicaciones.split("2.")[0].strip() if "2." in explicaciones else explicaciones,
                    "riesgo": explicaciones.split("2.")[1].split("3.")[0].strip() if "3." in explicaciones else "",
                    "precision_modelo": explicaciones.split("3.")[1].strip() if "3." in explicaciones else ""
                }
            }
            
            return resultado
            
        except Exception as e:
            return {"error": f"Error en predicción de fallos: {str(e)}"}
    
    def analizar_patrones_operacion(self, df: pd.DataFrame, n_clusters: int = 5) -> dict:
        """
        Analiza patrones de operación usando clustering
        """
        try:
            df_ml = self.preparar_datos_ml(df)
            
            if df_ml.empty:
                return {"error": "No hay suficientes datos para análisis de patrones"}
            
            # Crear matriz de características por variable
            pivot_data = df_ml.pivot_table(
                index='TimeString', 
                columns='VarName', 
                values='VarValue', 
                aggfunc='mean'
            ).fillna(0)
            
            if pivot_data.empty:
                return {"error": "No se pueden crear patrones con los datos disponibles"}
            
            # Reducir dimensionalidad si hay muchas variables
            if pivot_data.shape[1] > 20:
                pca = PCA(n_components=10)
                pivot_data_pca = pca.fit_transform(pivot_data)
                feature_names = [f"PC{i+1}" for i in range(10)]
            else:
                pivot_data_pca = pivot_data.values
                feature_names = pivot_data.columns.tolist()
            
            # Escalar datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(pivot_data_pca)
            
            # Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Analizar clusters
            cluster_analysis = []
            for i in range(n_clusters):
                cluster_mask = clusters == i
                cluster_data = pivot_data_pca[cluster_mask]
                
                if len(cluster_data) > 0:
                    cluster_info = {
                        "cluster": i,
                        "size": int(cluster_mask.sum()),
                        "porcentaje": round((cluster_mask.sum() / len(clusters)) * 100, 2),
                        "caracteristicas_promedio": cluster_data.mean(axis=0).tolist()[:5]  # Solo primeras 5
                    }
                    cluster_analysis.append(cluster_info)
            
            # Identificar cluster actual (últimos datos)
            ultimo_cluster = clusters[-1] if len(clusters) > 0 else 0
            
            resultado = {
                "total_patrones": n_clusters,
                "cluster_actual": int(ultimo_cluster),
                "analisis_clusters": cluster_analysis,
                "distribucion_temporal": pd.Series(clusters).value_counts().to_dict(),
                "recomendacion": self._interpretar_cluster(ultimo_cluster, cluster_analysis)
            }
            
            return resultado
            
        except Exception as e:
            return {"error": f"Error en análisis de patrones: {str(e)}"}
    
    def _interpretar_cluster(self, cluster_id: int, cluster_analysis: list) -> str:
        """
        Interpreta el significado del cluster actual
        """
        if not cluster_analysis:
            return "No se pueden interpretar los patrones"
        
        cluster_info = next((c for c in cluster_analysis if c["cluster"] == cluster_id), None)
        
        if not cluster_info:
            return "Cluster no encontrado"
        
        size = cluster_info["size"]
        porcentaje = cluster_info["porcentaje"]
        
        if porcentaje > 40:
            return f"Operación NORMAL - Patrón común ({porcentaje}% del tiempo)"
        elif porcentaje > 20:
            return f"Operación TÍPICA - Patrón frecuente ({porcentaje}% del tiempo)"
        elif porcentaje > 10:
            return f"Operación ESPECÍFICA - Patrón ocasional ({porcentaje}% del tiempo)"
        else:
            return f"Operación ATÍPICA - Patrón raro ({porcentaje}% del tiempo) - Requiere atención"
    
    def generar_reporte_ml(self, df: pd.DataFrame, variables_clave: list = None) -> str:
        """
        Genera un reporte completo de machine learning
        """
        try:
            print("🤖 Generando reporte de Machine Learning...")
            
            # 1. Detección de anomalías general
            anomalias_result = self.detectar_anomalias_ml(df)
            
            # 2. Análisis de patrones
            patrones_result = self.analizar_patrones_operacion(df)
            
            # 3. Predicción de fallos para variables clave
            predicciones = {}
            if variables_clave:
                for var in variables_clave:
                    pred_result = self.predecir_fallos(df, var)
                    if "error" not in pred_result:
                        predicciones[var] = pred_result
            
            # Construir reporte
            reporte = "🤖 REPORTE DE MACHINE LEARNING\n"
            reporte += "=" * 50 + "\n\n"
            
            # Anomalías
            if "error" not in anomalias_result:
                reporte += "🔍 DETECCIÓN DE ANOMALÍAS\n"
                reporte += f"• Total registros analizados: {anomalias_result['total_registros']}\n"
                reporte += f"• Anomalías detectadas: {anomalias_result['total_anomalias']} ({anomalias_result['porcentaje_anomalias']}%)\n"
                reporte += f"• Variables más afectadas: {list(anomalias_result['variables_afectadas'].keys())[:3]}\n\n"
            
            # Patrones
            if "error" not in patrones_result:
                reporte += "📊 ANÁLISIS DE PATRONES\n"
                reporte += f"• Patrón operacional actual: Cluster {patrones_result['cluster_actual']}\n"
                reporte += f"• Interpretación: {patrones_result['recomendacion']}\n"
                reporte += f"• Patrones identificados: {patrones_result['total_patrones']}\n\n"
            
            # Predicciones
            if predicciones:
                reporte += "🔮 PREDICCIONES DE FALLOS\n"
                for var, pred in predicciones.items():
                    reporte += f"• {var}:\n"
                    reporte += f"  - Riesgo: {pred['prediccion_reciente']['riesgo']}\n"
                    reporte += f"  - Probabilidad: {pred['prediccion_reciente']['probabilidad_fallo']:.1%}\n"
                    reporte += f"  - Precisión modelo: {pred['precision_test']:.1%}\n"
                    reporte += f"  - Explicaciones:\n"
                    reporte += f"    · Probabilidad de fallo: {pred['explicaciones']['probabilidad_fallo']}\n"
                    reporte += f"    · Riesgo: {pred['explicaciones']['riesgo']}\n"
                    reporte += f"    · Precisión del modelo: {pred['explicaciones']['precision_modelo']}\n"
                reporte += "\n"
            
            # Recomendaciones
            reporte += "💡 RECOMENDACIONES\n"
            if "error" not in anomalias_result and anomalias_result['porcentaje_anomalias'] > 15:
                reporte += "• ⚠️ Alto nivel de anomalías detectadas - Revisar sistemas\n"
            
            if predicciones:
                for var, pred in predicciones.items():
                    if pred['prediccion_reciente']['riesgo'] == "ALTO":
                        reporte += f"• 🚨 Atención inmediata requerida en {var}\n"
            
            if "error" not in patrones_result and "ATÍPICA" in patrones_result['recomendacion']:
                reporte += "• 🔍 Operación atípica detectada - Monitorear de cerca\n"
            
            return reporte
            
        except Exception as e:
            return f"❌ Error generando reporte ML: {str(e)}"
    
    def guardar_modelos(self, ruta_base: str = "modelos_ml/"):
        """
        Guarda los modelos entrenados
        """
        try:
            import os
            os.makedirs(ruta_base, exist_ok=True)
            
            for nombre, modelo in self.modelos_entrenados.items():
                ruta_archivo = os.path.join(ruta_base, f"{nombre}.pkl")
                with open(ruta_archivo, 'wb') as f:
                    pickle.dump(modelo, f)
            
            print(f"✅ Modelos guardados en {ruta_base}")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando modelos: {str(e)}")
            return False
    
    def cargar_modelos(self, ruta_base: str = "modelos_ml/"):
        """
        Carga modelos previamente entrenados
        """
        try:
            import os
            
            if not os.path.exists(ruta_base):
                print(f"⚠️ Directorio {ruta_base} no existe")
                return False
            
            for archivo in os.listdir(ruta_base):
                if archivo.endswith('.pkl'):
                    nombre = archivo.replace('.pkl', '')
                    ruta_archivo = os.path.join(ruta_base, archivo)
                    
                    with open(ruta_archivo, 'rb') as f:
                        modelo = pickle.load(f)
                    
                    self.modelos_entrenados[nombre] = modelo
            
            print(f"✅ Modelos cargados desde {ruta_base}")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelos: {str(e)}")
            return False

# Modificar la clase LocomotoraBot original agregando este método:
def agregar_funcionalidad_ml_a_locomotora_bot():
    """
    Agrega funcionalidad ML a la clase LocomotoraBot existente
    """
    def inicializar_ml(self):
        """Inicializa el bot de ML"""
        if not hasattr(self, 'ml_bot'):
            self.ml_bot = LocomotoraMLBot(self)
        return self.ml_bot
    
    def analizar_con_ml(self, df: pd.DataFrame, tipo_analisis: str = "completo") -> str:
        """
        Analiza datos usando machine learning
        
        Args:
            df: DataFrame con datos de sensores
            tipo_analisis: "anomalias", "patrones", "prediccion", "completo"
        """
        ml_bot = self.inicializar_ml()
        
        if tipo_analisis == "anomalias":
            resultado = ml_bot.detectar_anomalias_ml(df)
            return self._formatear_resultado_ml(resultado, "Detección de Anomalías")
        
        elif tipo_analisis == "patrones":
            resultado = ml_bot.analizar_patrones_operacion(df)
            return self._formatear_resultado_ml(resultado, "Análisis de Patrones")
        
        elif tipo_analisis == "prediccion":
            # Usar variables más comunes para predicción
            variables_clave = df['VarName'].value_counts().head(3).index.tolist()
            resultados = {}
            
            for var in variables_clave:
                resultado = ml_bot.predecir_fallos(df, var)
                if "error" not in resultado:
                    resultados[var] = resultado
            
            return self._formatear_predicciones(resultados)
        
        elif tipo_analisis == "completo":
            variables_clave = df['VarName'].value_counts().head(3).index.tolist()
            return ml_bot.generar_reporte_ml(df, variables_clave)
        
        else:
            return "❌ Tipo de análisis no válido. Opciones: anomalias, patrones, prediccion, completo"
    
    def _formatear_resultado_ml(self, resultado: dict, titulo: str) -> str:
        """Formatea resultados de ML para mostrar"""
        if "error" in resultado:
            return f"❌ {titulo}: {resultado['error']}"
        
        output = f"🤖 {titulo.upper()}\n"
        output += "=" * 40 + "\n"
        
        for key, value in resultado.items():
            if isinstance(value, (int, float)):
                output += f"• {key}: {value}\n"
            elif isinstance(value, str):
                output += f"• {key}: {value}\n"
            elif isinstance(value, list) and len(value) > 0:
                output += f"• {key}: {len(value)} items\n"
        
        return output
    
    def _formatear_predicciones(self, predicciones: dict) -> str:
        """Formatea predicciones para mostrar"""
        if not predicciones:
            return "❌ No se pudieron generar predicciones"
        
        output = "🔮 PREDICCIONES DE FALLOS\n"
        output += "=" * 40 + "\n"
        
        for var, pred in predicciones.items():
            output += f"\n📊 {var}:\n"
            output += f"• Riesgo: {pred['prediccion_reciente']['riesgo']}\n"
            output += f"• Probabilidad: {pred['prediccion_reciente']['probabilidad_fallo']:.1%}\n"
            output += f"• Precisión: {pred['precision_test']:.1%}\n"
            output += f"• Explicaciones:\n"
            output += f"  - Probabilidad de fallo: {pred['explicaciones']['probabilidad_fallo']}\n"
            output += f"  - Riesgo: {pred['explicaciones']['riesgo']}\n"
            output += f"  - Precisión del modelo: {pred['explicaciones']['precision_modelo']}\n"
        
        return output
    
    # Agregar métodos a la clase LocomotoraBot
    LocomotoraBot.inicializar_ml = inicializar_ml
    LocomotoraBot.analizar_con_ml = analizar_con_ml
    LocomotoraBot._formatear_resultado_ml = _formatear_resultado_ml
    LocomotoraBot._formatear_predicciones = _formatear_predicciones

# Ejecutar la extensión
agregar_funcionalidad_ml_a_locomotora_bot()

# Función de conveniencia para usar ML directamente
def analizar_locomotora_ml(df: pd.DataFrame, locomotora_bot: LocomotoraBot, 
                          tipo_analisis: str = "completo") -> str:
    """
    Función de conveniencia para análisis ML
    
    Args:
        df: DataFrame con datos de sensores
        locomotora_bot: Instancia de LocomotoraBot
        tipo_analisis: "anomalias", "patrones", "prediccion", "completo"
    
    Returns:
        Resultado del análisis ML
    """
    return locomotora_bot.analizar_con_ml(df, tipo_analisis)

modelo = genai.GenerativeModel('gemini-1.5-flash')
bot = LocomotoraBot(modelo)

# Interfaz mejorada
def consultar_bot(pregunta: str, df: Optional[pd.DataFrame] = None, ruta_csv: Optional[str] = None) -> str:
    if df is None and ruta_csv:
        df = cargar_csv(ruta_csv)
   
    return bot.analisis_con_codigo_sin_ver_df(pregunta, df)