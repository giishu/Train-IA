import streamlit as st
import pandas as pd
import chardet
from IA.datos import conectar_db, registrar_archivo
from IA.ia import LocomotoraBot
import google.generativeai as genai
import re

# Configuración de la API de Gemini
genai.configure(api_key='AIzaSyA2PipvauvVPmrGQz-Hn7nhu_VcWHypeEo')

# Inicializar el bot
modelo = genai.GenerativeModel('gemini-1.5-flash')
bot = LocomotoraBot(modelo)

# Configuración de la página
st.set_page_config(page_title="Análisis de Locomotoras", layout="wide")
st.title("📊 Análisis de Datos de Locomotoras con IA")

# Función mejorada para cargar CSV
def cargar_csv_streamlit(file):
    try:
        raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        file.seek(0)
        
        df = pd.read_csv(
            file,
            sep=';',
            quotechar='"',
            encoding=encoding,
            parse_dates=['TimeString'],
            dayfirst=True,
            on_bad_lines='skip',
            dtype={'VarValue': str},
            low_memory=False
        )
        
        if 'VarValue' in df.columns:
            df['VarValue'] = pd.to_numeric(df['VarValue'].str.replace(',', '.'), errors='coerce')
        if 'Time_ms' in df.columns:
            df['Time_ms'] = pd.to_numeric(df['Time_ms'].str.replace(',', '.'), errors='coerce')
        
        df = df.dropna(subset=['VarName', 'VarValue'])
        with conectar_db() as conn:
            registrar_archivo(file.name, df)
        
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {str(e)}")
        return pd.DataFrame()

# Función para formatear estadísticas
def format_stats(text):
    # Extraer y limpiar estadísticas
    stats = {}
    if 'count' in text and 'mean' in text and 'std' in text and 'min' in text and 'max' in text:
        pattern = r'count (\d+\.?\d*) mean (\d+\.?\d*) std (\d+\.?\d*) min (\d+\.?\d*) max (\d+\.?\d*)'
        match = re.search(pattern, text)
        if match:
            stats = {
                'Cantidad': int(float(match.group(1))),
                'Promedio': round(float(match.group(2)), 2),
                'Desviación': round(float(match.group(3)), 2),
                'Mínima': int(float(match.group(4))),
                'Máxima': int(float(match.group(5)))
            }
            return f"| **Métrica**       | **Valor** |\n|--------------------|-----------|\n" + "\n".join([f"| {k:<18} | {v} |" for k, v in stats.items()])
    return text

# Función para limpiar y formatear la respuesta completa
def format_ai_response(text):
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(key in line for key in ['Resumen estadístico', 'Estado de validación', 'Conclusión']):
            formatted_lines.append(f"### {line}")
        elif 'No se encontraron anomalías' in line:
            formatted_lines.append(f"- {line}")
        else:
            formatted_lines.append(format_stats(line) if 'Resumen estadístico' in '\n'.join(formatted_lines[-2:]) else f"- {line}")
    return '\n'.join(formatted_lines)

# Selección de locomotora
st.header("Selección de Locomotora")
locomotora_seleccionada = st.selectbox(
    "Seleccione el tipo de locomotora",
    ["ALCO", "GAIA", "GR12", "GT22"],
    index=0
)

# Carga de archivo CSV
st.header("Cargar Archivo CSV")
uploaded_file = st.file_uploader("Seleccione un archivo CSV", type=["csv"])

df = None
if uploaded_file is not None:
    with st.spinner("Cargando archivo CSV..."):
        df = cargar_csv_streamlit(uploaded_file)
        if not df.empty:
            st.success(f"✅ Archivo cargado: {len(df)} registros")
            st.subheader("Vista Previa de los Datos")
            st.dataframe(df.head(10))
        else:
            st.error("🔴 No se pudieron cargar los datos. Verifique el formato del archivo.")

# Consulta a la IA
st.header("Consulta a la IA")
pregunta = st.text_input("Ingrese su pregunta (ej. '¿Cuál es el promedio de temperatura?')")
usar_codigo = st.checkbox("¿Generar y ejecutar código para el análisis?", value=True)

if st.button("Consultar IA") and pregunta and df is not None:
    with st.spinner("Procesando consulta..."):
        try:
            df_processed = df.copy()
            df_processed['VarValue'] = pd.to_numeric(df_processed['VarValue'], errors='coerce')
            df_processed['TimeString'] = pd.to_datetime(df_processed['TimeString'], errors='coerce')
            df_processed = df_processed.dropna(subset=['VarName', 'VarValue'])
            
            resultado = bot.analisis_con_codigo_sin_ver_df(pregunta, df_processed, locomotora_seleccionada)
            
            # Formatear respuesta como markdown
            formatted_result = format_ai_response(resultado)
            st.markdown(formatted_result, unsafe_allow_html=False)
        except Exception as e:
            st.error(f"❌ Error al procesar la consulta: {str(e)}")

# Reporte de Machine Learning
st.header("Reporte de Machine Learning")
if st.button("Generar Reporte ML") and df is not None:
    with st.spinner("Generando reporte de Machine Learning..."):
        try:
            df_processed = df.copy()
            df_processed['VarValue'] = pd.to_numeric(df_processed['VarValue'], errors='coerce')
            df_processed['TimeString'] = pd.to_datetime(df_processed['TimeString'], errors='coerce')
            df_processed = df_processed.dropna(subset=['VarName', 'VarValue'])
            
            reporte_ml = bot.analizar_con_ml(df_processed, tipo_analisis="completo")
            
            # Formatear reporte como markdown con recomendaciones mejoradas
            lines = reporte_ml.split('\n')
            formatted_reporte = ["### Reporte de Machine Learning"]
            in_recommendations = False
            
            for line in lines:
                if line.startswith('💡 RECOMENDACIONES'):
                    in_recommendations = True
                    formatted_reporte.append(line)
                    # Añadir recomendaciones basadas en datos disponibles
                    if 'total_anomalias' in locals() and locals()['total_anomalias'] > 50:
                        formatted_reporte.append("• ⚠️ Alto nivel de anomalías detectadas - Revisar sistemas inmediatamente.")
                    if 'prediccion_reciente' in locals() and any(pred['riesgo'] == "ALTO" for pred in locals()['prediccion_reciente'].values()):
                        formatted_reporte.append("• 🚨 Atención inmediata requerida en variables de alto riesgo.")
                    if 'recomendacion' in locals() and "ATÍPICA" in locals()['recomendacion']:
                        formatted_reporte.append("• 🔍 Operación atípica detectada - Monitorear de cerca.")
                elif in_recommendations and not line.strip():
                    in_recommendations = False
                else:
                    formatted_reporte.append(line)
            
            st.markdown("\n".join(formatted_reporte), unsafe_allow_html=False)
        except Exception as e:
            st.error(f"❌ Error al generar el reporte ML: {str(e)}")