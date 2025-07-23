from flask import Flask, render_template, request, session, flash
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename
from IA.ia import LocomotoraBot, consultar_bot, analizar_locomotora_ml
from IA.datos import cargar_csv
from IA.ia import modelo
import re

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Clave segura para sesiones
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Crear carpeta uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inicializar el bot
bot = LocomotoraBot(modelo)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Variables para el template - IMPORTANTE: actualizar desde session SIEMPRE
    context = {
        'locomotora_seleccionada': session.get('locomotora_seleccionada'),
        'archivos_cargados': session.get('archivos_cargados'),
        'datos_disponibles': session.get('datos_disponibles', False),
        'mensaje': None,
        'tipo_mensaje': None,
        'respuesta': None,
        'resultado_ml': None,
        'error': None,
        'chart_data': None
    }
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        try:
            if action == 'seleccionar_locomotora':
                locomotora = request.form.get('locomotora')
                if locomotora in ['ALCO', 'GAIA', 'GR12', 'GT22']:
                    session['locomotora_seleccionada'] = locomotora
                    # Limpiar datos anteriores cuando se cambia de locomotora
                    session.pop('archivos_cargados', None)
                    session.pop('datos_disponibles', None)
                    session.pop('archivo_principal', None)
                    session.pop('archivo_secundario', None)
                    
                    # CRÍTICO: Actualizar el contexto inmediatamente
                    context['locomotora_seleccionada'] = locomotora
                    context['mensaje'] = f'Locomotora {locomotora} seleccionada correctamente'
                    context['tipo_mensaje'] = 'success'
                else:
                    context['error'] = 'Debes seleccionar una locomotora válida'
            
            elif action == 'cargar_datos':
                locomotora = session.get('locomotora_seleccionada')
                if not locomotora:
                    context['error'] = 'Primero debes seleccionar una locomotora'
                    return render_template('index.html', **context)
                
                # Mantener locomotora seleccionada en contexto
                context['locomotora_seleccionada'] = locomotora
                
                file1 = request.files.get('file1')
                file2 = request.files.get('file2')
                archivos_procesados = []
                
                if file1 and file1.filename:
                    if file1.filename.endswith('.csv'):
                        filename1 = secure_filename(f"{locomotora}_{file1.filename}")
                        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
                        file1.save(filepath1)
                        try:
                            df = cargar_csv(filepath1)
                            if df.empty:
                                context['error'] = 'El archivo CSV principal está vacío o no es válido'
                                return render_template('index.html', **context)
                            session['archivo_principal'] = filepath1
                            archivos_procesados.append(f"Principal: {file1.filename} ({len(df)} filas)")
                        except Exception as e:
                            context['error'] = f'Error al leer archivo principal: {str(e)}'
                            return render_template('index.html', **context)
                    else:
                        context['error'] = 'El archivo principal debe ser un CSV'
                        return render_template('index.html', **context)
                else:
                    context['error'] = 'Debes seleccionar un archivo CSV principal'
                    return render_template('index.html', **context)
                
                if file2 and file2.filename and file2.filename.endswith('.csv'):
                    filename2 = secure_filename(f"{locomotora}_{file2.filename}")
                    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
                    file2.save(filepath2)
                    try:
                        df = cargar_csv(filepath1, filepath2)
                        if df.empty:
                            context['error'] = 'Error al combinar archivos: datos vacíos o no válidos'
                            return render_template('index.html', **context)
                        session['archivo_secundario'] = filepath2
                        archivos_procesados.append(f"Secundario: {file2.filename} ({len(df)} filas)")
                    except Exception as e:
                        context['mensaje'] = f'Archivo secundario no válido: {str(e)}'
                        context['tipo_mensaje'] = 'error'
                
                if archivos_procesados:
                    session['archivos_cargados'] = ', '.join(archivos_procesados)
                    session['datos_disponibles'] = True
                    # CRÍTICO: Actualizar contexto inmediatamente
                    context['archivos_cargados'] = session['archivos_cargados']
                    context['datos_disponibles'] = True
                    context['mensaje'] = f'Datos cargados exitosamente para {locomotora}'
                    context['tipo_mensaje'] = 'success'
                    context['chart_data'] = generar_grafico_mejorado(df)
            
            elif action == 'consultar_ia':
                locomotora = session.get('locomotora_seleccionada')
                pregunta = request.form.get('pregunta')
                usar_codigo = 'usar_codigo' in request.form
                
                # Mantener datos en contexto
                context['locomotora_seleccionada'] = locomotora
                context['archivos_cargados'] = session.get('archivos_cargados')
                context['datos_disponibles'] = session.get('datos_disponibles', False)
                
                if not locomotora:
                    context['error'] = 'Primero debes seleccionar una locomotora'
                elif not session.get('datos_disponibles'):
                    context['error'] = 'Primero debes cargar los datos CSV'
                elif not pregunta:
                    context['error'] = 'Debes escribir una pregunta'
                else:
                    archivo_principal = session.get('archivo_principal')
                    archivo_secundario = session.get('archivo_secundario')
                    df = cargar_csv(archivo_principal, archivo_secundario)
                    if df.empty:
                        context['error'] = 'Error al cargar los datos para la consulta'
                        return render_template('index.html', **context)
                    
                    # Procesar consulta
                    respuesta = procesar_consulta_ia(locomotora, pregunta, usar_codigo, df)
                    context['respuesta'] = respuesta
                    context['resultado_ml'] = analizar_locomotora_ml(df, bot, "completo")
                    context['chart_data'] = generar_grafico_mejorado(df)
        
        except Exception as e:
            context['error'] = f'Error interno: {str(e)}'
            # Mantener datos existentes en caso de error
            context['locomotora_seleccionada'] = session.get('locomotora_seleccionada')
            context['archivos_cargados'] = session.get('archivos_cargados')
            context['datos_disponibles'] = session.get('datos_disponibles', False)
    
    else:
        # GET request - mostrar datos desde sesión
        context['locomotora_seleccionada'] = session.get('locomotora_seleccionada')
        context['archivos_cargados'] = session.get('archivos_cargados')
        context['datos_disponibles'] = session.get('datos_disponibles', False)
    
    return render_template('index.html', **context)

def procesar_consulta_ia(locomotora, pregunta, usar_codigo, df):
    """Procesa la consulta usando el bot de IA"""
    try:
        respuesta = bot.analisis_con_codigo_sin_ver_df(pregunta, df, locomotora) if usar_codigo else consultar_bot(pregunta, df)
        # Corregir errores en la respuesta
        respuesta = corregir_respuesta(respuesta)
        return respuesta
    except Exception as e:
        return f"Error al procesar consulta: {str(e)}"

def generar_grafico_mejorado(df):
    """Genera la configuración mejorada de Chart.js para el gráfico"""
    try:
        if df.empty:
            return None

        # Verificar columnas necesarias
        required_columns = ["TimeString", "VarValue"]
        if not all(col in df.columns for col in required_columns):
            print(f"Columnas faltantes. Disponibles: {df.columns.tolist()}")
            return None

        df = df.copy()
        
        # Convertir tiempo
        df["TimeString"] = pd.to_datetime(df["TimeString"], errors='coerce')
        df = df.dropna(subset=["TimeString", "VarValue"]).sort_values("TimeString")
        
        # Colores mejorados para mejor contraste
        colors = [
            '#667eea', '#f093fb', '#4facfe', '#43e97b',
            '#fa709a', '#ffecd2', '#a8edea', '#d299c2',
            '#89f7fe', '#66a6ff', '#f78ca0', '#96e6a1'
        ]
        
        datasets = []
        
        # Procesar por variable si existe la columna VarName
        if "VarName" in df.columns:
            variable_groups = df.groupby("VarName")
            
            for idx, (nombre_variable, subdf) in enumerate(variable_groups):
                if len(subdf) > 5:  # Solo variables con suficientes datos
                    # Preparar datos para el gráfico
                    time_labels = subdf["TimeString"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                    values = subdf["VarValue"].tolist()
                    
                    # Crear dataset con nombre completo preservado
                    dataset = {
                        "label": nombre_variable[:50] + "..." if len(nombre_variable) > 50 else nombre_variable,
                        "fullLabel": nombre_variable,  # Nombre completo para tooltips
                        "data": values,
                        "borderColor": colors[idx % len(colors)],
                        "backgroundColor": colors[idx % len(colors)] + "20",
                        "fill": False,
                        "tension": 0.1,
                        "borderWidth": 2,
                        "pointRadius": 2,
                        "pointHoverRadius": 5,
                        "pointBackgroundColor": colors[idx % len(colors)],
                        "pointBorderColor": "#ffffff",
                        "pointBorderWidth": 2
                    }
                    datasets.append(dataset)
        else:
            # Si no hay VarName, usar todos los datos como una serie
            if len(df) > 5:
                time_labels = df["TimeString"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                values = df["VarValue"].tolist()
                
                dataset = {
                    "label": "Valores de Variables",
                    "fullLabel": "Valores de Variables de la Locomotora",
                    "data": values,
                    "borderColor": colors[0],
                    "backgroundColor": colors[0] + "20",
                    "fill": False,
                    "tension": 0.1,
                    "borderWidth": 2,
                    "pointRadius": 2,
                    "pointHoverRadius": 5,
                    "pointBackgroundColor": colors[0],
                    "pointBorderColor": "#ffffff",
                    "pointBorderWidth": 2
                }
                datasets.append(dataset)
        
        if not datasets:
            return None

        # Generar etiquetas de tiempo únicas
        all_times = df["TimeString"].drop_duplicates().sort_values()
        time_labels = all_times.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        
        # Limitar etiquetas para mejor visualización
        if len(time_labels) > 20:
            step = len(time_labels) // 20
            time_labels = time_labels[::step]

        chart_config = {
            "type": "line",
            "data": {
                "labels": time_labels,
                "datasets": datasets
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "interaction": {
                    "intersect": False,
                    "mode": "index"
                },
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Análisis Temporal de Variables",
                        "font": {
                            "size": 18,
                            "weight": "bold"
                        }
                    },
                    "legend": {
                        "display": True,
                        "position": "top",
                        "labels": {
                            "usePointStyle": True,
                            "padding": 15,
                            "font": {"size": 11}
                        }
                    },
                    "tooltip": {
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "titleColor": "white",
                        "bodyColor": "white",
                        "borderColor": "rgba(102, 126, 234, 0.8)",
                        "borderWidth": 1,
                        "cornerRadius": 8,
                        "displayColors": True
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Tiempo",
                            "font": {"size": 14, "weight": "bold"}
                        },
                        "grid": {"color": "rgba(0, 0, 0, 0.1)"},
                        "ticks": {
                            "maxTicksLimit": 10,
                            "font": {"size": 10}
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Valor",
                            "font": {"size": 14, "weight": "bold"}
                        },
                        "grid": {"color": "rgba(0, 0, 0, 0.1)"},
                        "ticks": {"font": {"size": 10}}
                    }
                },
                "elements": {
                    "line": {"tension": 0.1, "borderWidth": 2},
                    "point": {"radius": 2, "hoverRadius": 5}
                }
            }
        }
        
        return chart_config
        
    except Exception as e:
        print(f"Error al generar gráfico mejorado: {str(e)}")
        return None

def corregir_respuesta(respuesta):
    """Corrige problemas específicos en la respuesta"""
    # Corregir porcentaje incorrecto (28651% -> 28651 registros)
    respuesta = re.sub(r"El (\d+)% de los registros", r"Se encontraron \1 registros", respuesta)
    # Corregir explicaciones incompletas de ML
    if "Precisión del modelo: 50 y" in respuesta:
        respuesta = respuesta.replace("Precisión del modelo: 50 y", "Precisión del modelo: (información incompleta, revisar modelo)")
    return respuesta

if __name__ == '__main__':
    app.run(debug=True)