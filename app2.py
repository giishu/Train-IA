from flask import Flask, render_template, request, session
import pandas as pd
import os
from werkzeug.utils import secure_filename
from IA.ia import LocomotoraBot, consultar_bot, analizar_locomotora_ml, modelo
from IA.datos import cargar_csv
import re
import json
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['TEMP_FOLDER'] = 'Temp'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

bot = LocomotoraBot(modelo)

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        'locomotora_seleccionada': session.get('locomotora_seleccionada'),
        'archivos_cargados': session.get('archivos_cargados'),
        'datos_disponibles': session.get('datos_disponibles', False),
        'mensaje': None,
        'tipo_mensaje': None,
        'respuesta': None,
        'resultado_ml': None,
        'error': None,
        'chart_data': None,
        'fecha_min': None,
        'fecha_max': None,
        'tabla_csv': None,
        'dias_disponibles': None,
        'horas_disponibles': None,
        'variables_disponibles': None,
        'current_page': session.get('current_page', 1),
        'total_pages': session.get('total_pages', 1),
        'per_page': 100
    }

    if request.method == 'POST':
        action = request.form.get('action')
        try:
            if action == 'seleccionar_locomotora':
                locomotora = request.form.get('locomotora')
                if locomotora in ['ALCO', 'GAIA', 'GR12', 'GT22']:
                    session['locomotora_seleccionada'] = locomotora
                    session.pop('archivos_cargados', None)
                    session.pop('datos_disponibles', None)
                    session.pop('archivo_principal', None)
                    session.pop('archivo_secundario', None)
                    session.pop('temp_df_path', None)
                    session.pop('filtros', None)
                    session.pop('current_page', None)
                    session.pop('total_pages', None)
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

                context['locomotora_seleccionada'] = locomotora
                file1 = request.files.get('file1')
                file2 = request.files.get('file2')
                archivos_procesados = []

                if file1 and file1.filename.endswith('.csv'):
                    filename1 = secure_filename(f"{locomotora}_{file1.filename}")
                    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
                    file1.save(filepath1)
                    df = cargar_csv(filepath1)
                    if df.empty:
                        context['error'] = 'El archivo CSV principal está vacío o no es válido'
                        return render_template('index.html', **context)
                    session['archivo_principal'] = filepath1
                    archivos_procesados.append(f"Principal: {file1.filename} ({len(df)} filas)")
                else:
                    context['error'] = 'Debes seleccionar un archivo CSV principal válido'
                    return render_template('index.html', **context)

                if file2 and file2.filename.endswith('.csv'):
                    filename2 = secure_filename(f"{locomotora}_{file2.filename}")
                    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
                    file2.save(filepath2)
                    df = cargar_csv(filepath1, filepath2)
                    if df.empty:
                        context['error'] = 'Error al combinar archivos'
                        return render_template('index.html', **context)
                    session['archivo_secundario'] = filepath2
                    archivos_procesados.append(f"Secundario: {file2.filename} ({len(df)} filas)")

                temp_df_path = os.path.join(app.config['TEMP_FOLDER'], f"{locomotora}_temp_df.parquet")
                df.to_parquet(temp_df_path)
                session['temp_df_path'] = temp_df_path

                df["TimeString"] = pd.to_datetime(df["TimeString"], errors="coerce")
                context['fecha_min'] = df["TimeString"].min()
                context['fecha_max'] = df["TimeString"].max()

                if "VarName" in df.columns and "VarValue" in df.columns:
                    try:
                        df_pivot = df.pivot_table(index="TimeString", columns="VarName", values="VarValue", aggfunc=lambda x: ', '.join(map(str, x)))
                        df_pivot.index = df_pivot.index.strftime("%Y-%m-%d %H:%M:%S")
                        df_pivot = df_pivot.fillna("-")
                        df_pivot = df_pivot.dropna(how='all')
                        total_rows = len(df_pivot)
                        context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                        context['current_page'] = 1
                        session['total_pages'] = context['total_pages']
                        session['current_page'] = context['current_page']
                        start_idx = 0
                        end_idx = context['per_page']
                        df_pivot = df_pivot.iloc[start_idx:end_idx]
                        df_pivot.index.name = None
                        headers = "<thead><tr><th>TimeString / VarName</th>" + "".join(f"<th>{col}</th>" for col in df_pivot.columns) + "</tr></thead>"
                        body = df_pivot.to_html(index=True, header=False, na_rep="-", border=0, classes='')
                        inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                        context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                    except Exception as e:
                        context['error'] = f"Error al crear la tabla pivote: {str(e)}. Puede haber problemas con los datos."
                        total_rows = len(df)
                        context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                        context['current_page'] = 1
                        session['total_pages'] = context['total_pages']
                        session['current_page'] = context['current_page']
                        start_idx = 0
                        end_idx = context['per_page']
                        body = df.iloc[start_idx:end_idx].to_html(index=False, header=True, na_rep="-", border=0, classes='')
                        inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                        headers = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df.columns) + "</tr></thead>"
                        context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                else:
                    context['error'] = f"El CSV no contiene las columnas 'VarName' o 'VarValue' necesarias para la tabla pivote. Columnas presentes: {list(df.columns)}."
                    total_rows = len(df)
                    context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                    context['current_page'] = 1
                    session['total_pages'] = context['total_pages']
                    session['current_page'] = context['current_page']
                    start_idx = 0
                    end_idx = context['per_page']
                    body = df.iloc[start_idx:end_idx].to_html(index=False, header=True, na_rep="-", border=0, classes='')
                    inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                    headers = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df.columns) + "</tr></thead>"
                    context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"

                context['dias_disponibles'] = sorted(df["TimeString"].dt.date.astype(str).dropna().unique())
                context['horas_disponibles'] = sorted(df["TimeString"].dt.hour.astype(str).str.zfill(2).dropna().unique())
                if "VarName" in df.columns:
                    context['variables_disponibles'] = sorted(df["VarName"].dropna().unique())

                session['archivos_cargados'] = ', '.join(archivos_procesados)
                session['datos_disponibles'] = True
                context['archivos_cargados'] = session['archivos_cargados']
                context['datos_disponibles'] = True
                context['mensaje'] = f'Datos cargados exitosamente para {locomotora}'
                context['tipo_mensaje'] = 'success'
                context['chart_data'] = generar_grafico(df)

            elif action == "filtrar_datos":
                temp_df_path = session.get('temp_df_path')
                if not temp_df_path or not os.path.exists(temp_df_path):
                    context['error'] = 'No hay datos cargados. Por favor carga un CSV primero.'
                    return render_template('index.html', **context)

                df = pd.read_parquet(temp_df_path)
                df["TimeString"] = pd.to_datetime(df["TimeString"], errors="coerce")

                filas_validas = len(df[df["TimeString"].notna()])
                if filas_validas == 0:
                    context['error'] = f"No hay valores válidos en la columna 'TimeString'. Asegúrate de que las fechas sean correctas en el CSV."
                    return render_template('index.html', **context)

                context['dias_disponibles'] = sorted(df["TimeString"].dt.date.astype(str).dropna().unique())
                context['horas_disponibles'] = sorted(df["TimeString"].dt.hour.astype(str).str.zfill(2).dropna().unique())
                if "VarName" in df.columns:
                    context['variables_disponibles'] = sorted(df["VarName"].dropna().unique())

                filtros = {
                    'dias': request.form.getlist("filtro_dias"),
                    'horas': [str(h).zfill(2) for h in request.form.getlist("filtro_horas")],
                    'variables': request.form.getlist("filtro_variables")  # Cambio a lista para múltiples variables
                }
                session['filtros'] = filtros

                print(f"Filtros recibidos: {filtros}")
                print(f"Filas con TimeString válido: {filas_validas}")
                print(f"Días disponibles en DF: {context['dias_disponibles'][:5]}")
                print(f"Horas disponibles en DF: {context['horas_disponibles'][:5]}")
                print(f"Valores únicos de TimeString (primeras 5 filas): {df['TimeString'].head().tolist()}")

                if not filtros['dias'] and not filtros['horas'] and not filtros['variables']:
                    context['error'] = "Debes seleccionar al menos un filtro (día, hora o variables)."
                    return render_template('index.html', **context)

                df_filtrado = df.copy()
                if filtros['dias']:
                    df_filtrado = df_filtrado[df_filtrado["TimeString"].dt.date.astype(str).isin(filtros['dias'])]
                if filtros['horas']:
                    df_filtrado = df_filtrado[df_filtrado["TimeString"].dt.hour.astype(str).str.zfill(2).isin(filtros['horas'])]
                if filtros['variables']:
                    df_filtrado = df_filtrado[df_filtrado["VarName"].isin(filtros['variables'])]

                print(f"Filas en df_filtrado final: {len(df_filtrado)}")

                if df_filtrado.empty:
                    context['error'] = f"No se encontraron datos que coincidan con los filtros seleccionados: Días={filtros['dias']}, Horas={filtros['horas']}, Variables={filtros['variables'] or 'Ninguna'}. Días disponibles: {context['dias_disponibles'][:5]}, Horas disponibles: {context['horas_disponibles'][:5]}."
                    return render_template('index.html', **context)

                if "VarName" not in df_filtrado.columns or "VarValue" not in df_filtrado.columns:
                    context['error'] = f"El CSV filtrado no contiene las columnas 'VarName' o 'VarValue'. Columnas presentes: {list(df_filtrado.columns)}."
                    return render_template('index.html', **context)

                try:
                    df_pivot = df_filtrado.pivot_table(index="TimeString", columns="VarName", values="VarValue", aggfunc=lambda x: ', '.join(map(str, x)))
                    df_pivot.index = df_pivot.index.strftime("%Y-%m-%d %H:%M:%S")
                    df_pivot = df_pivot.fillna("-")
                    df_pivot = df_pivot.dropna(how='all')
                    total_rows = len(df_pivot)
                    context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                    context['current_page'] = 1
                    session['total_pages'] = context['total_pages']
                    session['current_page'] = context['current_page']
                    start_idx = 0
                    end_idx = context['per_page']
                    df_pivot = df_pivot.iloc[start_idx:end_idx]
                    df_pivot.index.name = None
                    headers = "<thead><tr><th>TimeString / VarName</th>" + "".join(f"<th>{col}</th>" for col in df_pivot.columns) + "</tr></thead>"
                    body = df_pivot.to_html(index=True, header=False, na_rep="-", border=0, classes='')
                    inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                    context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                except Exception as e:
                    context['error'] = f"Error al crear la tabla pivote: {str(e)}. Puede haber problemas con los datos filtrados."
                    total_rows = len(df_filtrado)
                    context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                    context['current_page'] = 1
                    session['total_pages'] = context['total_pages']
                    session['current_page'] = context['current_page']
                    start_idx = 0
                    end_idx = context['per_page']
                    body = df_filtrado.iloc[start_idx:end_idx].to_html(index=False, header=True, na_rep="-", border=0, classes='')
                    inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                    headers = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df_filtrado.columns) + "</tr></thead>"
                    context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"

                context['chart_data'] = generar_grafico(df_filtrado)
                context['fecha_min'] = df_filtrado["TimeString"].min()
                context['fecha_max'] = df_filtrado["TimeString"].max()

            elif action == 'resetear_filtros':
                session['filtros'] = {}
                session['current_page'] = 1
                temp_df_path = session.get('temp_df_path')
                if temp_df_path and os.path.exists(temp_df_path):
                    df = pd.read_parquet(temp_df_path)
                    df["TimeString"] = pd.to_datetime(df["TimeString"], errors="coerce")

                    if "VarName" in df.columns and "VarValue" in df.columns:
                        try:
                            df_pivot = df.pivot_table(index="TimeString", columns="VarName", values="VarValue", aggfunc=lambda x: ', '.join(map(str, x)))
                            df_pivot.index = df_pivot.index.strftime("%Y-%m-%d %H:%M:%S")
                            df_pivot = df_pivot.fillna("-")
                            df_pivot = df_pivot.dropna(how='all')
                            total_rows = len(df_pivot)
                            context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                            session['total_pages'] = context['total_pages']
                            start_idx = 0
                            end_idx = context['per_page']
                            df_pivot = df_pivot.iloc[start_idx:end_idx]
                            df_pivot.index.name = None
                            headers = "<thead><tr><th>TimeString / VarName</th>" + "".join(f"<th>{col}</th>" for col in df_pivot.columns) + "</tr></thead>"
                            body = df_pivot.to_html(index=True, header=False, na_rep="-", border=0, classes='')
                            inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                            context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                        except Exception as e:
                            context['error'] = f"Error al crear la tabla pivote: {str(e)}. Puede haber problemas con los datos."
                            total_rows = len(df)
                            context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                            session['total_pages'] = context['total_pages']
                            start_idx = 0
                            end_idx = context['per_page']
                            body = df.iloc[start_idx:end_idx].to_html(index=False, header=True, na_rep="-", border=0, classes='')
                            inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                            headers = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df.columns) + "</tr></thead>"
                            context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                    else:
                        context['error'] = f"El CSV no contiene las columnas 'VarName' o 'VarValue' necesarias para la tabla pivote. Columnas presentes: {list(df.columns)}."
                        total_rows = len(df)
                        context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                        session['total_pages'] = context['total_pages']
                        start_idx = 0
                        end_idx = context['per_page']
                        body = df.iloc[start_idx:end_idx].to_html(index=False, header=True, na_rep="-", border=0, classes='')
                        inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                        headers = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df.columns) + "</tr></thead>"
                        context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"

                    context['chart_data'] = generar_grafico(df)
                    context['fecha_min'] = df["TimeString"].min()
                    context['fecha_max'] = df["TimeString"].max()

                    context['dias_disponibles'] = sorted(df["TimeString"].dt.date.astype(str).dropna().unique())
                    context['horas_disponibles'] = sorted(df["TimeString"].dt.hour.astype(str).str.zfill(2).dropna().unique())
                    if "VarName" in df.columns:
                        context['variables_disponibles'] = sorted(df["VarName"].dropna().unique())

                context['mensaje'] = 'Filtros reseteados correctamente.'
                context['tipo_mensaje'] = 'success'

            elif action == 'cambiar_pagina':
                temp_df_path = session.get('temp_df_path')
                if not temp_df_path or not os.path.exists(temp_df_path):
                    context['error'] = 'No hay datos cargados. Por favor carga un CSV primero.'
                    return render_template('index.html', **context)

                df = pd.read_parquet(temp_df_path)
                df["TimeString"] = pd.to_datetime(df["TimeString"], errors="coerce")

                context['dias_disponibles'] = sorted(df["TimeString"].dt.date.astype(str).dropna().unique())
                context['horas_disponibles'] = sorted(df["TimeString"].dt.hour.astype(str).str.zfill(2).dropna().unique())
                if "VarName" in df.columns:
                    context['variables_disponibles'] = sorted(df["VarName"].dropna().unique())

                filtros = session.get('filtros', {})
                df_filtrado = df.copy()
                if filtros.get('dias'):
                    df_filtrado = df_filtrado[df_filtrado["TimeString"].dt.date.astype(str).isin(filtros['dias'])]
                if filtros.get('horas'):
                    df_filtrado = df_filtrado[df_filtrado["TimeString"].dt.hour.astype(str).str.zfill(2).isin(filtros['horas'])]
                if filtros.get('variables'):
                    df_filtrado = df_filtrado[df_filtrado["VarName"].isin(filtros['variables'])]

                try:
                    page = int(request.form.get('page', 1))
                    if page < 1:
                        page = 1
                except ValueError:
                    page = 1

                if "VarName" in df_filtrado.columns and "VarValue" in df_filtrado.columns and not df_filtrado.empty:
                    try:
                        df_pivot = df_filtrado.pivot_table(index="TimeString", columns="VarName", values="VarValue", aggfunc=lambda x: ', '.join(map(str, x)))
                        df_pivot.index = df_pivot.index.strftime("%Y-%m-%d %H:%M:%S")
                        df_pivot = df_pivot.fillna("-")
                        df_pivot = df_pivot.dropna(how='all')
                        total_rows = len(df_pivot)
                        context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                        context['current_page'] = min(page, context['total_pages'])
                        session['total_pages'] = context['total_pages']
                        session['current_page'] = context['current_page']
                        start_idx = (context['current_page'] - 1) * context['per_page']
                        end_idx = start_idx + context['per_page']
                        df_pivot = df_pivot.iloc[start_idx:end_idx]
                        df_pivot.index.name = None
                        headers = "<thead><tr><th>TimeString / VarName</th>" + "".join(f"<th>{col}</th>" for col in df_pivot.columns) + "</tr></thead>"
                        body = df_pivot.to_html(index=True, header=False, na_rep="-", border=0, classes='')
                        inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                        context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                    except Exception as e:
                        context['error'] = f"Error al crear la tabla pivote: {str(e)}. Puede haber problemas con los datos filtrados."
                        total_rows = len(df_filtrado)
                        context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                        context['current_page'] = min(page, context['total_pages'])
                        session['total_pages'] = context['total_pages']
                        session['current_page'] = context['current_page']
                        start_idx = (context['current_page'] - 1) * context['per_page']
                        end_idx = start_idx + context['per_page']
                        body = df_filtrado.iloc[start_idx:end_idx].to_html(index=False, header=True, na_rep="-", border=0, classes='')
                        inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                        headers = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df_filtrado.columns) + "</tr></thead>"
                        context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                else:
                    context['error'] = f"El CSV filtrado no contiene las columnas 'VarName' o 'VarValue' o está vacío. Columnas presentes: {list(df_filtrado.columns)}."
                    total_rows = len(df_filtrado)
                    context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                    context['current_page'] = min(page, context['total_pages'])
                    session['total_pages'] = context['total_pages']
                    session['current_page'] = context['current_page']
                    start_idx = (context['current_page'] - 1) * context['per_page']
                    end_idx = start_idx + context['per_page']
                    body = df_filtrado.iloc[start_idx:end_idx].to_html(index=False, header=True, na_rep="-", border=0, classes='')
                    inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                    headers = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df_filtrado.columns) + "</tr></thead>"
                    context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"

                context['chart_data'] = generar_grafico(df_filtrado)
                context['fecha_min'] = df_filtrado["TimeString"].min()
                context['fecha_max'] = df_filtrado["TimeString"].max()

            elif action == 'consultar_ia':
                temp_df_path = session.get('temp_df_path')
                if not temp_df_path or not os.path.exists(temp_df_path):
                    context['error'] = 'No hay datos cargados. Por favor carga un CSV primero.'
                    return render_template('index.html', **context)

                locomotora = session.get('locomotora_seleccionada')
                pregunta = request.form.get('pregunta')
                # Forzamos usar_codigo como True ya que siempre estará activado
                usar_codigo = True

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
                    df = pd.read_parquet(temp_df_path)
                    df["TimeString"] = pd.to_datetime(df["TimeString"], errors="coerce")
                    if df.empty:
                        context['error'] = 'Error al cargar los datos para la consulta'
                        return render_template('index.html', **context)
                    respuesta = procesar_consulta_ia(locomotora, pregunta, usar_codigo, df)
                    context['respuesta'] = respuesta
                    # Temporalmente desactivamos el resultado ML
                    # context['resultado_ml'] = analizar_locomotora_ml(df, bot, "completo")

                    # Recalcular filtros para que estén disponibles
                    context['dias_disponibles'] = sorted(df["TimeString"].dt.date.astype(str).dropna().unique())
                    context['horas_disponibles'] = sorted(df["TimeString"].dt.hour.astype(str).str.zfill(2).dropna().unique())
                    if "VarName" in df.columns:
                        context['variables_disponibles'] = sorted(df["VarName"].dropna().unique())

                    df_filtrado = df.copy()
                    filtros = session.get('filtros', {})
                    if filtros.get('dias'):
                        df_filtrado = df_filtrado[df_filtrado["TimeString"].dt.date.astype(str).isin(filtros['dias'])]
                    if filtros.get('horas'):
                        df_filtrado = df_filtrado[df_filtrado["TimeString"].dt.hour.astype(str).str.zfill(2).isin(filtros['horas'])]
                    if filtros.get('variables'):
                        df_filtrado = df_filtrado[df_filtrado["VarName"].isin(filtros['variables'])]

                    if "VarName" in df_filtrado.columns and "VarValue" in df_filtrado.columns:
                        try:
                            df_pivot = df_filtrado.pivot_table(index="TimeString", columns="VarName", values="VarValue", aggfunc=lambda x: ', '.join(map(str, x)))
                            df_pivot.index = df_pivot.index.strftime("%Y-%m-%d %H:%M:%S")
                            df_pivot = df_pivot.fillna("-")
                            df_pivot = df_pivot.dropna(how='all')
                            total_rows = len(df_pivot)
                            context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                            context['current_page'] = session.get('current_page', 1)
                            session['total_pages'] = context['total_pages']
                            start_idx = (context['current_page'] - 1) * context['per_page']
                            end_idx = start_idx + context['per_page']
                            df_pivot = df_pivot.iloc[start_idx:end_idx]
                            df_pivot.index.name = None
                            headers = "<thead><tr><th>TimeString / VarName</th>" + "".join(f"<th>{col}</th>" for col in df_pivot.columns) + "</tr></thead>"
                            body = df_pivot.to_html(index=True, header=False, na_rep="-", border=0, classes='')
                            inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                            context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                        except Exception as e:
                            context['error'] = f"Error al crear la tabla pivote: {str(e)}. Puede haber problemas con los datos."
                            total_rows = len(df_filtrado)
                            context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                            context['current_page'] = session.get('current_page', 1)
                            session['total_pages'] = context['total_pages']
                            start_idx = (context['current_page'] - 1) * context['per_page']
                            end_idx = start_idx + context['per_page']
                            body = df_filtrado.iloc[start_idx:end_idx].to_html(index=False, header=True, na_rep="-", border=0, classes='')
                            inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                            headers = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df_filtrado.columns) + "</tr></thead>"
                            context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                    else:
                        context['error'] = f"El CSV no contiene las columnas 'VarName' o 'VarValue' necesarias para la tabla pivote. Columnas presentes: {list(df_filtrado.columns)}."
                        total_rows = len(df_filtrado)
                        context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                        context['current_page'] = session.get('current_page', 1)
                        session['total_pages'] = context['total_pages']
                        start_idx = (context['current_page'] - 1) * context['per_page']
                        end_idx = start_idx + context['per_page']
                        body = df_filtrado.iloc[start_idx:end_idx].to_html(index=False, header=True, na_rep="-", border=0, classes='')
                        inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                        headers = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df_filtrado.columns) + "</tr></thead>"
                        context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"

                    context['fecha_min'] = df_filtrado["TimeString"].min()
                    context['fecha_max'] = df_filtrado["TimeString"].max()

            elif action == 'graficar':
                temp_df_path = session.get('temp_df_path')
                if not temp_df_path or not os.path.exists(temp_df_path):
                    context['error'] = 'No hay datos cargados. Por favor carga un CSV primero.'
                else:
                    df = pd.read_parquet(temp_df_path)
                    df["TimeString"] = pd.to_datetime(df["TimeString"], errors="coerce")
                    if df.empty:
                        context['error'] = 'Error al cargar los datos para graficar'
                    else:
                        # Obtener variables seleccionadas (máximo 3)
                        variables = request.form.getlist('graficar_variables')
                        if not variables:
                            context['error'] = 'Debes seleccionar al menos una variable para graficar'
                        elif len(variables) > 3:
                            variables = variables[:3]  # Limitar a 3 variables
                            context['mensaje'] = 'Se limitó la selección a 3 variables'
                            context['tipo_mensaje'] = 'info'

                        # Obtener días y horas seleccionados
                        dias = request.form.getlist('graficar_dias')
                        horas = [str(h).zfill(2) for h in request.form.getlist('graficar_horas')]

                        # Filtrar datos por rango de tiempo
                        df_filtrado = df.copy()
                        if dias:
                            df_filtrado = df_filtrado[df_filtrado["TimeString"].dt.date.astype(str).isin(dias)]
                        if horas:
                            df_filtrado = df_filtrado[df_filtrado["TimeString"].dt.hour.astype(str).str.zfill(2).isin(horas)]
                        if df_filtrado.empty and (dias or horas):
                            context['error'] = 'No hay datos en el rango de tiempo seleccionado'
                        else:
                            # Generar gráfico
                            context['grafico'] = generar_grafico_variables(df_filtrado, variables)

                        # Preservar estado del contexto
                        context['locomotora_seleccionada'] = session.get('locomotora_seleccionada')
                        context['archivos_cargados'] = session.get('archivos_cargados')
                        context['datos_disponibles'] = session.get('datos_disponibles', False)
                        context['dias_disponibles'] = sorted(df["TimeString"].dt.date.astype(str).dropna().unique())
                        context['horas_disponibles'] = sorted(df["TimeString"].dt.hour.astype(str).str.zfill(2).dropna().unique())
                        if "VarName" in df.columns:
                            context['variables_disponibles'] = sorted(df["VarName"].dropna().unique())
                        if "VarName" in df_filtrado.columns and "VarValue" in df_filtrado.columns:
                            try:
                                df_pivot = df_filtrado.pivot_table(index="TimeString", columns="VarName", values="VarValue", aggfunc=lambda x: ', '.join(map(str, x)))
                                df_pivot.index = df_pivot.index.strftime("%Y-%m-%d %H:%M:%S")
                                df_pivot = df_pivot.fillna("-")
                                df_pivot = df_pivot.dropna(how='all')
                                total_rows = len(df_pivot)
                                context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                                context['current_page'] = session.get('current_page', 1)
                                session['total_pages'] = context['total_pages']
                                start_idx = (context['current_page'] - 1) * context['per_page']
                                end_idx = start_idx + context['per_page']
                                df_pivot = df_pivot.iloc[start_idx:end_idx]
                                df_pivot.index.name = None
                                headers = "<thead><tr><th>TimeString / VarName</th>" + "".join(f"<th>{col}</th>" for col in df_pivot.columns) + "</tr></thead>"
                                body = df_pivot.to_html(index=True, header=False, na_rep="-", border=0, classes='')
                                inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                                context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                            except Exception as e:
                                context['error'] = f"Error al crear la tabla pivote: {str(e)}. Puede haber problemas con los datos."
                                total_rows = len(df_filtrado)
                                context['total_pages'] = (total_rows + context['per_page'] - 1) // context['per_page']
                                context['current_page'] = session.get('current_page', 1)
                                session['total_pages'] = context['total_pages']
                                start_idx = (context['current_page'] - 1) * context['per_page']
                                end_idx = start_idx + context['per_page']
                                body = df_filtrado.iloc[start_idx:end_idx].to_html(index=False, header=True, na_rep="-", border=0, classes='')
                                inner_body = re.sub(r'<table[^>]*>', '', body).replace('</table>', '').strip()
                                headers = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df_filtrado.columns) + "</tr></thead>"
                                context['tabla_csv'] = f"<table class='csv-table'>{headers}{inner_body}</table>"
                        context['fecha_min'] = df_filtrado["TimeString"].min() if not df_filtrado.empty else None
                        context['fecha_max'] = df_filtrado["TimeString"].max() if not df_filtrado.empty else None
        except Exception as e:
            context['error'] = f'Error interno: {str(e)}'

    return render_template('index.html', **context)

def procesar_consulta_ia(locomotora, pregunta, usar_codigo, df):
    try:
        respuesta = bot.analisis_con_codigo_sin_ver_df(pregunta, df, locomotora) if usar_codigo else consultar_bot(pregunta, df)
        return corregir_respuesta(respuesta)
    except Exception as e:
        return f"Error al procesar consulta: {str(e)}"

def generar_grafico(df):
    try:
        if df.empty or "TimeString" not in df.columns or "VarValue" not in df.columns:
            return None
        df = df.copy()
        df["TimeString"] = pd.to_datetime(df["TimeString"], errors='coerce')
        df = df.dropna(subset=["TimeString", "VarValue"]).sort_values("TimeString")
        
        df["VarValue"] = pd.to_numeric(df["VarValue"], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["VarValue"])
        
        colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
        datasets = []
        times = df["TimeString"].dt.strftime("%Y-%m-%d %H:%M:%S").unique().tolist()
        if "VarName" in df.columns:
            for idx, (nombre, subdf) in enumerate(df.groupby("VarName")):
                if len(subdf) > 10:
                    data_dict = dict(zip(subdf["TimeString"].dt.strftime("%Y-%m-%d %H:%M:%S"), subdf["VarValue"]))
                    data = [data_dict.get(t, None) for t in times]
                    datasets.append({
                        "label": str(nombre),
                        "data": [float(x) if x is not None else None for x in data],
                        "borderColor": colors[idx % len(colors)],
                        "backgroundColor": colors[idx % len(colors)] + "80",
                        "fill": False
                    })
        else:
            if len(df) > 10:
                data = df["VarValue"].tolist()
                datasets.append({
                    "label": "VarValue",
                    "data": [float(x) for x in data],
                    "borderColor": colors[0],
                    "backgroundColor": colors[0] + "80",
                    "fill": False
                })
        if not datasets:
            return None
        chart_config = {
            "type": "line",
            "data": {
                "labels": times,
                "datasets": datasets
            },
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": "Tiempo"}},
                    "y": {"title": {"display": True, "text": "Valor"}}
                },
                "plugins": {"legend": {"display": True}},
                "responsive": True
            }
        }
        return json.dumps(chart_config, ensure_ascii=False)
    except Exception as e:
        print(f"Error en generar_grafico: {str(e)}")
        return None

def corregir_respuesta(respuesta):
    respuesta = re.sub(r"El (\d+)% de los registros", r"Se encontraron \1 registros", respuesta)
    return respuesta

def generar_grafico_variables(df, variables):
    try:
        if df.empty or "TimeString" not in df.columns or "VarValue" not in df.columns or "VarName" not in df.columns:
            return None

        # Filtrar solo las variables seleccionadas
        df = df[df["VarName"].isin(variables)].copy()
        df["TimeString"] = pd.to_datetime(df["TimeString"], errors='coerce')
        df["VarValue"] = pd.to_numeric(df["VarValue"], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["VarValue", "TimeString"])

        if df.empty:
            return None

        # Obtener tiempos únicos ordenados
        times = sorted(df["TimeString"].dt.strftime("%Y-%m-%d %H:%M:%S").unique().tolist())
        datasets = []
        colors = ['#3b82f6', '#ef4444', '#10b981']  # Colores distintivos para 3 variables

        for idx, var in enumerate(variables):
            subdf = df[df["VarName"] == var]
            data_dict = dict(zip(subdf["TimeString"].dt.strftime("%Y-%m-%d %H:%M:%S"), subdf["VarValue"]))
            data = [data_dict.get(t, None) for t in times]
            datasets.append({
                "label": var,
                "data": [float(x) if x is not None else None for x in data],
                "borderColor": colors[idx],
                "backgroundColor": colors[idx] + "80",
                "fill": False
            })

        if not datasets:
            return None

        # Configuración del gráfico
        grafico = {
            "type": "line",  # Especificamos 'line' como tipo por defecto
            "data": {
                "labels": times,
                "datasets": datasets
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False
            }
        }
        return grafico
    except Exception as e:
        print(f"Error en generar_grafico_variables: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True)