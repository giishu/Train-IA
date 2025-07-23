@echo off
chcp 65001 >nul 2>&1
color 0A
title Análisis de Locomotoras - Sistema de Monitoreo

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    ANÁLISIS DE LOCOMOTORAS                   ║
echo ║                   Sistema de Monitoreo v2.0                 ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 🚂 Iniciando sistema de análisis...
echo.

REM Verificar si Python está instalado
echo [1/5] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python no está instalado o no está en el PATH
    echo.
    echo 📋 Instrucciones de instalación:
    echo    1. Descarga Python desde: https://www.python.org/downloads/
    echo    2. Durante la instalación, marca "Add Python to PATH"
    echo    3. Reinicia este programa
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Python detectado correctamente
)

REM Verificar dependencias
echo [2/5] Verificando dependencias...
python -c "import flask, pandas, werkzeug" >nul 2>&1
if errorlevel 1 (
    echo 📦 Instalando dependencias necesarias...
    echo    - Flask (servidor web)
    echo    - Pandas (análisis de datos)  
    echo    - Werkzeug (utilidades web)
    echo.
    pip install flask pandas werkzeug --quiet --disable-pip-version-check
    if errorlevel 1 (
        echo ❌ Error al instalar dependencias
        echo 💡 Intenta ejecutar como administrador
        pause
        exit /b 1
    )
    echo ✅ Dependencias instaladas correctamente
) else (
    echo ✅ Todas las dependencias están disponibles
)

REM Verificar estructura de carpetas
echo [3/5] Verificando estructura del proyecto...
if not exist "IA" (
    echo ⚠️  Advertencia: Carpeta 'IA' no encontrada
    echo 📁 Creando estructura básica...
    mkdir IA >nul 2>&1
)
if not exist "Uploads" (
    echo 📁 Creando carpeta de uploads...
    mkdir Uploads >nul 2>&1
)
echo ✅ Estructura del proyecto verificada

REM Cambiar al directorio del script
echo [4/5] Configurando entorno...
cd /d "%~dp0"
echo ✅ Directorio de trabajo configurado

REM Mostrar información del sistema
echo [5/5] Iniciando servidor...
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                      INFORMACIÓN DEL SISTEMA                ║
echo ╠══════════════════════════════════════════════════════════════╣
echo ║ 🌐 URL Local:      http://localhost:5000                    ║
echo ║ 🌐 URL Red:        http://%COMPUTERNAME%:5000               ║
echo ║ 📂 Carpeta Datos:  %cd%\Uploads                            ║
echo ║ 🔧 Modo Debug:     Activado                                 ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 📋 INSTRUCCIONES DE USO:
echo    1. El navegador se abrirá automáticamente en 3 segundos
echo    2. Selecciona el tipo de locomotora (ALCO, GAIA, GR12, GT22)
echo    3. Carga tus archivos CSV con los datos
echo    4. Realiza consultas usando IA para análisis avanzado
echo.
echo ⚠️  IMPORTANTE: 
echo    • No cierres esta ventana mientras uses la aplicación
echo    • Para detener el servidor, presiona Ctrl+C
echo    • Los archivos se guardan en la carpeta 'Uploads'
echo.
echo 🔄 Iniciando en 3 segundos...

REM Contador visual
for /L %%i in (3,-1,1) do (
    echo ⏳ %%i...
    timeout /t 1 /nobreak >nul
)

REM Abrir navegador automáticamente
echo.
echo 🌐 Abriendo navegador...
start "" http://localhost:5000

REM Ejecutar Flask con manejo de errores
echo.
echo ════════════════════════════════════════════════════════════════
echo                    SERVIDOR INICIADO
echo ════════════════════════════════════════════════════════════════
echo.

python app2.py
set FLASK_ERROR=%ERRORLEVEL%

echo.
echo ════════════════════════════════════════════════════════════════
echo                    SERVIDOR DETENIDO
echo ════════════════════════════════════════════════════════════════

if %FLASK_ERROR% neq 0 (
    echo.
    echo ❌ El servidor se cerró con errores
    echo 💡 Posibles soluciones:
    echo    • Verifica que el archivo app2.py existe
    echo    • Revisa que las carpetas IA/ existen
    echo    • Comprueba los permisos de archivo
    echo    • Ejecuta como administrador si es necesario
    echo.
) else (
    echo.
    echo ✅ Servidor cerrado correctamente
    echo 👋 ¡Gracias por usar el Sistema de Análisis de Locomotoras!
    echo.
)

echo Presiona cualquier tecla para salir...
pause >nul