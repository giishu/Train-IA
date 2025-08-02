@echo off
echo ========================================
echo     ANALISIS DE LOCOMOTORAS 
echo ========================================
echo.
echo Iniciando aplicacion...
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo.
    echo Por favor instala Python desde: https://www.python.org/downloads/
    echo Asegurate de marcar "Add Python to PATH" durante la instalacion
    pause
    exit /b 1
)

REM Verificar si Flask está instalado
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo 📦 Instalando dependencias necesarias...
    pip install flask pandas werkzeug
    if errorlevel 1 (
        echo Error al instalar dependencias
        pause
        exit /b 1
    )
)

REM Cambiar al directorio del script
cd /d "%~dp0"

REM Ejecutar la aplicación Flask
echo Iniciando servidor web...
echo.
echo La aplicacion se abrira en: http://localhost:5000
echo.
echo ⚠️  IMPORTANTE: No cierres esta ventana mientras uses la aplicacion
echo    Para cerrar la aplicacion, presiona Ctrl+C aqui
echo.

REM Abrir navegador automáticamente después de 3 segundos
timeout /t 3 /nobreak >nul
start http://localhost:5000

REM Ejecutar Flask
python app2.py

pause