@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ═══════════════════════════════════════
echo  SEmail Spam Classifier (Spam vs. Ham)
echo ═══════════════════════════════════════
echo.
REM Check Python exists
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Install Python 3.10+ and try again.
    pause
    exit /b 1
)
REM Check required files exist
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found in this folder.
    pause
    exit /b 1
)
if not exist "app.py" (
    echo ERROR: app.py not found in this folder.
    pause
    exit /b 1
)
REM Create venv if missing
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)
set "PYTHON=.venv\Scripts\python.exe"
echo Upgrading pip...
"%PYTHON%" -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo Installing requirements...
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)
echo Downloading NLTK stopwords...
"%PYTHON%" -c "import nltk; nltk.download('stopwords', quiet=True)" >nul 2>&1
echo Starting server at http://127.0.0.1:5000
start "" "http://127.0.0.1:5000"
"%PYTHON%" app.py
endlocal
pause