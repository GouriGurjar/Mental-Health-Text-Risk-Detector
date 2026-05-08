@echo off
title MindGuard Setup ^& Launcher
color 0A

echo.
echo ============================================
echo   MindGuard — Mental Health Risk Detector
echo   Auto Setup ^& Launch
echo ============================================
echo.

:: Step 1 — Install all libraries
echo [1/3] Installing required libraries...
echo.
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib imbalanced-learn --quiet --upgrade

if %errorlevel% neq 0 (
    echo.
    echo  ERROR: pip install failed. Make sure Python is installed
    echo  and added to PATH. Download from https://python.org
    pause
    exit /b
)

echo.
echo  All libraries installed successfully!
echo.

:: Step 2 — Retrain model (optional — comment out if model.pkl already exists)
echo [2/3] Training model (this may take 1-2 minutes)...
echo.
python train_model.py

if %errorlevel% neq 0 (
    echo.
    echo  ERROR: train_model.py failed. Check the error above.
    pause
    exit /b
)

echo.
echo  Model trained and saved!
echo.

:: Step 3 — Launch Streamlit app
echo [3/3] Launching MindGuard app in your browser...
echo.
echo  App will open at: http://localhost:8501
echo  Press Ctrl+C in this window to stop the server.
echo.
streamlit run app.py

pause
