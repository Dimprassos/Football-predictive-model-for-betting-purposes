@echo off
echo ==========================================
echo Setting up the Football Prediction Project
echo ==========================================

echo.
echo Checking for existing virtual environment...
IF NOT EXIST "venv\Scripts\activate" (
    echo Creating virtual environment venv...
    python -m venv venv
) ELSE (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Upgrading pip to the latest version...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
:: Δοκιμάζει όλες τις πιθανές διαδρομές για να βρει το αρχείο
IF EXIST "requirements\requirements.txt" (
    pip install -r requirements\requirements.txt
) ELSE IF EXIST "requirements\base.txt" (
    pip install -r requirements\base.txt
) ELSE IF EXIST "requirments\requirements.txt" (
    pip install -r requirments\requirements.txt
) ELSE (
    echo [ERROR] Could not find requirements file in the folder.
    echo Please make sure the file is inside the requirements folder.
)

echo.
echo ==========================================
echo Setup Complete! 
echo To run the model, simply type:
echo python main.py
echo ==========================================
cmd /k