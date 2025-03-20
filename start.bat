@echo off

REM Get the current directory path (where the .bat file is executed)
set "PROJECT_PATH=%~dp0"

REM Navigate to the project directory
cd %PROJECT_PATH%

REM Check if the "besser_venv" folder exists
if not exist "besser_venv" (
    echo No virtual environment found. Creating the virtual environment...
    python -m venv besser_venv
)

REM Check if the activation script exists, otherwise display an error
if exist "besser_venv\Scripts\Activate.ps1" (
    REM Set up the PowerShell command to activate the virtual environment, set PYTHONPATH, and install requirements
    echo Activating the virtual environment in PowerShell...

    REM Launch a new PowerShell session with the environment activated, bypassing execution policy
    powershell -NoExit -ExecutionPolicy Bypass -Command ^
        "cd '%PROJECT_PATH%';" ^
        ". .\besser_venv\Scripts\Activate.ps1;" ^
        "$env:PYTHONPATH='%PROJECT_PATH%';" ^
        "if (Test-Path requirements.txt) { pip install -r requirements.txt }"

) else (
    echo Error: Unable to activate the virtual environment.
    echo Make sure Python is installed correctly and the "besser_venv" folder is created.
)

pause

