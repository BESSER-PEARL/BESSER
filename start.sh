#!/bin/bash

# Get the current directory path (where the script is executed)
PROJECT_PATH="$(cd "$(dirname "$0")" && pwd)"

# Navigate to the project directory
cd "$PROJECT_PATH"

# Check if the "venv" folder exists
if [ ! -d "besser_venv" ]; then
    echo "No virtual environment found. Creating the virtual environment..."
    python3 -m venv besser_venv
fi

# Check if the activation script exists, otherwise display an error
if [ -f "besser_venv/bin/activate" ]; then
    # Activate the virtual environment
    source "besser_venv/bin/activate"
    
    # Set PYTHONPATH and install requirements if requirements.txt exists
    export PYTHONPATH="$PROJECT_PATH"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    
    # Keep the shell session open after running the script
    exec "$SHELL"
else
    echo "Error: Unable to activate the virtual environment."
    echo "Make sure Python is installed correctly and the 'besser_venv' folder is created."
fi
