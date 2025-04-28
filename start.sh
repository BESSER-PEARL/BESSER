#!/bin/bash

# Get the current directory path (where the script is executed)
PROJECT_PATH="$(cd "$(dirname "$0")" && pwd)"

# Get requirements file path from command line argument or use default
REQ_FILE="${1:-$PROJECT_PATH/requirements.txt}"

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
    
    # Set PYTHONPATH and install requirements if requirements file exists
    export PYTHONPATH="$PROJECT_PATH"
    if [ -f "$REQ_FILE" ]; then
        echo "Installing requirements from: $REQ_FILE"
        pip install -r "$REQ_FILE"
    fi
    
    # Keep the shell session open after running the script
    exec "$SHELL"
else
    echo "Error: Unable to activate the virtual environment."
    echo "Make sure Python is installed correctly and the 'besser_venv' folder is created."
fi
