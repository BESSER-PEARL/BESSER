#!/usr/bin/env python3
"""
This script sets up the environment by running the appropriate startup script
based on the detected operating system.
"""

import os
import platform
import subprocess
import argparse
import tempfile

# Get the current directory path (where the script is executed)
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Determine the operating system
os_type = platform.system()

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Set up the environment for the project.")
    parser.add_argument("--nn", action="store_true", help="Install neural network packages")
    return parser.parse_args()

def get_requirements_file(add_nn_packages=False):
    """
    Returns the path to the requirements file.
    If add_nn_packages is True, creates a temporary requirements file with neural network packages added.
    """
    original_req_file = os.path.join(PROJECT_PATH, "requirements.txt")
    
    if not add_nn_packages or not os.path.exists(original_req_file):
        return original_req_file
    
    # Create a temporary file with additional packages
    temp_req_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt')
    
    # Copy original requirements
    if os.path.exists(original_req_file):
        with open(original_req_file, 'r') as orig_file:
            temp_req_file.write(orig_file.read())
    
    # Add neural network packages
    nn_packages = [
        "pillow>=11.1.0",
        "numpy>=2.2.4",
        "torch>=2.6.0"
    ]
    
    temp_req_file.write("\n# Neural network packages\n")
    for package in nn_packages:
        temp_req_file.write(f"{package}\n")
    
    temp_req_file.close()
    return temp_req_file.name

# Run the appropriate script based on the OS
def run_script(add_nn_packages=False):
    """
    Run the appropriate script based on the operating system.
    Optionally add neural network packages.
    """
    req_file = get_requirements_file(add_nn_packages)
    
    if os_type == "Windows":
        print("Detected Windows OS. Running the batch script...")
        batch_script = os.path.join(PROJECT_PATH, "start.bat")
        subprocess.run(["cmd.exe", "/c", batch_script, req_file], check=True)
    elif os_type in ["Linux", "Darwin"]:
        print(f"Detected {os_type} OS. Running the shell script...")
        shell_script = os.path.join(PROJECT_PATH, "start.sh")
        subprocess.run(["bash", shell_script, req_file], check=True)
    else:
        print(f"Unsupported operating system: {os_type}")
    
    # Clean up temporary file if created
    if req_file != os.path.join(PROJECT_PATH, "requirements.txt") and os.path.exists(req_file):
        os.unlink(req_file)

if __name__ == "__main__":
    args = parse_arguments()
    run_script(args.nn)
