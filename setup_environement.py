#!/usr/bin/env python3
"""
This script sets up the environment by running the appropriate startup script
based on the detected operating system.
"""

import os
import platform
import subprocess

# Get the current directory path (where the script is executed)
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Determine the operating system
os_type = platform.system()

# Run the appropriate script based on the OS
def run_script():
    """
    Run the appropriate script based on the operating system.
    """
    if os_type == "Windows":
        print("Detected Windows OS. Running the batch script...")
        batch_script = os.path.join(PROJECT_PATH, "start.bat")
        subprocess.run(["cmd.exe", "/c", batch_script], check=True)
    elif os_type in ["Linux", "Darwin"]:
        print(f"Detected {os_type} OS. Running the shell script...")
        shell_script = os.path.join(PROJECT_PATH, "start.sh")
        subprocess.run(["bash", shell_script], check=True)
    else:
        print(f"Unsupported operating system: {os_type}")

if __name__ == "__main__":
    run_script()
