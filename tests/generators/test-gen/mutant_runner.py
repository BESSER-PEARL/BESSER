import subprocess
import shutil
import os
from pathlib import Path

MUTANT_DIR = "mutants"
PROJECT_DIR = "project"
TARGET_FILE = "account.py"   # file under test

killed = 0
survived = 0

mutants = list(Path(MUTANT_DIR).glob("*.py"))

for mutant in mutants:

    print(f"Testing mutant: {mutant}")

    target_path = Path(PROJECT_DIR) / TARGET_FILE

    backup = target_path.with_suffix(".bak")
    shutil.copy(target_path, backup)

    shutil.copy(mutant, target_path)

    result = subprocess.run(
        ["pytest", "-q"],
        capture_output=True
    )

    if result.returncode != 0:
        killed += 1
        print("Mutant killed")
    else:
        survived += 1
        print("Mutant survived")

    shutil.move(backup, target_path)

print("\nMutation Results")
print("-------------------")
print("Total mutants:", len(mutants))
print("Killed:", killed)
print("Survived:", survived)

score = killed / len(mutants) * 100
print("Mutation Score:", round(score, 2), "%")