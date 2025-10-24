"""
Test script to regenerate React output from projecteditor.json
"""
import json
import sys
from pathlib import Path

# Ensure project modules are importable
sys.path.insert(0, ".")


class MockDockerModule:
    @staticmethod
    def from_env(*args, **kwargs):
        raise RuntimeError("Docker module is not available in this environment.")


sys.modules.setdefault("docker", MockDockerModule())


class MockBOCL:
    class OCLWrapper:
        pass


sys.modules["bocl"] = MockBOCL()
sys.modules["bocl.OCLWrapper"] = MockBOCL

from besser.generators.web_app.web_app import WebAppGenerator
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram,
)

# Load the project
BASE_DIR = Path(__file__).resolve().parent
input_path = BASE_DIR / "output" / "TEST_GUI.json"

with open(input_path, "r", encoding="utf-8") as f:
    project_data = json.load(f)

# Extract diagrams
project = project_data["project"]
class_diagram = project["diagrams"]["ClassDiagram"]
gui_diagram = project["diagrams"]["GUINoCodeDiagram"]

# Process class diagram to BUML
print("Processing class diagram...")
buml_model = process_class_diagram(class_diagram)

# Process GUI diagram to BUML
print("Processing GUI diagram...")
gui_model = process_gui_diagram(
    gui_diagram["model"],
    class_diagram["model"],
    buml_model,
)

# Generate React TypeScript output
print("Generating React TypeScript output...")
output_dir = str(BASE_DIR / "react_output_test")
generator = WebAppGenerator(buml_model, gui_model, output_dir=output_dir)
generator.generate()

print(f"\n[OK] Generation complete! Output in: {output_dir}")
