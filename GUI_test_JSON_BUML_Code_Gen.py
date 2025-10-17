"""
Test script to regenerate React output from projecteditor.json
"""
import json
import sys
# Prevent bocl import error
sys.path.insert(0, '.')

# Mock the bocl module to avoid import error
import importlib
class MockBOCL:
    class OCLWrapper:
        pass

sys.modules['bocl'] = MockBOCL()
sys.modules['bocl.OCLWrapper'] = MockBOCL

from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_class_diagram,
    process_gui_diagram
)
from besser.generators.react_ts import ReactTSGenerator

# Load the project
with open('output/v2project.json', 'r', encoding='utf-8') as f:
    project_data = json.load(f)

# Extract diagrams
project = project_data['project']
class_diagram = project['diagrams']['ClassDiagram']
gui_diagram = project['diagrams']['GUINoCodeDiagram']

# Process class diagram to BUML
print("Processing class diagram...")
buml_model = process_class_diagram(class_diagram['model'])

# Process GUI diagram to BUML
print("Processing GUI diagram...")
gui_model = process_gui_diagram(gui_diagram['model'], class_diagram['model'], buml_model)

# Generate React TypeScript output
print("Generating React TypeScript output...")
output_dir = 'react_output_test'
generator = ReactTSGenerator(buml_model, gui_model, output_dir=output_dir)
generator.generate()

print(f"\n✅ Generation complete! Output in: {output_dir}")
print("\nCheck the generated files:")
print(f"  - {output_dir}/frontend/src/generated/guiModel.ts")
print(f"  - {output_dir}/frontend/src/App.tsx")
print(f"  - {output_dir}/frontend/src/utils/eventHandlers.ts")
