"""Test script to verify quantum circuit JSON to BUML conversion"""
import json
import sys
sys.path.insert(0, 'c:\\Users\\sulejmani\\Desktop\\BESSER')

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.quantum_diagram_processor import process_quantum_diagram
from besser.utilities.buml_code_builder.quantum_model_builder import quantum_model_to_code

# Load the JSON file
with open('c:\\Users\\sulejmani\\Desktop\\BESSER\\QUANTUMTEST\\efdsfdsf.json', 'r') as f:
    data = json.load(f)

# Extract quantum circuit model
quantum_model = data['project']['diagrams']['QuantumCircuitDiagram']['model']

print("Processing quantum diagram...")
qc = process_quantum_diagram(quantum_model)

print(f"\nCircuit name: {qc.name}")
print(f"Number of qubits: {qc.num_qubits}")
print(f"Number of operations: {len(qc.operations)}")

print("\nOperations:")
for i, op in enumerate(qc.operations):
    print(f"{i+1}. {op}")

# Generate Python code
output_file = 'c:\\Users\\sulejmani\\Desktop\\BESSER\\QUANTUMTEST\\efdsfdsf_project_fixed.py'
quantum_model_to_code(qc, output_file)
print(f"\n✅ Generated BUML code: {output_file}")
