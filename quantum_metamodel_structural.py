"""
BUML Structural Class Diagram for Quantum Metamodel
This file defines the Quantum Circuit metamodel using BUML's structural classes.
"""

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Multiplicity, Enumeration, EnumerationLiteral,
    DataType, PrimitiveDataType, BinaryAssociation, Generalization, Method, Parameter
)

def create_quantum_metamodel():
    """
    Creates the quantum circuit metamodel as a BUML structural domain model.
    
    Returns:
        DomainModel: The quantum circuit domain model
    """
    
    # Create the domain model
    model = DomainModel(name="QuantumCircuitMetamodel")
    
    # ============================================
    # Enumeration: ControlState
    # ============================================
    control_state_enum = Enumeration(
        name="ControlState",
        literals={
            EnumerationLiteral("CONTROL"),
            EnumerationLiteral("ANTI_CONTROL")
        }
    )
    
    # ============================================
    # Classes (Abstract Base Classes)
    # ============================================
    
    # QuantumCircuit class (inherits from Model)
    quantum_circuit_class = Class(
        name="QuantumCircuit",
        is_abstract=False,
        attributes={
            Property(name="name", type="str"),
        }
    )
    
    # QuantumRegister class
    quantum_register_class = Class(
        name="QuantumRegister",
        is_abstract=False,
        attributes={
            Property(name="name", type="str"),
            Property(name="size", type="int")
        }
    )
    
    # ClassicalRegister class
    classical_register_class = Class(
        name="ClassicalRegister",
        is_abstract=False,
        attributes={
            Property(name="name", type="str"),
            Property(name="size", type="int")
        }
    )
    
    # QuantumOperation class (abstract)
    quantum_operation_class = Class(
        name="QuantumOperation",
        is_abstract=True,
        attributes={
            Property(name="name", type="str"),
            Property(name="target_qubits", type="str"),  # List[int] represented as str
            Property(name="control_qubits", type="str"),  # List[int] represented as str
            Property(name="control_states", type="str")   # List[ControlState] represented as str
        }
    )
    
    # Gate class (abstract, inherits from QuantumOperation)
    gate_class = Class(
        name="Gate",
        is_abstract=True
    )
    
    # ============================================
    # Concrete Gate Classes
    # ============================================
    
    # PrimitiveGate
    primitive_gate_class = Class(
        name="PrimitiveGate",
        is_abstract=False,
        attributes={
            Property(name="type_name", type="str")
        }
    )
    
    # ParametricGate
    parametric_gate_class = Class(
        name="ParametricGate",
        is_abstract=False,
        attributes={
            Property(name="type_name", type="str"),
            Property(name="parameter", type="float")
        }
    )
    
    # ArithmeticGate
    arithmetic_gate_class = Class(
        name="ArithmeticGate",
        is_abstract=False,
        attributes={
            Property(name="operation", type="str"),
            Property(name="input_qubits", type="str")  # List[int]
        }
    )
    
    # ModularArithmeticGate
    modular_arithmetic_gate_class = Class(
        name="ModularArithmeticGate",
        is_abstract=False,
        attributes={
            Property(name="operation", type="str"),
            Property(name="modulo", type="int"),
            Property(name="input_qubits", type="str")  # List[int]
        }
    )
    
    # ComparisonGate
    comparison_gate_class = Class(
        name="ComparisonGate",
        is_abstract=False,
        attributes={
            Property(name="operation", type="str"),
            Property(name="input_qubits", type="str")  # List[int]
        }
    )
    
    # QFTGate
    qft_gate_class = Class(
        name="QFTGate",
        is_abstract=False,
        attributes={
            Property(name="inverse", type="bool")
        }
    )
    
    # PhaseGradientGate
    phase_gradient_gate_class = Class(
        name="PhaseGradientGate",
        is_abstract=False,
        attributes={
            Property(name="inverse", type="bool")
        }
    )
    
    # InputGate
    input_gate_class = Class(
        name="InputGate",
        is_abstract=False,
        attributes={
            Property(name="input_type", type="str"),
            Property(name="value", type="int")
        }
    )
    
    # OrderGate
    order_gate_class = Class(
        name="OrderGate",
        is_abstract=False,
        attributes={
            Property(name="order_type", type="str")
        }
    )
    
    # ScalarGate
    scalar_gate_class = Class(
        name="ScalarGate",
        is_abstract=False,
        attributes={
            Property(name="scalar_type", type="str")
        }
    )
    
    # CustomGate
    custom_gate_class = Class(
        name="CustomGate",
        is_abstract=False,
        attributes={
            Property(name="definition", type="str")  # QuantumCircuit reference
        }
    )
    
    # TimeDependentGate
    time_dependent_gate_class = Class(
        name="TimeDependentGate",
        is_abstract=False,
        attributes={
            Property(name="type_name", type="str"),
            Property(name="parameter_expr", type="str")
        }
    )
    
    # SpacerGate
    spacer_gate_class = Class(
        name="SpacerGate",
        is_abstract=False
    )
    
    # ============================================
    # Specific Primitive Gates
    # ============================================
    
    hadamard_gate_class = Class(name="HadamardGate", is_abstract=False)
    pauli_x_gate_class = Class(name="PauliXGate", is_abstract=False)
    pauli_y_gate_class = Class(name="PauliYGate", is_abstract=False)
    pauli_z_gate_class = Class(name="PauliZGate", is_abstract=False)
    swap_gate_class = Class(name="SwapGate", is_abstract=False)
    s_gate_class = Class(name="SGate", is_abstract=False)
    t_gate_class = Class(name="TGate", is_abstract=False)
    
    # ============================================
    # Specific Parametric Gates
    # ============================================
    
    rx_gate_class = Class(name="RXGate", is_abstract=False)
    ry_gate_class = Class(name="RYGate", is_abstract=False)
    rz_gate_class = Class(name="RZGate", is_abstract=False)
    phase_gate_class = Class(name="PhaseGate", is_abstract=False)
    
    # ============================================
    # Operation Classes
    # ============================================
    
    # DisplayOperation
    display_operation_class = Class(
        name="DisplayOperation",
        is_abstract=False,
        attributes={
            Property(name="display_type", type="str")
        }
    )
    
    # Measurement
    measurement_class = Class(
        name="Measurement",
        is_abstract=False,
        attributes={
            Property(name="output_bit", type="int"),
            Property(name="basis", type="str")
        }
    )
    
    # PostSelection
    post_selection_class = Class(
        name="PostSelection",
        is_abstract=False,
        attributes={
            Property(name="value", type="int"),
            Property(name="basis", type="str")
        }
    )
    
    # ============================================
    # Add all types to the model
    # ============================================
    
    model.types = {
        # Enumerations
        control_state_enum,
        
        # Main classes
        quantum_circuit_class,
        quantum_register_class,
        classical_register_class,
        quantum_operation_class,
        gate_class,
        
        # Gate subclasses
        primitive_gate_class,
        parametric_gate_class,
        arithmetic_gate_class,
        modular_arithmetic_gate_class,
        comparison_gate_class,
        qft_gate_class,
        phase_gradient_gate_class,
        input_gate_class,
        order_gate_class,
        scalar_gate_class,
        custom_gate_class,
        time_dependent_gate_class,
        spacer_gate_class,
        
        # Specific gates
        hadamard_gate_class,
        pauli_x_gate_class,
        pauli_y_gate_class,
        pauli_z_gate_class,
        swap_gate_class,
        s_gate_class,
        t_gate_class,
        rx_gate_class,
        ry_gate_class,
        rz_gate_class,
        phase_gate_class,
        
        # Operations
        display_operation_class,
        measurement_class,
        post_selection_class
    }
    
    # ============================================
    # Associations
    # ============================================
    
    # QuantumCircuit has QuantumRegisters
    qc_qreg_assoc = BinaryAssociation(
        name="QuantumCircuit_QuantumRegister",
        ends={
            Property(
                name="qregs",
                type=quantum_register_class,
                owner=quantum_circuit_class,
                multiplicity=Multiplicity(0, "*"),
                is_composite=True
            ),
            Property(
                name="circuit",
                type=quantum_circuit_class,
                owner=quantum_register_class,
                multiplicity=Multiplicity(0, 1)
            )
        }
    )
    
    # QuantumCircuit has ClassicalRegisters
    qc_creg_assoc = BinaryAssociation(
        name="QuantumCircuit_ClassicalRegister",
        ends={
            Property(
                name="cregs",
                type=classical_register_class,
                owner=quantum_circuit_class,
                multiplicity=Multiplicity(0, "*"),
                is_composite=True
            ),
            Property(
                name="circuit",
                type=quantum_circuit_class,
                owner=classical_register_class,
                multiplicity=Multiplicity(0, 1)
            )
        }
    )
    
    # QuantumCircuit has QuantumOperations
    qc_op_assoc = BinaryAssociation(
        name="QuantumCircuit_QuantumOperation",
        ends={
            Property(
                name="operations",
                type=quantum_operation_class,
                owner=quantum_circuit_class,
                multiplicity=Multiplicity(0, "*"),
                is_composite=True
            ),
            Property(
                name="circuit",
                type=quantum_circuit_class,
                owner=quantum_operation_class,
                multiplicity=Multiplicity(0, 1)
            )
        }
    )
    
    # CustomGate references QuantumCircuit (definition)
    custom_gate_def_assoc = BinaryAssociation(
        name="CustomGate_Definition",
        ends={
            Property(
                name="definition",
                type=quantum_circuit_class,
                owner=custom_gate_class,
                multiplicity=Multiplicity(0, 1)
            ),
            Property(
                name="custom_gates",
                type=custom_gate_class,
                owner=quantum_circuit_class,
                multiplicity=Multiplicity(0, "*")
            )
        }
    )
    
    model.associations = {
        qc_qreg_assoc,
        qc_creg_assoc,
        qc_op_assoc,
        custom_gate_def_assoc
    }
    
    # ============================================
    # Generalizations (Inheritance)
    # ============================================
    
    # Gate inherits from QuantumOperation
    gate_inherits_op = Generalization(
        general=quantum_operation_class,
        specific=gate_class
    )
    
    # All gate types inherit from Gate
    primitive_inherits_gate = Generalization(general=gate_class, specific=primitive_gate_class)
    parametric_inherits_gate = Generalization(general=gate_class, specific=parametric_gate_class)
    arithmetic_inherits_gate = Generalization(general=gate_class, specific=arithmetic_gate_class)
    modular_arith_inherits_gate = Generalization(general=gate_class, specific=modular_arithmetic_gate_class)
    comparison_inherits_gate = Generalization(general=gate_class, specific=comparison_gate_class)
    qft_inherits_gate = Generalization(general=gate_class, specific=qft_gate_class)
    phase_grad_inherits_gate = Generalization(general=gate_class, specific=phase_gradient_gate_class)
    input_inherits_gate = Generalization(general=gate_class, specific=input_gate_class)
    order_inherits_gate = Generalization(general=gate_class, specific=order_gate_class)
    scalar_inherits_gate = Generalization(general=gate_class, specific=scalar_gate_class)
    custom_inherits_gate = Generalization(general=gate_class, specific=custom_gate_class)
    time_dep_inherits_gate = Generalization(general=gate_class, specific=time_dependent_gate_class)
    spacer_inherits_gate = Generalization(general=gate_class, specific=spacer_gate_class)
    
    # Specific primitive gates inherit from PrimitiveGate
    hadamard_inherits_primitive = Generalization(general=primitive_gate_class, specific=hadamard_gate_class)
    pauli_x_inherits_primitive = Generalization(general=primitive_gate_class, specific=pauli_x_gate_class)
    pauli_y_inherits_primitive = Generalization(general=primitive_gate_class, specific=pauli_y_gate_class)
    pauli_z_inherits_primitive = Generalization(general=primitive_gate_class, specific=pauli_z_gate_class)
    swap_inherits_primitive = Generalization(general=primitive_gate_class, specific=swap_gate_class)
    s_inherits_primitive = Generalization(general=primitive_gate_class, specific=s_gate_class)
    t_inherits_primitive = Generalization(general=primitive_gate_class, specific=t_gate_class)
    
    # Specific parametric gates inherit from ParametricGate
    rx_inherits_parametric = Generalization(general=parametric_gate_class, specific=rx_gate_class)
    ry_inherits_parametric = Generalization(general=parametric_gate_class, specific=ry_gate_class)
    rz_inherits_parametric = Generalization(general=parametric_gate_class, specific=rz_gate_class)
    phase_inherits_parametric = Generalization(general=parametric_gate_class, specific=phase_gate_class)
    
    # Other operations inherit from QuantumOperation
    display_inherits_op = Generalization(general=quantum_operation_class, specific=display_operation_class)
    measurement_inherits_op = Generalization(general=quantum_operation_class, specific=measurement_class)
    post_sel_inherits_op = Generalization(general=quantum_operation_class, specific=post_selection_class)
    
    model.generalizations = {
        gate_inherits_op,
        primitive_inherits_gate,
        parametric_inherits_gate,
        arithmetic_inherits_gate,
        modular_arith_inherits_gate,
        comparison_inherits_gate,
        qft_inherits_gate,
        phase_grad_inherits_gate,
        input_inherits_gate,
        order_inherits_gate,
        scalar_inherits_gate,
        custom_inherits_gate,
        time_dep_inherits_gate,
        spacer_inherits_gate,
        hadamard_inherits_primitive,
        pauli_x_inherits_primitive,
        pauli_y_inherits_primitive,
        pauli_z_inherits_primitive,
        swap_inherits_primitive,
        s_inherits_primitive,
        t_inherits_primitive,
        rx_inherits_parametric,
        ry_inherits_parametric,
        rz_inherits_parametric,
        phase_inherits_parametric,
        display_inherits_op,
        measurement_inherits_op,
        post_sel_inherits_op
    }
    
    return model


if __name__ == "__main__":
    print("=" * 70)
    print(" Creating Quantum Circuit Metamodel as BUML Structural Diagram")
    print("=" * 70)
    
    # Create the metamodel
    quantum_model = create_quantum_metamodel()
    
    print(f"\n✓ Model created: {quantum_model.name}")
    print(f"✓ Types: {len(quantum_model.types)}")
    print(f"✓ Associations: {len(quantum_model.associations)}")
    print(f"✓ Generalizations: {len(quantum_model.generalizations)}")
    
    # Display classes
    print("\n" + "-" * 70)
    print("CLASSES:")
    print("-" * 70)
    for cls in quantum_model.get_classes():
        abstract_marker = " (abstract)" if cls.is_abstract else ""
        print(f"  • {cls.name}{abstract_marker}")
        if cls.attributes:
            for attr in cls.attributes:
                print(f"      - {attr.name}: {attr.type.name}")
    
    # Display enumerations
    print("\n" + "-" * 70)
    print("ENUMERATIONS:")
    print("-" * 70)
    for enum in quantum_model.get_enumerations():
        print(f"  • {enum.name}")
        for literal in enum.literals:
            print(f"      - {literal.name}")
    
    # Display associations
    print("\n" + "-" * 70)
    print("ASSOCIATIONS:")
    print("-" * 70)
    for assoc in quantum_model.associations:
        print(f"  • {assoc.name}")
        for end in assoc.ends:
            mult_str = f"[{end.multiplicity.min}..{'*' if end.multiplicity.max == 9999 else end.multiplicity.max}]"
            print(f"      - {end.name}: {end.type.name} {mult_str}")
    
    # Display generalizations
    print("\n" + "-" * 70)
    print("GENERALIZATIONS:")
    print("-" * 70)
    for gen in quantum_model.generalizations:
        print(f"  • {gen.specific.name} → {gen.general.name}")
    
    # Validate the model
    print("\n" + "-" * 70)
    print("VALIDATION:")
    print("-" * 70)
    validation_result = quantum_model.validate(raise_exception=False)
    if validation_result["success"]:
        print("  ✓ Model is valid!")
    else:
        print("  ✗ Model has errors:")
        for error in validation_result["errors"]:
            print(f"      - {error}")
    
    print("\n" + "=" * 70)
    print(" Quantum Metamodel Successfully Created!")
    print("=" * 70 + "\n")
