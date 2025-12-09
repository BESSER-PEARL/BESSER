"""
BUML Structural Class Diagram for Quantum Metamodel V2
This file defines the Quantum Circuit metamodel using BUML's structural classes.
"""

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Multiplicity, Enumeration, EnumerationLiteral,
    DataType, PrimitiveDataType, BinaryAssociation, Generalization, Method, Parameter,NamedElement
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
    # Base Classes from Structural Metamodel
    # ============================================



    # Target class (abstract)
    target = Class(name="Target", is_abstract=True)

    # Hardware class
    hardware = Class(name="Hardware", is_abstract=False, attributes={
            Property(name="name", type="str")
        })

    # Simulator class
    simulator = Class(name="Simulator", is_abstract=False, attributes={
            Property(name="name", type="str")
        })

    # ============================================
    # Main Quantum Classes
    # ============================================

    # QuantumCircuit class
    quantum_circuit = Class(
        name="QuantumCircuit",
        is_abstract=False,

    )

    # QuantumRegister class
    quantum_register = Class(
        name="QuantumRegister",
        is_abstract=False,
        attributes={
            Property(name="size", type="int")
        }
    )

    # ClassicalRegister class
    classical_register = Class(
        name="ClassicalRegister",
        is_abstract=False,
        attributes={
            Property(name="size", type="int")
        }
    )

    # Qubit class
    qubit = Class(
        name="qubit",
        is_abstract=False,
        attributes={
            Property(name="id", type="int"),
            Property(name="index", type="int")
        }
    )

    # ClassicalBit class
    classical_bit = Class(
        name="classical_bit",
        is_abstract=False,
        attributes={
            Property(name="id", type="int"),
            Property(name="index", type="int")
        }
    )

    # QuantumOracle class
    quantum_oracle = Class(
        name="QuantumOracle",
        is_abstract=False,
        attributes={
            Property(name="specification", type="str")
        }
    )



    # ============================================
    # QuantumOperation hierarchy
    # ============================================

    # QuantumOperation class (abstract)
    quantum_operation = Class(
        name="QuantumOperation",
        is_abstract=True
    )

    # Measurement class
    measurement = Class(
        name="Measurement",
        is_abstract=False,
        attributes={
            Property(name="output_bit", type="int"),
            Property(name="basis", type="str")
        }
    )

    # ============================================
    # Gate Hierarchy
    # ============================================

    # Gate class (abstract)
    gate = Class(
        name="Gate",
        is_abstract=True
    )

    # GateDefinition class
    gate_definition = Class(
        name="GateDefinition",
        is_abstract=False
    )

    # CustomGate class
    custom_gate = Class(
        name="CustomGate",
        is_abstract=False,
        attributes={
            Property(name="definition", type="str")        }
    )

    # InputGate class
    input_gate = Class(
        name="InputGate",
        is_abstract=False,
        attributes={
            Property(name="input_type", type="str"),
            Property(name="value", type="str")
        }
    )

    # ============================================
    # Abstract Gate Categories
    # ============================================

    # SingleBitGate (abstract)
    single_bit_gate = Class(
        name="SingleBitGate",
        is_abstract=True
    )

    # MultiBitGate (abstract)
    multi_bit_gate = Class(
        name="MultiBitGate",
        is_abstract=True
    )

    # ArithmeticGate (abstract)
    arithmetic_gate = Class(
        name="ArithmeticGate",
        is_abstract=True,
        attributes={
            Property(name="operation_type", type="str")
        }
    )

    # ComparisonGate class
    comparison_gate = Class(
        name="ComparisonGate",
        is_abstract=False,
        attributes={
            Property(name="operation", type="str")
        }
    )

    # ============================================
    # Single Bit Gates
    # ============================================

    # Pauli Gates
    pauli_z_gate = Class(name="PauliZGate", is_abstract=False)
    pauli_y_gate = Class(name="PauliYGate", is_abstract=False)
    pauli_x_gate = Class(name="PauliXGate", is_abstract=False)

    # Hadamard Gate
    hadamard_gate = Class(name="HadamardGate", is_abstract=False)

    # Phase Gates
    # Phase Gates
    t_gate = Class(name="TGate", is_abstract=False)
    s_gate = Class(name="SGate", is_abstract=False)
    phase_gate_z = Class(name="PhaseGateZ", is_abstract=False)

    # Rotation Gates
    rx_gate = Class(
        name="RXGate",
        is_abstract=False,
        attributes={
            Property(name="theta", type="float")
        }
    )
    ry_gate = Class(
        name="RYGate",
        is_abstract=False,
        attributes={
            Property(name="theta", type="float")
        }
    )
    rz_gate = Class(
        name="RZGate",
        is_abstract=False,
        attributes={
            Property(name="theta", type="float")
        }
    )

    # Identity Gate
    identity_gate = Class(name="IdentityGate", is_abstract=False)

    # ============================================
    # Multi Bit Gates
    # ============================================

    # CNOT Gate
    cnot_gate = Class(name="CNOT", is_abstract=False)

    # C-Phase Gate
    c_phase_gate = Class(
        name="CPhaseGate",
        is_abstract=False,
        attributes={
            Property(name="theta", type="float")
        }
    )

    # CY Gate
    cy_gate = Class(name="CYGate", is_abstract=False)

    # CZ Gate
    cz_gate = Class(name="CZGate", is_abstract=False)

    # CX Gate
    cx_gate = Class(name="CXGate", is_abstract=False)

    # CH Gate
    ch_gate = Class(name="CHGate", is_abstract=False)

    # QFT Gate
    qft_gate = Class(
        name="QFTGate",
        is_abstract=False,
        attributes={
            Property(name="inverse", type="bool")
        }
    )

    # RXX Gate
    rxx_gate = Class(
        name="RXXGate",
        is_abstract=False,
        attributes={
            Property(name="theta", type="float")
        }
    )

    # RZZ Gate
    rzz_gate = Class(name="RZZGate", is_abstract=False)

    # CRX Gate
    crx_gate = Class(
        name="CRXGate",
        is_abstract=False,
        attributes={
            Property(name="theta", type="float")
        }
    )

    # CRZ Gate
    crz_gate = Class(
        name="CRZGate",
        is_abstract=False,
        attributes={
            Property(name="theta", type="float")
        }
    )

    # CRY Gate
    cry_gate = Class(
        name="CRYGate",
        is_abstract=False,
        attributes={
            Property(name="theta", type="float")
        }
    )

    # Bell Gate
    bell_gate = Class(name="BellGate", is_abstract=False)

    # iSwap Gate
    iswap_gate = Class(name="iSwapGate", is_abstract=False)

    # Swap Gate
    swap_gate = Class(name="SwapGate", is_abstract=False)

    # Sqrt Swap Gate
    sqrt_swap_gate = Class(name="SqrtSwapGate", is_abstract=False)

    # RYY Gate
    ryy_gate = Class(
        name="RYYGate",
        is_abstract=False,
        attributes={
            Property(name="theta", type="float")
        }
    )

    # RZX Gate
    rzx_gate = Class(
        name="RZXGate",
        is_abstract=False,
        attributes={
            Property(name="theta", type="float")
        }
    )

    # ============================================
    # Missing Gate Classes (Added for Builder Compatibility)
    # ============================================

    phase_gate = Class(
        name="PhaseGate",
        is_abstract=False,
        attributes={
            Property(name="angle", type="float")
        }
    )

    phase_gradient_gate = Class(
        name="PhaseGradientGate",
        is_abstract=False,
        attributes={
            Property(name="inverse", type="bool")
        }
    )

    modular_arithmetic_gate = Class(
        name="ModularArithmeticGate",
        is_abstract=False,
        attributes={
            Property(name="modulo", type="int")
        }
    )

    display_operation = Class(
        name="DisplayOperation",
        is_abstract=False,
        attributes={
            Property(name="display_type", type="str")
        }
    )

    time_dependent_gate = Class(
        name="TimeDependentGate",
        is_abstract=False,
        attributes={
            Property(name="type_name", type="str"),
            Property(name="parameter_expr", type="str")
        }
    )

    spacer_gate = Class(name="SpacerGate", is_abstract=False)

    post_selection = Class(
        name="PostSelection",
        is_abstract=False,
        attributes={
            Property(name="value", type="int"),
            Property(name="basis", type="str")
        }
    )

    primitive_gate = Class(
        name="PrimitiveGate",
        is_abstract=False,
        attributes={
            Property(name="type_name", type="str")
        }
    )

    parametric_gate = Class(
        name="ParametricGate",
        is_abstract=False,
        attributes={
            Property(name="type_name", type="str"),
            Property(name="parameter", type="float")
        }
    )

    order_gate = Class(
        name="OrderGate",
        is_abstract=False,
        attributes={
            Property(name="order_type", type="str")
        }
    )

    scalar_gate = Class(
        name="ScalarGate",
        is_abstract=False,
        attributes={
            Property(name="scalar_type", type="str")
        }
    )

    # ============================================
    # Add all types to the model
    # ============================================

    model.types = {
        # Base structural classes
        model,
        target,
        hardware,
        simulator,

        # Main quantum classes
        quantum_circuit,
        quantum_register,
        classical_register,
        qubit,
        classical_bit,
        quantum_oracle,

        # Operation hierarchy
        quantum_operation,
        measurement,

        # Gate hierarchy
        gate,
        gate_definition,
        custom_gate,
        input_gate,

        # Abstract gate categories
        single_bit_gate,
        multi_bit_gate,
        arithmetic_gate,
        comparison_gate,

        # Single bit gates
        pauli_z_gate,
        pauli_y_gate,
        pauli_x_gate,
        hadamard_gate,
        pauli_x_gate,
        hadamard_gate,
        t_gate,
        s_gate,
        phase_gate_z,
        rx_gate,
        ry_gate,
        rz_gate,
        identity_gate,

        # Multi bit gates
        cnot_gate,
        c_phase_gate,
        cy_gate,
        cz_gate,
        cx_gate,
        ch_gate,
        qft_gate,
        rxx_gate,
        rzz_gate,
        crx_gate,
        crz_gate,
        cry_gate,
        bell_gate,
        iswap_gate,
        swap_gate,
        sqrt_swap_gate,
        ryy_gate,
        ryy_gate,
        rzx_gate,
        
        # New gates
        phase_gate,
        phase_gradient_gate,
        modular_arithmetic_gate,
        display_operation,
        time_dependent_gate,
        spacer_gate,
        post_selection,
        primitive_gate,
        parametric_gate,
        order_gate,
        scalar_gate
    }

    # ============================================
    # Associations
    # ============================================

    # QuantumCircuit has QuantumRegisters (qregs)
    qc_qreg_assoc = BinaryAssociation(
        name="QuantumCircuit_QuantumRegister",
        ends={
            Property(
                name="qregs",
                type=quantum_register,
                owner=quantum_circuit,
                multiplicity=Multiplicity(0, "*"),
                is_composite=True
            ),
            Property(
                name="circuit",
                type=quantum_circuit,
                owner=quantum_register,
                multiplicity=Multiplicity(0, 1)
            )
        }
    )

    # QuantumCircuit has ClassicalRegisters (cregs)
    qc_creg_assoc = BinaryAssociation(
        name="QuantumCircuiticalRegister",
        ends={
            Property(
                name="cregs",
                type=classical_register,
                owner=quantum_circuit,
                multiplicity=Multiplicity(0, "*"),
                is_composite=True
            ),
            Property(
                name="circuit",
                type=quantum_circuit,
                owner=classical_register,
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
                type=quantum_operation,
                owner=quantum_circuit,
                multiplicity=Multiplicity(0, "*"),
                is_composite=True
            ),
            Property(
                name="circuit",
                type=quantum_circuit,
                owner=quantum_operation,
                multiplicity=Multiplicity(0, 1)
            )
        }
    )

    # CustomGate has GateDefinition
    custom_gate_def_assoc = BinaryAssociation(
        name="CustomGate_GateDefinition",
        ends={
            Property(
                name="definition",
                type=gate_definition,
                owner=custom_gate,
                multiplicity=Multiplicity(1, 1)
            ),
            Property(
                name="custom_gate",
                type=custom_gate,
                owner=gate_definition,
                multiplicity=Multiplicity(0, 1)
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

    generalizations = []




    # Target inheritance
    generalizations.append(Generalization(general=target, specific=hardware))
    generalizations.append(Generalization(general=target, specific=simulator))

    # Gate and Measurement inherit from QuantumOperation
    generalizations.append(Generalization(general=quantum_operation, specific=gate))
    generalizations.append(Generalization(general=quantum_operation, specific=measurement))

    # CustomGate and InputGate inherit from Gate
    generalizations.append(Generalization(general=gate, specific=custom_gate))
    generalizations.append(Generalization(general=gate, specific=input_gate))

    # SingleBitGate and MultiBitGate inherit from Gate
    generalizations.append(Generalization(general=gate, specific=single_bit_gate))
    generalizations.append(Generalization(general=gate, specific=multi_bit_gate))

    # ArithmeticGate and ComparisonGate inherit from Gate
    generalizations.append(Generalization(general=gate, specific=arithmetic_gate))
    generalizations.append(Generalization(general=gate, specific=comparison_gate))

    # Single bit gates inherit from SingleBitGate
    single_bit_gates = [
        pauli_z_gate, pauli_y_gate, pauli_x_gate,
        hadamard_gate, t_gate, s_gate,
        phase_gate_z, rx_gate, ry_gate, rz_gate,
        identity_gate
    ]
    for gate in single_bit_gates:
        generalizations.append(Generalization(general=single_bit_gate, specific=gate))

    # Multi bit gates inherit from MultiBitGate
    multi_bit_gates = [
        cnot_gate, c_phase_gate, cy_gate, cz_gate,
        cx_gate, ch_gate, qft_gate, rxx_gate,
        rzz_gate, crx_gate, crz_gate, cry_gate,
        bell_gate, iswap_gate, swap_gate, sqrt_swap_gate,
        ryy_gate, rzx_gate
    ]
    for gate in multi_bit_gates:
        generalizations.append(Generalization(general=multi_bit_gate, specific=gate))

    # Generalizations for new gates
    generalizations.append(Generalization(general=single_bit_gate, specific=phase_gate))
    generalizations.append(Generalization(general=multi_bit_gate, specific=phase_gradient_gate))
    generalizations.append(Generalization(general=arithmetic_gate, specific=modular_arithmetic_gate))
    generalizations.append(Generalization(general=quantum_operation, specific=display_operation))
    generalizations.append(Generalization(general=gate, specific=time_dependent_gate))
    generalizations.append(Generalization(general=gate, specific=spacer_gate))
    generalizations.append(Generalization(general=quantum_operation, specific=post_selection))
    generalizations.append(Generalization(general=gate, specific=primitive_gate))
    generalizations.append(Generalization(general=gate, specific=parametric_gate))
    generalizations.append(Generalization(general=gate, specific=order_gate))
    generalizations.append(Generalization(general=quantum_operation, specific=scalar_gate))

    model.generalizations = set(generalizations)

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
