"""
Quantum diagram processing for converting JSON to BUML format.
"""

from besser.BUML.metamodel.quantum.quantum import (
    QuantumCircuit,
    HadamardGate, PauliXGate, PauliYGate, PauliZGate,
    SGate, TGate, SwapGate, RXGate, RYGate, RZGate,
    ControlState, Measurement, InputGate, QFTGate, PhaseGradientGate,
    ArithmeticGate, ModularArithmeticGate, ComparisonGate, CustomGate,
    DisplayOperation, TimeDependentGate, SpacerGate, PostSelection,
    ParametricGate, OrderGate, ScalarGate
)
import math


def process_quantum_diagram(json_data):
    model_data = json_data.get('model', {})
    if not model_data:
        model_data = json_data

    cols = model_data.get('cols', [])

    raw_title = json_data.get('title', 'Quantum_Circuit')
    title = raw_title.replace(' ', '_').replace('-', '_')
    title = ''.join(c for c in title if c.isalnum() or c == '_')
    if not title:
        title = 'Quantum_Circuit'

    num_qubits = 0
    for col in cols:
        if len(col) > num_qubits:
            num_qubits = len(col)

    qc = QuantumCircuit(name=title, qubits=num_qubits)

    for col_idx, col in enumerate(cols):
        col_gates = {}
        controls = []
        swap_qubits = []

        for q_idx, symbol in enumerate(col):
            if symbol == 1 or symbol == "1":
                continue
            if symbol == "\u2022":  # • (bullet/control)
                controls.append((q_idx, ControlState.CONTROL))
            elif symbol == "\u25e6":  # ◦ (anti-control)
                controls.append((q_idx, ControlState.ANTI_CONTROL))
            elif symbol == "Swap":
                swap_qubits.append(q_idx)
            else:
                col_gates[q_idx] = symbol

        if len(swap_qubits) >= 2:
            gate = SwapGate(swap_qubits[0], swap_qubits[1],
                          control_qubits=[c[0] for c in controls],
                          control_states=[c[1] for c in controls])
            qc.add_operation(gate)

        for q_idx, symbol in col_gates.items():
            gate = _create_gate(symbol, q_idx, controls)
            if gate:
                qc.add_operation(gate)

    return qc


def _create_gate(symbol, target_qubit, controls):
    control_qubits = [c[0] for c in controls] if controls else []
    control_states = [c[1] for c in controls] if controls else []

    # === Half Turns ===
    if symbol == 'H':
        return HadamardGate(target_qubit, control_qubits=control_qubits, control_states=control_states)
    if symbol == 'X':
        return PauliXGate(target_qubit, control_qubits=control_qubits, control_states=control_states)
    if symbol == 'Y':
        return PauliYGate(target_qubit, control_qubits=control_qubits, control_states=control_states)
    if symbol == 'Z':
        return PauliZGate(target_qubit, control_qubits=control_qubits, control_states=control_states)

    # === Quarter Turns ===
    if symbol in ('S', 'Z^½'):
        return SGate(target_qubit, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('S_DAG', 'Z^-½'):
        return ParametricGate('S_DAG', [target_qubit], -math.pi/2, control_qubits, control_states)
    if symbol in ('V', 'X^½'):
        return ParametricGate('V', [target_qubit], math.pi/2, control_qubits, control_states)
    if symbol in ('V_DAG', 'X^-½'):
        return ParametricGate('V_DAG', [target_qubit], -math.pi/2, control_qubits, control_states)
    if symbol in ('SQRT_Y', 'Y^½'):
        return ParametricGate('SQRT_Y', [target_qubit], math.pi/2, control_qubits, control_states)
    if symbol in ('SQRT_Y_DAG', 'Y^-½'):
        return ParametricGate('SQRT_Y_DAG', [target_qubit], -math.pi/2, control_qubits, control_states)

    # === Eighth Turns ===
    if symbol in ('T', 'Z^¼'):
        return TGate(target_qubit, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('T_DAG', 'Z^-¼'):
        return ParametricGate('T_DAG', [target_qubit], -math.pi/4, control_qubits, control_states)
    if symbol in ('SQRT_SQRT_X', 'X^¼'):
        return ParametricGate('X^1/4', [target_qubit], math.pi/4, control_qubits, control_states)
    if symbol in ('SQRT_SQRT_X_DAG', 'X^-¼'):
        return ParametricGate('X^-1/4', [target_qubit], -math.pi/4, control_qubits, control_states)
    if symbol in ('SQRT_SQRT_Y', 'Y^¼'):
        return ParametricGate('Y^1/4', [target_qubit], math.pi/4, control_qubits, control_states)
    if symbol in ('SQRT_SQRT_Y_DAG', 'Y^-¼'):
        return ParametricGate('Y^-1/4', [target_qubit], -math.pi/4, control_qubits, control_states)

    # === Parametric Rotations ===
    if symbol.startswith('Rx') or symbol == 'X_POW':
        return RXGate(target_qubit, 0.0, control_qubits=control_qubits, control_states=control_states)
    if symbol.startswith('Ry') or symbol == 'Y_POW':
        return RYGate(target_qubit, 0.0, control_qubits=control_qubits, control_states=control_states)
    if symbol.startswith('Rz') or symbol == 'Z_POW':
        return RZGate(target_qubit, 0.0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('EXP_X', 'exp(iXt)'):
        return TimeDependentGate('EXP_X', [target_qubit], 'exp(iXt)', control_qubits, control_states)
    if symbol in ('EXP_Y', 'exp(iYt)'):
        return TimeDependentGate('EXP_Y', [target_qubit], 'exp(iYt)', control_qubits, control_states)
    if symbol in ('EXP_Z', 'exp(iZt)'):
        return TimeDependentGate('EXP_Z', [target_qubit], 'exp(iZt)', control_qubits, control_states)

    # === Time-Dependent (Spinning) Gates ===
    if symbol in ('Z_POW_T', 'Z^t'):
        return TimeDependentGate('Z^t', [target_qubit], 't', control_qubits, control_states)
    if symbol in ('Z_POW_NEG_T', 'Z^-t'):
        return TimeDependentGate('Z^-t', [target_qubit], '-t', control_qubits, control_states)
    if symbol in ('Y_POW_T', 'Y^t'):
        return TimeDependentGate('Y^t', [target_qubit], 't', control_qubits, control_states)
    if symbol in ('Y_POW_NEG_T', 'Y^-t'):
        return TimeDependentGate('Y^-t', [target_qubit], '-t', control_qubits, control_states)
    if symbol in ('X_POW_T', 'X^t'):
        return TimeDependentGate('X^t', [target_qubit], 't', control_qubits, control_states)
    if symbol in ('X_POW_NEG_T', 'X^-t'):
        return TimeDependentGate('X^-t', [target_qubit], '-t', control_qubits, control_states)

    # === Formulaic Gates ===
    if symbol in ('Z_FUNC_T', 'RZ_FUNC_T'):
        return TimeDependentGate('Z_FUNC', [target_qubit], 'f(t)', control_qubits, control_states)
    if symbol in ('Y_FUNC_T', 'RY_FUNC_T'):
        return TimeDependentGate('Y_FUNC', [target_qubit], 'f(t)', control_qubits, control_states)
    if symbol in ('X_FUNC_T', 'RX_FUNC_T'):
        return TimeDependentGate('X_FUNC', [target_qubit], 'f(t)', control_qubits, control_states)
    if symbol in ('TIME_SHIFT', 'TimeShift'):
        return OrderGate('TimeShift', [target_qubit])
    if symbol in ('TIME_SHIFT_INV', 'TimeShiftInv'):
        return OrderGate('TimeShiftInv', [target_qubit])

    # === Display Operations ===
    if symbol in ('Bloch', 'BLOCH'):
        return DisplayOperation('Bloch', [target_qubit])
    if symbol in ('Density', 'DENSITY'):
        return DisplayOperation('Density', [target_qubit])
    if symbol in ('Amps', 'Amplitude', 'AMPLITUDE', 'AMP'):
        return DisplayOperation('Amplitude', [target_qubit])
    if symbol in ('Chance', 'CHANCE'):
        return DisplayOperation('Chance', [target_qubit])

    # === Sampling Gates ===
    if symbol in ('SAMPLE', 'Sample'):
        return DisplayOperation('Sample', [target_qubit])
    if symbol in ('DETECT', 'Detect'):
        return DisplayOperation('Detect', [target_qubit])
    if symbol in ('AXIS_SAMPLE', 'AxisSample'):
        return DisplayOperation('AxisSample', [target_qubit])

    # === Measurement ===
    if symbol in ('Measure', 'MEASURE'):
        return Measurement(target_qubit, target_qubit, basis='Z')
    if symbol in ('MEASURE_X', 'MeasureX'):
        return Measurement(target_qubit, target_qubit, basis='X')
    if symbol in ('MEASURE_Y', 'MeasureY'):
        return Measurement(target_qubit, target_qubit, basis='Y')

    # === Post-Selection ===
    if symbol in ('POST_SELECT_OFF', '|0⟩'):
        return PostSelection(target_qubit, 0, 'Z')
    if symbol in ('POST_SELECT_ON', '|1⟩'):
        return PostSelection(target_qubit, 1, 'Z')
    if symbol in ('POST_SELECT_X_OFF', '|+⟩'):
        return PostSelection(target_qubit, 0, 'X')
    if symbol in ('POST_SELECT_X_ON', '|−⟩'):
        return PostSelection(target_qubit, 1, 'X')
    if symbol in ('POST_SELECT_Y_OFF', '|i⟩'):
        return PostSelection(target_qubit, 0, 'Y')
    if symbol in ('POST_SELECT_Y_ON', '|-i⟩'):
        return PostSelection(target_qubit, 1, 'Y')

    # === Input Gates ===
    if symbol.startswith('input') or symbol.startswith('Input') or symbol.startswith('INPUT'):
        return InputGate(symbol, [target_qubit])
    if symbol in ('RANDOM', 'Random'):
        return InputGate('Random', [target_qubit])

    # === Frequency Gates ===
    if symbol in ('QFT', 'qft'):
        return QFTGate([target_qubit], inverse=False, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('QFT_DAG', 'qft†', 'QFT†'):
        return QFTGate([target_qubit], inverse=True, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('PhaseGradient', 'PHASE_GRADIENT'):
        return PhaseGradientGate([target_qubit], inverse=False, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('PHASE_GRADIENT_DAG', 'PhaseGradient†'):
        return PhaseGradientGate([target_qubit], inverse=True, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('PHASE_GRADIENT_INV', 'PhaseGradientInv'):
        return PhaseGradientGate([target_qubit], inverse=True, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('PHASE_GRADIENT_INV_DAG', 'PhaseGradientInv†'):
        return PhaseGradientGate([target_qubit], inverse=False, control_qubits=control_qubits, control_states=control_states)

    # === Arithmetic Gates ===
    if symbol in ('+1', 'INC', 'inc'):
        return ArithmeticGate('Increment', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('-1', 'DEC', 'dec'):
        return ArithmeticGate('Decrement', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('+A', 'ADD', 'add'):
        return ArithmeticGate('Add', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('-A', 'SUB', 'sub'):
        return ArithmeticGate('Subtract', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('×A', 'MUL', 'mul'):
        return ArithmeticGate('Multiply', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('+AB', 'ADD_AB'):
        return ArithmeticGate('AddAB', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('-AB', 'SUB_AB'):
        return ArithmeticGate('SubtractAB', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('×A⁻¹', 'MUL_INV'):
        return ArithmeticGate('MultiplyInverse', [target_qubit], control_qubits=control_qubits, control_states=control_states)

    # === Modular Arithmetic Gates ===
    if symbol in ('MOD_INC', 'ModInc'):
        return ModularArithmeticGate('Increment', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_DEC', 'ModDec'):
        return ModularArithmeticGate('Decrement', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_ADD', 'ModAdd'):
        return ModularArithmeticGate('Add', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_SUB', 'ModSub'):
        return ModularArithmeticGate('Subtract', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_MUL', 'ModMul'):
        return ModularArithmeticGate('Multiply', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_INV_MUL', 'ModInvMul'):
        return ModularArithmeticGate('MultiplyInverse', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_MUL_B', 'ModMulB'):
        return ModularArithmeticGate('MultiplyB', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)
    if symbol in ('MOD_MUL_B_INV', 'ModMulBInv'):
        return ModularArithmeticGate('MultiplyBInverse', [target_qubit], modulo=0, control_qubits=control_qubits, control_states=control_states)

    # === Comparison Gates ===
    if symbol in ('COMPARE', 'compare', '<'):
        return ComparisonGate('LessThan', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('GREATER_THAN', '>'):
        return ComparisonGate('GreaterThan', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('LESS_EQUAL', '<=', '≤'):
        return ComparisonGate('LessEqual', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('GREATER_EQUAL', '>=', '≥'):
        return ComparisonGate('GreaterEqual', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('EQUAL', '==', '='):
        return ComparisonGate('Equal', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('NOT_EQUAL', '!=', '≠'):
        return ComparisonGate('NotEqual', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('COMPARE_A_LT', 'A<'):
        return ComparisonGate('ALessThan', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('COMPARE_A_GT', 'A>'):
        return ComparisonGate('AGreaterThan', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('COMPARE_A_EQ', 'A='):
        return ComparisonGate('AEqual', [target_qubit], control_qubits=control_qubits, control_states=control_states)

    # === Logic Gates ===
    if symbol in ('COUNT_1S', 'Count'):
        return ArithmeticGate('CountOnes', [target_qubit], control_qubits=control_qubits, control_states=control_states)
    if symbol in ('CYCLE_BITS', 'Cycle'):
        return OrderGate('CycleBits', [target_qubit])
    if symbol in ('XOR', 'Xor'):
        return ArithmeticGate('XOR', [target_qubit], control_qubits=control_qubits, control_states=control_states)

    # === Order Gates ===
    if symbol in ('INTERLEAVE', 'Interleave'):
        return OrderGate('Interleave', [target_qubit])
    if symbol in ('DEINTERLEAVE', 'Deinterleave'):
        return OrderGate('Deinterleave', [target_qubit])
    if symbol in ('REVERSE_BITS', 'Reverse'):
        return OrderGate('Reverse', [target_qubit])
    if symbol.startswith('<<') or symbol in ('ROTATE_BITS_LEFT', 'RotateLeft'):
        return OrderGate('RotateLeft', [target_qubit])
    if symbol.startswith('>>') or symbol in ('ROTATE_BITS_RIGHT', 'RotateRight'):
        return OrderGate('RotateRight', [target_qubit])

    # === Scalar (Global Phase) Gates ===
    if symbol in ('ONE', '1'):
        return ScalarGate('1')
    if symbol == 'MINUS_ONE':
        return ScalarGate('-1')
    if symbol in ('PHASE_I', 'i'):
        return ScalarGate('i')
    if symbol in ('PHASE_MINUS_I', '-i'):
        return ScalarGate('-i')
    if symbol in ('PHASE_SQRT_I', '√i'):
        return ScalarGate('√i')
    if symbol in ('PHASE_SQRT_MINUS_I', '√-i'):
        return ScalarGate('√-i')

    # === Spacer ===
    if symbol in ('SPACER', '…', 'spacer'):
        return SpacerGate([target_qubit])

    # === Custom/Unknown - fallback ===
    return CustomGate(symbol, [target_qubit], control_qubits=control_qubits, control_states=control_states)
