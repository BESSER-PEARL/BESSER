# Quantum Gate Support Matrix

This document provides a comprehensive overview of all gates available in the BESSER Quantum Editor and their support status when generating Qiskit code.

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | **Native Support** - Maps directly to Qiskit gates |
| 🔧 | **Custom Implementation** - Implemented via helper class |
| ⚠️ | **Placeholder** - Generates runnable code but no actual operation |
| 📝 | **Comment/Special** - Generates a comment or handled specially |

---

## Half Turns (Basic Gates)

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Hadamard | H | ✅ | `HGate()` |
| Pauli-X | X | ✅ | `XGate()` |
| Pauli-Y | Y | ✅ | `YGate()` |
| Pauli-Z | Z | ✅ | `ZGate()` |
| Swap | SWAP | ✅ | `SwapGate()` |

---

## Quarter Turns

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| S Gate | S | ✅ | `SGate()` |
| S Dagger | S† | ✅ | `SdgGate()` |
| V Gate (√X) | V | ✅ | `SXGate()` |
| V Dagger | V† | ✅ | `SXdgGate()` |
| √Y | √Y | ✅ | `RYGate(π/2)` |
| √Y Dagger | √Y† | ✅ | `RYGate(-π/2)` |

---

## Eighth Turns

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| T Gate | T | ✅ | `TGate()` |
| T Dagger | T† | ✅ | `TdgGate()` |
| X^(1/4) | ⁴√X | ✅ | `PhaseGate(π/4)` |
| X^(-1/4) | ⁴√X† | ✅ | `PhaseGate(-π/4)` |
| Y^(1/4) | ⁴√Y | ✅ | `RYGate(π/4)` |
| Y^(-1/4) | ⁴√Y† | ✅ | `RYGate(-π/4)` |

---

## Parametrized Rotations

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| X Power | X^t | ✅ | `RXGate(θ)` |
| Y Power | Y^t | ✅ | `RYGate(θ)` |
| Z Power | Z^t | ✅ | `RZGate(θ)` / `PhaseGate(θ)` |
| Exp X | e^(iXt) | ✅ | `RXGate(2t)` |
| Exp Y | e^(iYt) | ✅ | `RYGate(2t)` |
| Exp Z | e^(iZt) | ✅ | `RZGate(2t)` |

---

## Frequency Gates

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| QFT | QFT | ✅ | `QFT(n).to_instruction()` |
| QFT Dagger | QFT† | ✅ | `QFT(n, inverse=True).to_instruction()` |
| Phase Gradient | ∇φ | 🔧 | `PhaseGradient(n)` helper class |
| Phase Gradient Dagger | ∇φ† | 🔧 | `PhaseGradient(n, inverse=True)` |
| Phase Gradient Inverse | ∇φ⁻¹ | 🔧 | `PhaseGradient(n, inverse=True)` |
| Phase Gradient Inv Dagger | ∇φ⁻¹† | 🔧 | `PhaseGradient(n)` |

---

## Measurement & Probes

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Measure (Z-basis) | M | ✅ | `qc.measure(q, c)` |
| Measure X | Mx | ✅ | `qc.h(); qc.measure()` |
| Measure Y | My | ✅ | `qc.sdg(); qc.h(); qc.measure()` |
| Control | ● | ✅ | `.control()` modifier |
| Anti-Control | ○ | ✅ | `.control(ctrl_state='0')` |
| Control X | ●x | ✅ | `.control()` with X-basis |
| Control Y | ●y | ✅ | `.control()` with Y-basis |
| Post-Select Off | ⊥0 | 📝 | Comment only (not simulatable) |
| Post-Select On | ⊥1 | 📝 | Comment only |
| Post-Select X Off | ⊥x0 | 📝 | Comment only |
| Post-Select X On | ⊥x1 | 📝 | Comment only |
| Post-Select Y Off | ⊥y0 | 📝 | Comment only |
| Post-Select Y On | ⊥y1 | 📝 | Comment only |

---

## Order Gates

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Reverse Bits | ⟲ | 🔧 | `ReverseBits(n)` helper class |
| Interleave | ⫘ | 🔧 | `Interleave(n)` helper class |
| Deinterleave | ⫗ | 🔧 | `Deinterleave(n)` helper class |
| Rotate Left | << | 🔧 | `RotateBitsLeft(n)` helper class |
| Rotate Right | >> | 🔧 | `RotateBitsRight(n)` helper class |
| Cycle Bits | ↻ | ⚠️ | Placeholder |
| Time Shift | τ | ⚠️ | Placeholder |
| Time Shift Inverse | τ⁻¹ | ⚠️ | Placeholder |

---

## Arithmetic Gates

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Increment | +1 | 🔧 | `Increment(n)` helper class |
| Decrement | -1 | 🔧 | `Decrement(n)` helper class |
| Add | +A | ✅ | `DraperQFTAdder(n)` |
| Subtract | -A | ✅ | `DraperQFTAdder(n).inverse()` |
| Multiply | ×A | ✅ | `HRSCumulativeMultiplier(n)` |
| Add AB | +AB | ⚠️ | Placeholder |
| Subtract AB | -AB | ⚠️ | Placeholder |
| Multiply Inverse | ×A⁻¹ | ⚠️ | Placeholder |
| Count Ones | Σ1 | ⚠️ | Placeholder |
| XOR | ⊕ | ⚠️ | Placeholder |

---

## Modular Arithmetic Gates

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Mod Increment | +1 mod | ⚠️ | Placeholder |
| Mod Decrement | -1 mod | ⚠️ | Placeholder |
| Mod Add | +A mod | ⚠️ | Placeholder |
| Mod Subtract | -A mod | ⚠️ | Placeholder |
| Mod Multiply | ×A mod | ⚠️ | Placeholder |
| Mod Inverse Multiply | ×A⁻¹ mod | ⚠️ | Placeholder |
| Mod Multiply B | ×B mod | ⚠️ | Placeholder |
| Mod Multiply B Inverse | ×B⁻¹ mod | ⚠️ | Placeholder |

---

## Comparison Gates

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Less Than | < | ⚠️ | Placeholder |
| Greater Than | > | ⚠️ | Placeholder |
| Less Equal | ≤ | ⚠️ | Placeholder |
| Greater Equal | ≥ | ⚠️ | Placeholder |
| Equal | = | ⚠️ | Placeholder |
| Not Equal | ≠ | ⚠️ | Placeholder |
| A Less Than | A< | ⚠️ | Placeholder |
| A Greater Than | A> | ⚠️ | Placeholder |
| A Equal | A= | ⚠️ | Placeholder |

---

## Scalar Gates (Global Phase)

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Phase i | i | ✅ | `GlobalPhaseGate(π/2)` |
| Phase -i | -i | ✅ | `GlobalPhaseGate(-π/2)` |
| Phase √i | √i | ✅ | `GlobalPhaseGate(π/4)` |
| Phase -√i | -√i | ✅ | `GlobalPhaseGate(-π/4)` |
| One | 1 | ✅ | `GlobalPhaseGate(0)` |
| Minus One | -1 | ✅ | `GlobalPhaseGate(π)` |

---

## Time-Dependent (Spinning) Gates

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Z^t | Z^t | ✅ | `RZGate(π*t)` |
| Z^(-t) | Z^-t | ✅ | `RZGate(-π*t)` |
| Y^t | Y^t | ✅ | `RYGate(π*t)` |
| Y^(-t) | Y^-t | ✅ | `RYGate(-π*t)` |
| X^t | X^t | ✅ | `RXGate(π*t)` |
| X^(-t) | X^-t | ✅ | `RXGate(-π*t)` |

---

## Formulaic Gates

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Z(f(t)) | Z(f) | ⚠️ | Placeholder (requires expression parsing) |
| Rz(f(t)) | Rz(f) | ⚠️ | Placeholder |
| Y(f(t)) | Y(f) | ⚠️ | Placeholder |
| Ry(f(t)) | Ry(f) | ⚠️ | Placeholder |
| X(f(t)) | X(f) | ⚠️ | Placeholder |
| Rx(f(t)) | Rx(f) | ⚠️ | Placeholder |

---

## Input Gates

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Input A | A | 📝 | X gates based on value bits |
| Input B | B | 📝 | X gates based on value bits |
| Input R | R | 📝 | X gates based on value bits |
| Random | ? | ⚠️ | Placeholder |

---

## Display Gates

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Bloch Sphere | 🔮 | 📝 | `qc.save_statevector()` |
| Density Matrix | ρ | 📝 | `qc.save_density_matrix()` |
| Amplitude | Amp | 📝 | `qc.save_statevector()` |
| Chance | % | 📝 | `qc.save_probabilities()` |

---

## Special Gates

| Editor Gate | Symbol | Qiskit Status | Qiskit Mapping |
|-------------|--------|---------------|----------------|
| Spacer | … | ✅ | `IGate()` (identity) |
| Function Gate | f() | 🔧 | Custom function definition |
| Mystery | ? | ⚠️ | Placeholder |
| Zero | 0 | ⚠️ | Placeholder |
| Universal NOT | ¬ | ⚠️ | Placeholder |

---

## Summary Statistics

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Native Support | ~40 gates | ~45% |
| 🔧 Custom Implementation | ~10 gates | ~11% |
| ⚠️ Placeholder | ~30 gates | ~34% |
| 📝 Comment/Special | ~10 gates | ~10% |

---

## Notes

### Placeholders
Placeholder gates generate valid, runnable Qiskit code but do not perform actual quantum operations. They appear as labeled barriers in the circuit visualization. This is intentional for:
- **Modeling purposes** - Users can design complete algorithms even if some gates aren't implemented
- **Future extensibility** - Implementations can be added later
- **Documentation** - The circuit clearly shows what operations are intended

### Custom Implementations
Helper classes (like `Increment`, `ReverseBits`, etc.) are **only included in the generated code when used**. This keeps simple circuits clean and minimal.

### Why Not All Gates?
The BESSER Quantum Editor is **framework-agnostic** and supports gates from multiple paradigms:
- Standard quantum computing (Qiskit, Cirq, etc.)
- Quirk simulator patterns
- Theoretical/educational gates
- Algorithm-specific operations

Not all gates have direct Qiskit equivalents, but the modeling capability remains complete.
