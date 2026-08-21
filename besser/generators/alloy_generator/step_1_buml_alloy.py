"""Paso 1 del pipeline BUML → Alloy."""

import os
import subprocess
import sys
from pathlib import Path


def generate_alloy(input_model: str) -> bool:
    model_path = Path(input_model).resolve()

    if not model_path.exists():
        print(f"✗ File not foundo: {input_model}")
        return False

    model_dir = model_path.parent
    model_name = model_path.stem

    original_dir = Path.cwd()
    os.chdir(model_dir)

    result = subprocess.run(
        [sys.executable, model_path.name],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    os.chdir(original_dir)

    # Mostrar salida para diagnóstico
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print("── stderr ──")
        print(result.stderr.strip())

    if result.returncode != 0:
        print(f"\n Error: {result.returncode}")
        return False

    # Buscar el .als generado por BESSER
    als_file = _find_als(model_dir, model_name)

    
    print(f"Alloy model generated: {als_file}")
    return True


def _find_als(model_dir: Path, model_name: str) -> Path | None:
    candidates = [
        model_dir / "output" / f"{model_name}.als",
        model_dir / f"{model_name}.als",
        model_dir / f"{model_name}_spec.als",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Búsqueda recursiva: el .als más reciente dentro del directorio
    als_files = sorted(model_dir.rglob("*.als"), key=lambda p: p.stat().st_mtime, reverse=True)
    if als_files:
        return als_files[0]

    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python buml_to_alloy.py <modelo.py>")
        sys.exit(1)

    success = generate_alloy(sys.argv[1])
    sys.exit(0 if success else 1)