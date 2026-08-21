"""Paso 2 del pipeline BUML → Alloy.

Toma un archivo .als generado por BESSER y ejecuta Alloy Analyzer
para producir el archivo instance_0.xml en el directorio output/.

Uso:
    python alloy_to_xml.py output/team3.als
    python alloy_to_xml.py output/team3.als --alloy-jar /ruta/a/org.alloytools.alloy.dist.jar
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Nombre del JAR por defecto (buscado en el mismo directorio que este script)
DEFAULT_JAR_NAME = "org.alloytools.alloy.dist.jar"


def find_alloy_jar(jar_arg: str | None = None) -> Path | None:
    """
    Busca el JAR de Alloy en este orden:
      1. Ruta explícita pasada por parámetro
      2. Mismo directorio que este script
    """
    if jar_arg:
        path = Path(jar_arg).resolve()
        if path.exists():
            return path
        print(f"✗ No se encontró el JAR indicado: {jar_arg}")
        return None

    # Directorio donde está este script
    script_dir = Path(__file__).parent.resolve()
    default_path = script_dir / DEFAULT_JAR_NAME
    if default_path.exists():
        return default_path

    print(f"✗ No se encontró {DEFAULT_JAR_NAME} en {script_dir}")
    print("  Especificá la ruta con --alloy-jar <ruta>")
    return None


def ensure_run_command(als_path: Path) -> Path:
    """
    Verifica que el .als tenga un comando 'run'.
    Si no tiene ninguno, agrega 'run {} for 5' al final
    y devuelve la ruta a un archivo temporal con ese añadido.
    """
    content = als_path.read_text(encoding="utf-8")

    # Buscar si ya existe algún 'run' o 'check'
    lines = content.splitlines()
    has_run = any(
        line.strip().startswith("run") or line.strip().startswith("check")
        for line in lines
    )

    if has_run:
        print("  El .als ya contiene un comando 'run' o 'check'.")
        return als_path

    # Agregar run por defecto
    print("  El .als no tiene comando 'run'. Agregando: run {} for 5")
    patched_content = content + "\nrun {} for 5\n"
    patched_path = als_path.parent / (als_path.stem + "_patched.als")
    patched_path.write_text(patched_content, encoding="utf-8")
    print(f"  Archivo parcheado: {patched_path}")
    return patched_path


def run_alloy(als_file: str, alloy_jar: str | None = None) -> bool:
    als_path = Path(als_file).resolve()
    current_working_dir = Path.cwd()

    if not als_path.exists():
        print(f"✗ No se encontró el archivo .als: {als_file}")
        return False

    jar_path = find_alloy_jar(alloy_jar)
    if jar_path is None:
        return False

    output_dir = current_working_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_output = output_dir / "instance_0.xml"

    als_to_run = ensure_run_command(als_path, output_dir)

    print(f"Ejecutando Alloy sobre: {als_to_run}")
    cmd = ["java", "-jar", str(jar_path), "exec", str(als_to_run)]
    result = _run_cmd(cmd, cwd=output_dir)

    # --- DETECCIÓN DE UNSAT ---
    combined_output = (result.stdout or "") + (result.stderr or "")
    unsat_keywords = [
        "No instance found",
        "unsatisfiable",
        "No counterexample found",   # para comandos 'check'
        "Predicate may be inconsistent",
    ]
    if any(kw.lower() in combined_output.lower() for kw in unsat_keywords):
        print("\n⚠ UNSAT: Alloy no encontró instancias satisfacibles.")
        print(f"  Mensaje de Alloy: {combined_output.strip()}")
        # Crear un archivo de reporte en lugar del XML
        unsat_report = output_dir / "unsat_report.txt"
        unsat_report.write_text(
            f"RESULTADO: UNSAT\n"
            f"Archivo: {als_to_run}\n"
            f"Salida de Alloy:\n{combined_output.strip()}\n",
            encoding="utf-8"
        )
        print(f"  Reporte guardado en: {unsat_report}")
        return False

    # Renombrar si fue generado con otro nombre
    generated_xml = list(output_dir.glob("*.xml"))
    if generated_xml and not xml_output.exists():
        generated_xml[0].rename(xml_output)

    if not xml_output.exists():
        print("\n✗ No se generó el archivo XML (error desconocido)")
        print(f"  Salida de Alloy: {combined_output.strip()}")
        return False

    print(f"\n✓ Instancia generada: {xml_output}")
    return True


def _run_cmd(cmd: list, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print(f"  Comando: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=cwd, check=False)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print("── stderr ──")
        print(result.stderr.strip())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paso 2: ejecuta Alloy y genera instance_0.xml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python alloy_to_xml.py output/team3.als
  python alloy_to_xml.py output/team3.als --alloy-jar /ruta/a/org.alloytools.alloy.dist.jar
        """
    )
    parser.add_argument("als_file", help="Archivo .als a ejecutar")
    parser.add_argument(
        "--alloy-jar",
        help=f"Ruta al JAR de Alloy (por defecto busca {DEFAULT_JAR_NAME} junto a este script)"
    )

    args = parser.parse_args()

    success = run_alloy(args.als_file, args.alloy_jar)
    sys.exit(0 if success else 1)
