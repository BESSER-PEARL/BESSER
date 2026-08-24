import os
import shutil
import subprocess
import sys
from pathlib import Path

# Directorio donde reside este script (alloygenerator/)
PIPELINE_DIR = Path(__file__).resolve().parent

def run_step(command, description, cwd=None):
    print(f"===> {description}")
    try:
        subprocess.run(command, check=True, cwd=cwd or PIPELINE_DIR)
        print("Ok.\n")
    except subprocess.CalledProcessError:
        print(f"Error en: {description}")
        sys.exit(1)

def main():
    if len(sys.argv) < 4:
        print("Error: Faltan parámetros.")
        print("Uso: python pipeline.py <modelo.py> <alloy.jar> <num_instancias>")
        sys.exit(1)

    # 1. Rutas absolutas de los inputs
    modelo_buml = Path(sys.argv[1]).resolve()
    ruta_jar    = Path(sys.argv[2]).resolve()
    i           = int(sys.argv[3])

    nombre_base = modelo_buml.stem

    # 2. Directorio del modelo (donde está dataset.py, etc.)
    directorio_modelo = modelo_buml.parent

    # 3. El .als lo genera step_1 en <directorio_modelo>/output/
    directorio_output_modelo = directorio_modelo / "output"
    archivo_als = directorio_output_modelo / "model.als"

    # 4. Alloy genera los XMLs intermedios en una carpeta <nombre>_spec/
    #    junto al .als, es decir, dentro de directorio_output_modelo
    directorio_spec = directorio_output_modelo / f"{nombre_base}_spec"

    # 5. Output final: subcarpeta output/ relativa al modelo (Dataset/output/)
    directorio_final = directorio_modelo / "output"

    # Limpieza de XMLs intermedios anteriores
    shutil.rmtree(directorio_spec, ignore_errors=True)
    os.makedirs(directorio_final, exist_ok=True)

    # Paso 1: BUML → Alloy (.als)
    run_step(
        [sys.executable, str(PIPELINE_DIR / "step_1_buml_alloy.py"), str(modelo_buml)],
        "Transformando BUML → Alloy..."
    )

    # Paso 2: Alloy Analyzer → XMLs de instancias
    # Ejecutamos con cwd=directorio_output_modelo para que Alloy escriba los XMLs allí
    comando_java = [
        "java", "-jar", str(ruta_jar),
        "exec", "-t", "xml", "-r", str(i),
        str(archivo_als)
    ]
    run_step(
        comando_java,
        "Buscando instancias con Alloy Analyzer...",
        cwd=directorio_output_modelo   # <-- clave: Alloy escribe XMLs aquí
    )

    # Paso 3: XMLs → BUML (diagramas de objetos)
    for n in range(i):
        xml_file   = directorio_spec / f"instance_model-solution-{n}.xml"
        output_py  = directorio_final / f"{nombre_base}_do_{n}.py"

        run_step(
            [sys.executable, str(PIPELINE_DIR / "step_3_alloy_to_buml.py"),
             str(modelo_buml), str(xml_file), str(output_py)],
            f"Generando diagrama de objetos BUML {n+1}/{i}..."
        )

    # Limpieza de XMLs intermedios
    shutil.rmtree(directorio_spec, ignore_errors=True)

if __name__ == "__main__":
    main()