import os
from besser.BUML.notations.sourceCode_to_buml.refactoring_model \
    import RefactoredModelGenerator
from besser.BUML.notations.sourceCode_to_buml.utilities import get_file_name


def detect_keyworld(svg_code_file_path: str) -> str:
    """
    Detect whether the file contains ```xml or ```svg.
    Defaults to 'xml' if nothing is found.
    """
    try:
        with open(svg_code_file_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
            if "```svg" in content:
                return "svg"
            elif "```xml" in content:
                return "xml"
    except Exception as e:
        print(f"⚠️ Could not detect keyworld automatically: {e}")

    return "xml"  # fallback

def refactor_svg_code(svg_code_file_path: str, output_folder: str):
    """
    Refactor a single SVG code file.

    Args:
        svg_code_file_path (str): Path to the input SVG file.

    Returns:
        str: Path to the refactored SVG file.
    """

    svg_output_dir=os.path.join(output_folder, "svg")

    if not os.path.exists(svg_code_file_path):
        raise FileNotFoundError(f"❌ SVG file does not exist: {svg_code_file_path}")

    expected_svg_file = f"enhanced_{get_file_name(svg_code_file_path)}.svg"
    expected_svg_path = os.path.join(svg_output_dir, expected_svg_file)

    # 🔍 Detect whether fenced block is ```xml or ```svg
    keyworld = detect_keyworld(svg_code_file_path)

    gui_generator = RefactoredModelGenerator(
        output_dir=svg_output_dir,
        output_file_name=expected_svg_file,
        structure_file_path=None,  # None means auto structure detection
        code_file=svg_code_file_path,
        keyworld=keyworld
    )

    try:
        gui_generator.generate()
        print(f"Refactored SVG saved as: {expected_svg_path}")
        return expected_svg_path
    except Exception as e:
        print(f"❌ Failed to refactor {svg_code_file_path}: {e}")
        return None

def refactor_all_in_dir(svg_code_dir: str, svg_output_dir: str):
    """
    Refactor all .svg or .xml code files in the given directory.
    """
    if not os.path.isdir(svg_code_dir):
        raise NotADirectoryError(f"❌ Not a valid directory: {svg_code_dir}")

    processed_files = []
    for file_name in os.listdir(svg_code_dir):
        if file_name.lower().endswith((".svg", ".xml")):
            file_path = os.path.join(svg_code_dir, file_name)
            result = refactor_svg_code(file_path, svg_output_dir)
            if result:
                processed_files.append(result)

    print(f"\n✅ Completed processing {len(processed_files)} files.")
    return processed_files
