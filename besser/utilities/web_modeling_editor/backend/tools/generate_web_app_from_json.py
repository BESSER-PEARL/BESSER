"""Generate a web app from a BESSER Web Modeling Editor JSON export.

Usage:
    python -m besser.utilities.web_modeling_editor.backend.tools.generate_web_app_from_json <path.json>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from besser.generators.web_app import WebAppGenerator
from besser.utilities.web_modeling_editor.backend.services.converters import (
    process_agent_diagram,
    process_class_diagram,
    process_gui_diagram,
)


def load_project_data(json_path: Path) -> Dict[str, Any]:
    """Load a WME export and return the project payload."""
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "project" in data:
        return data["project"]
    if isinstance(data, dict):
        return data
    raise ValueError("Expected a JSON object with a 'project' key or top-level project payload.")


def _get_diagram(diagrams: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    diagram = diagrams.get(key)
    if diagram is None:
        return None
    if isinstance(diagram, dict):
        return diagram
    try:
        return diagram.model_dump()
    except AttributeError:
        return diagram


def generate_web_app(json_path: Path, output_dir: Path, include_agent: bool = True, verbose: bool = False) -> Path:
    project = load_project_data(json_path)
    diagrams = project.get("diagrams", {}) if isinstance(project, dict) else {}

    class_diagram = _get_diagram(diagrams, "ClassDiagram")
    gui_diagram = _get_diagram(diagrams, "GUINoCodeDiagram")

    if not class_diagram or not gui_diagram:
        available = ", ".join(sorted(diagrams.keys())) if diagrams else "none"
        raise ValueError(
            "Missing ClassDiagram or GUINoCodeDiagram in the JSON export. "
            f"Available diagrams: {available}."
        )

    domain_model = process_class_diagram(class_diagram)
    gui_model = process_gui_diagram(
        gui_diagram.get("model", {}),
        class_diagram.get("model", {}),
        domain_model,
    )

    agent_model = None
    if include_agent:
        agent_diagram = _get_diagram(diagrams, "AgentDiagram")
        if agent_diagram:
            if verbose:
                print("Detected AgentDiagram. Generating agent output.")
            agent_model = process_agent_diagram(agent_diagram)

    output_dir.mkdir(parents=True, exist_ok=True)
    generator = WebAppGenerator(
        domain_model,
        gui_model,
        output_dir=str(output_dir),
        agent_model=agent_model,
    )
    generator.generate()
    return output_dir


def build_default_output_dir(base_dir: Path, json_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = json_path.stem.replace(" ", "_")
    return base_dir / f"web_app_{stem}_{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a web app from a BESSER editor JSON export."
    )
    parser.add_argument("json_path", help="Path to the editor JSON export.")
    parser.add_argument(
        "--out",
        dest="output_dir",
        help="Output directory for the generated web app.",
    )
    parser.add_argument(
        "--base-dir",
        default="generated",
        help="Base output directory when --out is not provided (default: generated).",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Skip AgentDiagram processing even if present.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress details.",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = build_default_output_dir(Path(args.base_dir).resolve(), json_path)

    generated_path = generate_web_app(
        json_path,
        output_dir,
        include_agent=not args.no_agent,
        verbose=args.verbose,
    )
    print(f"Web app generated at: {generated_path}")


if __name__ == "__main__":
    main()
